r"""
Basic training script for PyTorch
"""

import argparse
import os

import torch
from torch.utils.collect_env import get_pretty_env_info
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
from alphaction.solver import make_lr_scheduler, make_optimizer
from alphaction.engine.inference import inference
from alphaction.engine.trainer import do_train
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.utils.comm import synchronize, get_rank
from alphaction.utils.logger import setup_logger, setup_tblogger
from alphaction.utils.random_seed import set_seed
from alphaction.utils.IA_helper import has_memory
from alphaction.structures.memory_pool import MemoryPool
# pytorch issuse #973
import resource


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))


def train(cfg, local_rank, distributed, tblogger=None, transfer_weight=False, adjust_lr=False, skip_val=False,
          no_head=False):
    # build the model.
    model = build_detection_model(cfg)

    device = torch.device("cuda")
    model.to(device)

    # make solver.
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0
    arguments["person_pool"] = MemoryPool()

    output_dir = cfg.OUTPUT_DIR

    # load weight.
    save_to_disk = get_rank() == 0
    checkpointer = ActionCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, model_weight_only=transfer_weight,
                                              adjust_scheduler=adjust_lr, no_head=no_head)

    arguments.update(extra_checkpoint_data)

    # make dataloader.
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments['iteration'],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    val_period = cfg.SOLVER.EVAL_PERIOD

    mem_active = has_memory(cfg.MODEL.IA_STRUCTURE)

    # make validation dataloader if necessary
    if not skip_val:
        dataset_names_val = cfg.DATASETS.TEST
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        dataset_names_val = []
        data_loaders_val = []
    # training
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tblogger,
        val_period,
        dataset_names_val,
        data_loaders_val,
        distributed,
        mem_active,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder
    # make test dataloader.
    data_loaders_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    # test for each dataset.
    for output_folder, dataset_name, data_loader_test in zip(output_folders, dataset_names, data_loaders_test):
        inference(
            model,
            data_loader_test,
            dataset_name,
            mem_active=has_memory(cfg.MODEL.IA_STRUCTURE),
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Action Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-final-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--skip-val-in-train",
        dest="skip_val",
        help="Do not validate during training",
        action="store_true",
    )
    parser.add_argument(
        "--transfer",
        dest="transfer_weight",
        help="Transfer weight from a pretrained model",
        action="store_true"
    )
    parser.add_argument(
        "--adjust-lr",
        dest="adjust_lr",
        help="Adjust learning rate scheduler from old checkpoint",
        action="store_true"
    )
    parser.add_argument(
        "--no-head",
        dest="no_head",
        help="Not load the head layer parameters from weight file",
        action="store_true"
    )
    parser.add_argument(
        "--use-tfboard",
        action='store_true',
        dest='tfboard',
        help='Use tensorboard to log stats'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Manual seed at the begining."
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    global_rank = get_rank()

    # Merge config.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Print experimental infos.
    logger = setup_logger("alphaction", output_dir, global_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    tblogger = None
    if args.tfboard:
        tblogger = setup_tblogger(output_dir, global_rank)

    set_seed(args.seed, global_rank, num_gpus)

    # do training.
    model = train(cfg, args.local_rank, args.distributed, tblogger, args.transfer_weight, args.adjust_lr, args.skip_val,
                  args.no_head)

    if tblogger is not None:
        tblogger.close()

    # do final testing.
    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
