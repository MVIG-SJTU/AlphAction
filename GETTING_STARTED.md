## Getting Started with AlphAction

The hyper-parameters of each experiment are controlled by 
a .yaml config file, which is located in the directory 
`config_files`. All of these configuration files assume 
that we are running on 8 GPUs. We need to create a symbolic
link to the directory `output`, where the output (logs and checkpoints)
will be saved. Besides, we recommend to create a directory `models` to place 
model weights. These can be done with following commands.

```shell
mkdir -p /path/to/output
ln -s /path/to/output data/output
mkdir -p /path/to/models
ln -s /path/to/models data/models
```

### Training

Download pre-trained models from [MODEL_ZOO.md](MODEL_ZOO.md#pre-trained-models).
Then place pre-trained models in `data/models` directory with following structure:

```
models/
|_ pretrained_models/
|  |_ SlowFast-ResNet50-4x16.pth
|  |_ SlowFast-ResNet101-8x8.pth
```

To train on a single GPU, you only need to run following command. The 
argument `--use-tfboard` enables tensorboard to log training process. 
Because the config files assume that we are using 8 GPUs, the global 
batch size `SOLVER.VIDEOS_PER_BATCH` and `TEST.VIDEOS_PER_BATCH` can
be too large for a single GPU. Therefore, in the following command, we 
modify the batch size and also adjust the learning rate and schedule 
length according to the linear scaling rule.

```shell
python train_net.py --config-file "path/to/config/file.yaml" \
--transfer --no-head --use-tfboard \
SOLVER.BASE_LR 0.000125 \
SOLVER.STEPS '(560000, 720000)' \
SOLVER.MAX_ITER 880000 \ 
SOLVER.VIDEOS_PER_BATCH 2 \
TEST.VIDEOS_PER_BATCH 2
```

We use the launch utility `torch.distributed.launch` to launch multiple 
processes for distributed training on multiple gpus. `GPU_NUM` should be
replaced by the number of gpus to use. Hyper-parameters in the config file
can still be modified in the way used in single-GPU training.

```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM \
train_net.py --config-file "path/to/config/file.yaml" \
--transfer --no-head --use-tfboard
```

### Inference

To do inference on multiple GPUs, you should run the following command. Note that 
our code first trys to load the `last_checkpoint` in the `OUTPUT_DIR`. If there
 is no such file in `OUTPUT_DIR`, it will then load the model from the 
 path specified in `MODEL.WEIGHT`. To use `MODEL.WEIGHT` to do the inference,
 you need to ensure that there is no `last_checkpoint` in `OUTPUT_DIR`. 
 You can download the model weights from [MODEL_ZOO.md](MODEL_ZOO.md#ava-models).
 
 ```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM \
test_net.py --config-file "path/to/config/file.yaml" \
MODEL.WEIGHT "path/to/model/weight"
 ```