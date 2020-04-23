# AlphAction

AlphAction aims to detect the actions of multiple persons in videos. It is 
**the first open-source project that achieves 30+ mAP (32.4 mAP) with single 
model on AVA dataset.** 

This project is the official implementation of paper 
[Asynchronous Interaction Aggregation for Action Detection](https://arxiv.org/abs/2004.07485), authored
by Jiajun Tang*, Jin Xia* (equal contribution), Xinzhi Mu, Bo Pang, 
[Cewu Lu](http://mvig.sjtu.edu.cn/) (corresponding author). 

<br/>
<div align="center">
  <img  src="gifs/demo1.gif" height=320 alt="demo1">
  <img  src="gifs/demo2.gif" height=320 alt="demo2">
</div>
<div align="center">
  <img  src="gifs/demo3.gif" width=836 alt="demo3">
</div>
<br/>

## Installation 

You need first to install this project, please check [INSTALL.md](INSTALL.md)


## Data Preparation

To do training or inference on AVA dataset, please check [DATA.md](DATA.md)
for data preparation instructions.

## Model Zoo

| config | backbone | structure | mAP | in paper | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [resnet50_4x16f_parallel](config_files/resnet50_4x16f_parallel.yaml) | ResNet-50 | Parallel | 29.0 | 28.9 | [[link]](https://drive.google.com/open?id=13iDNnkxjDqo8OuEhnHFe3P-fERHTbFaD) |
| [resnet50_4x16f_serial](config_files/resnet50_4x16f_serial.yaml) | ResNet-50 | Serial | 29.8 | 29.6 | [[link]](https://drive.google.com/open?id=1S6NIPQ8NoZpzOKkHjzdpFVOtsU6GjqIv) |
| [resnet50_4x16f_denseserial](config_files/resnet50_4x16f_denseserial.yaml) | ResNet-50 | Dense Serial | 30.0 | 29.8 | [[link]](https://drive.google.com/open?id=1OZmlA6V6XoWEA_usyijUREOYujzYL_kP) | 
| [resnet101_8x8f_denseserial](config_files/resnet101_8x8f_denseserial.yaml) | ResNet-101 | Dense Serial | 32.4 | 32.3 | [[link]](https://drive.google.com/open?id=1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x) |

## Visual Demo

To run the demo program on video or webcam, please check the folder [demo](demo).
We select 15 common categories from the 80 action categories of AVA, and 
provide a practical model which achieves high accuracy (about 70 mAP) on these categories. 

## Training and Inference

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

The pre-trained model weights and the training code will be public 
available later. :wink:

### Inference

First, you need to download the model weights from [Model Zoo](#model-zoo).

To do inference on single GPU, you only need to run the following command.
It will load the model from the path speicified in `MODEL.WEIGHT`.
Note that the config `VIDEOS_PER_BATCH` is a global config, if you face
OOM error, you could overwrite the config in the command line as we do 
in below command.
 ```shell
python test_net.py --config-file "path/to/config/file.yaml" \
MODEL.WEIGHT "path/to/model/weight" \
TEST.VIDEOS_PER_BATCH 4
 ```

We use the launch utility `torch.distributed.launch` to launch multiple 
processes for inference on multiple GPUs. `GPU_NUM` should be
replaced by the number of gpus to use. Hyper-parameters in the config file
can still be modified in the way used in single-GPU inference.
 
 ```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM \
test_net.py --config-file "path/to/config/file.yaml" \
MODEL.WEIGHT "path/to/model/weight"
 ```

## Acknowledgement
We thankfully acknowledge the computing resource support of Huawei Corporation
for this project. 

## Citation

If this project helps you in your research or project, please cite
this paper:

```
@article{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  journal={arXiv preprint arXiv:2004.07485},
  year={2020}
}
```