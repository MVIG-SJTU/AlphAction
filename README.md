# AlphAction

AlphAction aims to detect the actions of multiple persons in videos. It is 
**the first open-source project that achieves 30+ mAP (32.4 mAP) with single 
model on AVA dataset.** 

This project is the official implementation of paper 
[Asynchronous Interaction Aggregation for Action Detection](https://arxiv.org/abs/2004.07485) (**ECCV 2020**), authored
by Jiajun Tang*, Jin Xia* (equal contribution), Xinzhi Mu, [Bo Pang](https://bopang1996.github.io/), 
[Cewu Lu](http://mvig.sjtu.edu.cn/) (corresponding author). 

<br/>
<div align="center">
  <img  src="https://user-images.githubusercontent.com/22748802/94115535-71fc9580-fe7c-11ea-98af-d8e9a8a2de82.gif" width="410" alt="demo1">
  <img  src="https://user-images.githubusercontent.com/22748802/94115605-8ccf0a00-fe7c-11ea-8855-ab84232612a0.gif" width="410" alt="demo2">
</div>
<div align="center">
  <img  src="https://user-images.githubusercontent.com/22748802/94115715-b12ae680-fe7c-11ea-8180-8e3d7f57a4bb.gif" alt="demo3">
</div>
<br/>

## Demo Video

[![AlphAction demo video](https://user-images.githubusercontent.com/22748802/94115680-a83a1500-fe7c-11ea-878c-536db277fba7.jpg)](https://www.youtube.com/watch?v=TdGmbOJ9hoE "AlphAction demo video")
[[YouTube]](https://www.youtube.com/watch?v=TdGmbOJ9hoE) [[BiliBili]](https://www.bilibili.com/video/BV14A411J7Xv)

## Installation 

You need first to install this project, please check [INSTALL.md](INSTALL.md)

## Data Preparation

To do training or inference on AVA dataset, please check [DATA.md](DATA.md)
for data preparation instructions. If you have difficulty accessing Google Drive, you can instead find most files (including models) on Baidu NetDisk([[link]](https://pan.baidu.com/s/1MmYiZ4Vyeznke5_3L4WjYw), code: `smti`).

## Model Zoo

Please see [MODEL_ZOO.md](MODEL_ZOO.md) for downloading models.

## Training and Inference

To do training or inference with AlphAction, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Demo Program

To run the demo program on video or webcam, please check the folder [demo](demo).
We select 15 common categories from the 80 action categories of AVA, and 
provide a practical model which achieves high accuracy (about 70 mAP) on these categories. 

## Acknowledgement
We thankfully acknowledge the computing resource support of Huawei Corporation
for this project. 

## Citation

If this project helps you in your research or project, please cite
this paper:

```
@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```
