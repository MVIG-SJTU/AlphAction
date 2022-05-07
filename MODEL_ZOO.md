## AlphAction Model Zoo

### Pre-trained Models

We provide backbone models pre-trained on Kinetics dataset, used for further
fine-tuning on AVA dataset. The reported accuracy are obtained by 30-view testing.

| backbone | pre-train | frame length | sample rate | top-1 | top-5 | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast-R50 | Kinetics-700 | 4 | 16 | 66.34 | 86.66 | [[link]](https://drive.google.com/file/d/1hqFuhD1p0lMpl3Yi5paIGY-hlPTVYgyi/view?usp=sharing) |
| SlowFast-R101 | Kinetics-700 | 8 | 8 | 69.32 | 88.84 | [[link]](https://drive.google.com/file/d/1JDQLyyL-GFd3qi0S31Mdt5oNmUXnyJza/view?usp=sharing) |

### AVA Models

| config | backbone | IA structure | mAP | in paper | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [resnet50_4x16f_baseline](config_files/resnet50_4x16f_baseline.yaml) | SlowFast-R50-4x16 | w/o | 26.7 | 26.5 | [[link]](https://drive.google.com/file/d/1HmFVEe_wsOP9WUNdU_W7PkgWsHzZtgDf/view?usp=sharing) |
| [resnet50_4x16f_parallel](config_files/resnet50_4x16f_parallel.yaml) | SlowFast-R50-4x16 | Parallel | 29.0 | 28.9 | [[link]](https://drive.google.com/file/d/1CdgwZk6HQGryBssVhE7E48HZ3CCubBg0/view?usp=sharing) |
| [resnet50_4x16f_serial](config_files/resnet50_4x16f_serial.yaml) | SlowFast-R50-4x16 | Serial | 29.8 | 29.6 | [[link]](https://drive.google.com/file/d/1RTUi_ARCtar1r-u7UaTxCaLyJFnVe9Ro/view?usp=sharing) |
| [resnet50_4x16f_denseserial](config_files/resnet50_4x16f_denseserial.yaml) | SlowFast-R50-4x16 | Dense Serial | 30.0 | 29.8 | [[link]](https://drive.google.com/file/d/1bYxGyf6kptfUBNAHtFcG7x4Ryp7mcWxH/view?usp=sharing) | 
| [resnet101_8x8f_baseline](config_files/resnet101_8x8f_baseline.yaml) | SlowFast-R101-8x8 | w/o | 29.3 | 29.3 | [[link]](https://drive.google.com/file/d/1oVGRV82iIaxm7XJqAXw7AoDTxFtdqvfv/view?usp=sharing) |
| [resnet101_8x8f_denseserial](config_files/resnet101_8x8f_denseserial.yaml) | SlowFast-R101-8x8 | Dense Serial | 32.4 | 32.3 | [[link]](https://drive.google.com/file/d/1yqqc2_X6Ywi165PIuq68NdTs2WwMygHh/view?usp=sharing) |
