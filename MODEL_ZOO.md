## AlphAction Model Zoo

### Pre-trained Models

We provide backbone models pre-trained on Kinetics dataset, used for further
fine-tuning on AVA dataset. The reported accuracy are obtained by 30-view testing.

| backbone | pre-train | frame length | sample rate | top-1 | top-5 | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast-R50 | Kinetics-700 | 4 | 16 | 66.34 | 86.66 | [[link]](https://drive.google.com/file/d/1bNcF295jxY4Zbqf0mdtsw9QifpXnvOyh/view?usp=sharing) |
| SlowFast-R101 | Kinetics-700 | 8 | 8 | 69.32 | 88.84 | [[link]](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) |

### AVA Models

| config | backbone | IA structure | mAP | in paper | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [resnet50_4x16f_baseline](config_files/resnet50_4x16f_baseline.yaml) | SlowFast-R50-4x16 | w/o | 26.7 | 26.5 | [[link]](https://drive.google.com/file/d/1_yAxk6R58Dn6IjBCx-WBwCQEf_582Vuv/view?usp=sharing) |
| [resnet50_4x16f_parallel](config_files/resnet50_4x16f_parallel.yaml) | SlowFast-R50-4x16 | Parallel | 29.0 | 28.9 | [[link]](https://drive.google.com/file/d/13iDNnkxjDqo8OuEhnHFe3P-fERHTbFaD/view?usp=sharing) |
| [resnet50_4x16f_serial](config_files/resnet50_4x16f_serial.yaml) | SlowFast-R50-4x16 | Serial | 29.8 | 29.6 | [[link]](https://drive.google.com/file/d/1S6NIPQ8NoZpzOKkHjzdpFVOtsU6GjqIv/view?usp=sharing) |
| [resnet50_4x16f_denseserial](config_files/resnet50_4x16f_denseserial.yaml) | SlowFast-R50-4x16 | Dense Serial | 30.0 | 29.8 | [[link]](https://drive.google.com/file/d/1OZmlA6V6XoWEA_usyijUREOYujzYL_kP/view?usp=sharing) | 
| [resnet101_8x8f_baseline](config_files/resnet101_8x8f_baseline.yaml) | SlowFast-R101-8x8 | w/o | 29.3 | 29.3 | [[link]](https://drive.google.com/file/d/1GC56oNEX00oEH8aiGYdFMKENdWf2VvAY/view?usp=sharing) |
| [resnet101_8x8f_denseserial](config_files/resnet101_8x8f_denseserial.yaml) | SlowFast-R101-8x8 | Dense Serial | 32.4 | 32.3 | [[link]](https://drive.google.com/file/d/1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x/view?usp=sharing) |
