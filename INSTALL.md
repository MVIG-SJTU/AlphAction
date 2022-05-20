## Installation

**Requirements**

- Python >= 3.5
- [Pytorch](https://pytorch.org/) == 1.4.0 （other versions are not tested)
- [PyAV](https://github.com/mikeboers/PyAV) >= 6.2.0
- [yacs](https://github.com/rbgirshick/yacs)
- [OpenCV](https://opencv.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [FFmpeg](https://www.ffmpeg.org/)
- [Cython](https://cython.org/), [cython_bbox](https://github.com/samson-wang/cython_bbox), [SciPy](https://scipy.org/scipylib/), [matplotlib](https://matplotlib.org/), [easydict](https://github.com/makinacorpus/easydict) (for running demo)
- Linux + Nvidia GPUs

We recommend to setup the environment with Anaconda, 
the step-by-step installation script is shown below.

```bash
conda create -n alphaction python=3.7
conda activate alphaction

# install pytorch with the same cuda version as in your environment
cuda_version=$(nvcc --version | grep -oP '(?<=release )[\d\.]*?(?=,)')
conda install pytorch=1.4.0 torchvision cudatoolkit=$cuda_version -c pytorch
# you should check manually if you successfully install pytorch here, there may be no such package for some cuda versions.

conda install av -c conda-forge
conda install cython

git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction
pip install -e .    # Other dependicies will be installed here

```
