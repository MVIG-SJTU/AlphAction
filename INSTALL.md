## Installation

**Requirements**

- Python >= 3.5
- [Pytorch](https://pytorch.org/) >= 1.3.0
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
conda create -n action_det python=3.7
conda activate action_det

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install av -c conda-forge
conda install tqdm

pip install yacs opencv-python tensorboardX

######################
# dependicies for demo
conda install Cython SciPy matplotlib
pip install cython-bbox easydict
######################

git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction
python setup.py build develop
```