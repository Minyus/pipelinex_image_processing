# Pipelinex Image Processing

A project to use [PipelineX](https://github.com/Minyus/pipelinex) for image processing. 

## How to run the code

### 1. Install Python packages

#### Install PipelineX

Option 1: PyPI
```bash
$ pip3 install pipelinex==0.1.5
```

Option 2: GitHub repository
```bash
$ pip3 install git+https://github.com/Minyus/pipelinex.git
```

#### Install the other dependencies
```
opencv-python==3.4.5.20
scikit-image==0.16.2
ocrd-fork-pylsd==0.0.3
Pillow==4.3.0
numpy==1.16.4
scipy==1.2.1
kedro==0.15.5
mlflow==1.4.0
pandas==0.25.1
PyYAML==5.2
```

These versions are recommended, but other recent versions will likely work.

### 2. Clone `https://github.com/Minyus/pipelinex_image_processing.git`

### 3. Place input images in `data/input/TRIMG` folder

An example image is at:
https://cdn.foodlogistics.com/files/base/acbm/fl/image/2018/09/960w/GettyImages_485190815.5b9bfb5550ded.jpg

### 4. Run `main.py`

## Recommended environment
- Python 3.6.8 (Python 3.6.x will likely work.)
- Ubuntu 18.04.3 LTS (Recent Linux versions will likely work.)

## Author
Yusuke Minami

- [GitHub](https://github.com/Minyus)
- [Linkedin](https://www.linkedin.com/in/yusukeminami/)
- [Twitter](https://twitter.com/Minyus86)

