# Pipelinex Image Processing

An example project using [PipelineX](https://github.com/Minyus/pipelinex), Kedro, OpenCV, Scikit-image, and TensorFlow/Keras for image processing.

<p align="center">
<img src="img/kedro_pipeline.PNG">
Pipeline visualized by Kedro-viz
</p>

## Directories

- `conf`
  - YAML config files for PipelineX project
- `data`
  - empty folders (output files will be saved here)
- `logs`
  - empty folders (log files will be saved here)
- `src`
  - `empty_area.py`
    - The algorithm to estimate empty area ratio
  - `roi.py`
    - Supplementary algorithm to compute ROI (Region of Interest) from segmentation image
  - `semantic_segmentation.py`
    - Semantic segmentation using PSPNet model pretrained with ADE20K dataset

## How to run the code

### 1. Install Python packages

#### Install tensorflow 1.x and keras-segmentation

```bash
$ pip install tensorflow<2 keras-segmentation
```

##### If you want to use TensorFlow 2.x, install fork of keras-segmentation modified to work with TensorFlow 2.x

```bash
$ pip install tensorflow>=2.0.0
$ pip install git+https://github.com/Minyus/image-segmentation-keras.git
```

#### Install the other packages 

```bash
$ pip install pipelinex opencv-python scikit-image ocrd-fork-pylsd Keras Pillow pandas numpy requests kedro mlflow kedro-viz
```

Note: `mlflow` and `kedro-viz` are optional.

### 2. Clone `https://github.com/Minyus/pipelinex_image_processing.git`

```bash
$ git clone https://github.com/Minyus/pipelinex_image_processing.git
$ cd pipelinex_image_processing
```

### 3. Run `main.py`

```bash
$ python main.py
```

As configured in [catalog.yml](https://github.com/Minyus/pipelinex_image_processing/blob/master/conf/base/catalog.yml), the following 2 images will be downloaded by http requests. 

![Image](https://cdn.foodlogistics.com/files/base/acbm/fl/image/2018/09/960w/GettyImages_485190815.5b9bfb5550ded.jpg)
![Image](https://www.thetrailerconnection.com/zupload/library/180/-1279-840x600-0.jpg)

## Tested environment

- Python 3.6.8
