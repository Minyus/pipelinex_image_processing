import numpy as np
import cv2
from tensorflow import keras
from keras_segmentation.pretrained import model_from_checkpoint_path


def pspnet_50_ADE_20K():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 150,
        "model_class": "pspnet_50",
    }

    model_url = "https://www.dropbox.com/s/" "0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1"
    latest_weights = keras.utils.get_file("pspnet50_ade20k.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)


def predict(model, img):
    assert len(img.shape) == 3, "Image should be h,w,3 "
    original_shape = img.shape[:2]
    output_width = model.output_width
    output_height = model.output_height
    n_classes = model.n_classes
    resized_img = cv2.resize(
        img, dsize=(output_width, output_height), interpolation=cv2.INTER_AREA
    )
    raw_pred_img = model.predict(np.array([resized_img]))[0]
    resized_pred_img = raw_pred_img.reshape((output_height, output_width, n_classes))

    pred_img = cv2.resize(
        resized_pred_img, dsize=original_shape, interpolation=cv2.INTER_LINEAR
    )
    seg_img = pred_img.argmax(axis=2)
    return seg_img.astype(np.uint8)


def get_semantic_segments(img):
    model = pspnet_50_ADE_20K()
    out = predict(model, img=img)
    return out
