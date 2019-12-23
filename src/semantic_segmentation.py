import numpy as np
import cv2
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.predict import predict


def predict_segmentation(model, img):
    resized_img = cv2.resize(img, (473, 473))
    resized_out = predict(model, inp=resized_img)
    out = cv2.resize(resized_out, (512, 512), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


def get_semantic_segments(img):
    model = pspnet_50_ADE_20K()
    out = predict_segmentation(model, img)
    return out
