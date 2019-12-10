import cv2
from pylsd.lsd import lsd
import numpy as np


def overlay_line_segments(img):

    lines = lsd(img)

    min_length = 0.02 * img.shape[1]
    img_out = 255 * np.ones_like(img)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        if long_enough(x1, y1, x2, y2, min_length):
            img_out = cv2.line(
                img_out, pt1=(x1, y1), pt2=(x2, y2), color=0, thickness=1
            )

    return img_out


def long_enough(x1, y1, x2, y2, min_length):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2 > min_length ** 2
