import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import norm


def overlay_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    c = s / 2
    min_length = 0.02 * w
    img_out = 255 * np.ones_like(img)
    for line in lines:
        pt1, pt2 = list_to_arrays(line, c)
        if norm(pt1 - pt2) > min_length:
            img_out = cv2.line(
                img_out, pt1=tuple(pt1), pt2=tuple(pt2), color=0, thickness=1
            )

    return img_out


def list_to_arrays(line, c):
    x1, y1, x2, y2 = map(int, line[:4])

    pt1 = np.array([x1, y1])
    pt2 = np.array([x2, y2])
    if norm(pt1 - c) > norm(pt2 - c):
        pt1, pt2 = pt2, pt1
    return pt1, pt2
