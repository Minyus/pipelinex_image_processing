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
    zeros = np.zeros_like(img)
    lines_img = np.zeros_like(img)
    for line in lines:
        pt1, pt2 = list_to_arrays(line, c)
        pt1, pt2 = extend(pt1, pt2, scale=-3)
        if min_length < norm(pt1 - pt2) < 0.8 * norm(pt1 - pt2, ord=1):
            line_img = cv2.line(
                zeros, pt1=tuple(pt1), pt2=tuple(pt2), color=1, thickness=5
            )
            lines_img += line_img
    lines_img = (255 * (lines_img / np.max(lines_img, keepdims=True))).astype(np.uint8)
    cross_pt = argsmax(lines_img)
    return lines_img


def list_to_arrays(line, c):
    x1, y1, x2, y2 = map(int, line[:4])

    pt1 = np.array([x1, y1])
    pt2 = np.array([x2, y2])
    if norm(pt1 - c) > norm(pt2 - c):
        pt1, pt2 = pt2, pt1
    return pt1, pt2


def extend(pt1, pt2, scale=1):
    pt2 = scale * (pt2 - pt1) + pt1
    return pt1, pt2


def argsmax(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)
