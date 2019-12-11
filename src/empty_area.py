import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import norm


def overlay_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    c = s / 2
    min_length = 0.001 * w
    zeros = np.zeros_like(img)
    lines_img = np.zeros_like(img)
    line_points_list = []
    for line in lines:
        pt1, pt2 = list_to_arrays(line, c)
        pt1, pt3 = extend(pt1, pt2, scale=-3)
        if min_length < norm(pt1 - pt2) < 0.8 * norm(pt1 - pt2, ord=1):
            line_img = cv2.line(
                zeros, pt1=tuple(pt1), pt2=tuple(pt3), color=1, thickness=5
            )
            lines_img += line_img
            line_points_list.append((pt1, pt2))
    lines_img = cv2.GaussianBlur(lines_img, ksize=(51, 51), sigmaX=51)
    pt0_yx = argsmax(lines_img)
    pt0 = pt0_yx[::-1]
    max_val = int(np.max(lines_img))
    lines_img = cv2.line(
        lines_img, pt1=tuple(pt0), pt2=tuple(pt0), color=max_val * 2, thickness=10
    )
    lines_img = (255 * (lines_img / np.max(lines_img, keepdims=True))).astype(np.uint8)

    depth_line_points_list = []
    for line_points in line_points_list:
        pt1, pt2 = line_points
        v12 = pt2 - pt1
        nv12 = v12 / norm(v12)
        v01 = pt1 - pt0
        cross_dist = np.abs(nv12[0] * v01[1] - nv12[1] * v01[0])
        depth_flag = cross_dist < 10
        if depth_flag:
            depth_line_points_list.append(line_points)

    depth_line_img = np.zeros_like(img)
    for depth_line_points in depth_line_points_list:
        pt1, pt2 = depth_line_points
        depth_line_img = cv2.line(
            depth_line_img, pt1=tuple(pt1), pt2=tuple(pt2), color=255, thickness=1
        )
    return depth_line_img


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
