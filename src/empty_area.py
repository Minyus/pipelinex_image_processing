import math

import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import norm
import cmath


def overlay_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    c = s / 2
    min_length = 0.02 * w
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

    depth_line_points_list = connect_line_segments(depth_line_points_list, pt0)
    depth_line_points_list = filter_line_segments(depth_line_points_list, pt0)

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


def connect_line_segments(depth_line_points_list, pt0):

    n_lines = len(depth_line_points_list)

    polar_pt1_list = [
        cmath.polar(complex(*tuple(pt1 - pt0))) for pt1, _ in depth_line_points_list
    ]
    polar_pt2_list = [
        cmath.polar(complex(*tuple(pt2 - pt0))) for _, pt2 in depth_line_points_list
    ]

    connected_line_points_list = []
    connected_index_list = []
    for i in range(n_lines):
        for j in range(n_lines):
            polar_pt1 = polar_pt1_list[i]
            polar_pt2 = polar_pt2_list[j]
            if (
                (i != j)
                and (0 < (polar_pt1[0] - polar_pt2[0]) < 100)
                and (abs(polar_pt2[1] - polar_pt1[1]) < (np.pi / 180))
            ):
                points = (depth_line_points_list[j][0], depth_line_points_list[i][1])
                connected_line_points_list.append(points)
                connected_index_list.extend([i, j])

    reduced_line_points_list = [
        depth_line_points_list[i]
        for i in range(n_lines)
        if i not in connected_index_list
    ]

    combined_line_points_list = reduced_line_points_list + connected_line_points_list
    return combined_line_points_list
    # return depth_line_points_list


def get_line_length(t):
    return norm(t[0] - t[1])


def filter_line_segments(ls_list, pt0):
    ls_list.sort(key=get_line_length, reverse=True)
    top_left_ls_list = []
    top_right_ls_list = []
    bottom_left_ls_list = []
    bottom_right_ls_list = []
    for points in ls_list:
        if points[0][1] < pt0[1]:
            if points[0][0] < pt0[0]:
                bottom_left_ls_list.append(points)
            else:
                bottom_right_ls_list.append(points)
        else:
            if points[0][0] < pt0[0]:
                top_left_ls_list.append(points)
            else:
                top_right_ls_list.append(points)

    q_ls_list = [
        top_left_ls_list,
        top_right_ls_list,
        bottom_left_ls_list,
        bottom_right_ls_list,
    ]

    ls_list_out = []
    for q in q_ls_list:
        if len(q) >= 3:
            q = q[: int(math.ceil(len(q) * 0.5))]
        ls_list_out.extend(q)
    return ls_list_out
