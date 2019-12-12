import math
import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import norm
import cmath
import itertools
import logging

log = logging.getLogger(__name__)


def detect_lines_and_estimate_empty_ratio(img):

    ext_lines_img, line_points_list = detect_line_segments(img)
    pt0 = get_cross_point(ext_lines_img)
    depth_line_points_list = extract_depth_line_segments(line_points_list, pt0)
    depth_line_points_list = connect_line_segments(depth_line_points_list, pt0)
    q_depth_line_points_list = extract_q_line_segments(depth_line_points_list, pt0)
    container_box = estimate_container_box(q_depth_line_points_list)
    empty_ratio_dict = estimate_empty_area_ratio(
        q_depth_line_points_list, container_box, pt0
    )
    vis_depth_line_img = visualize_depth_line_img(
        img, q_depth_line_points_list, container_box
    )
    return empty_ratio_dict, ext_lines_img, vis_depth_line_img


def detect_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    c = s / 2
    min_length = 0.02 * w
    zeros = np.zeros_like(img)
    ext_lines_img = np.zeros_like(img)
    line_points_list = []
    for line in lines:
        pt1, pt2 = list_to_arrays(line, c)
        pt1, pt3 = extend(pt1, pt2, scale=-3)
        if min_length < norm(pt1 - pt2) < 0.8 * norm(pt1 - pt2, ord=1):
            line_img = cv2.line(
                zeros, pt1=tuple(pt1), pt2=tuple(pt3), color=1, thickness=5
            )
            ext_lines_img += line_img
            line_points_list.append((pt1, pt2))
    ext_lines_img = cv2.GaussianBlur(ext_lines_img, ksize=(51, 51), sigmaX=51)

    return ext_lines_img, line_points_list


def visualize_lines_img(lines_img, pt0):
    max_val = int(np.max(lines_img))
    lines_img = cv2.line(
        lines_img, pt1=tuple(pt0), pt2=tuple(pt0), color=max_val * 2, thickness=10
    )
    lines_img = (255 * (lines_img / np.max(lines_img, keepdims=True))).astype(np.uint8)

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


def get_cross_point(a):
    return argsmax(a)[::-1]


def extract_depth_line_segments(line_points_list, pt0):
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
    return depth_line_points_list


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


def extract_q_line_segments(ls_list, pt0, flat=False):
    ls_list.sort(key=get_line_length, reverse=True)
    top_left_ls_list = []
    top_right_ls_list = []
    bottom_left_ls_list = []
    bottom_right_ls_list = []
    for points in ls_list:
        if points[0][1] < pt0[1]:
            if points[0][0] < pt0[0]:
                top_left_ls_list.append(points)
            else:
                top_right_ls_list.append(points)
        else:
            if points[0][0] < pt0[0]:
                bottom_left_ls_list.append(points)
            else:
                bottom_right_ls_list.append(points)

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
        if flat:
            ls_list_out.extend(q)
        else:
            ls_list_out.append(q)
    return ls_list_out


def estimate_container_box(ls_list):
    ls_list = flatten(ls_list)
    pt2_x_list = [pt2[0] for _, pt2 in ls_list]
    pt2_y_list = [pt2[1] for _, pt2 in ls_list]

    pt2_x_list.sort()
    pt2_y_list.sort()

    x_min = int(np.mean(np.array(pt2_x_list[:2])))
    y_min = int(np.mean(np.array(pt2_y_list[:2])))
    x_max = int(np.mean(np.array(pt2_x_list[-2:])))
    y_max = int(np.mean(np.array(pt2_y_list[-2:])))
    return x_min, y_min, x_max, y_max


def visualize_depth_line_img(img, q_depth_line_points_list=None, container_box=None):
    depth_line_img = np.zeros_like(img)
    if q_depth_line_points_list is not None:
        depth_line_points_list = flatten(q_depth_line_points_list)
        for depth_line_points in depth_line_points_list:
            pt1, pt2 = depth_line_points
            depth_line_img = cv2.line(
                depth_line_img, pt1=tuple(pt1), pt2=tuple(pt2), color=255, thickness=1
            )
    if container_box is not None:
        x_min, y_min, x_max, y_max = container_box
        edges = [
            dict(pt1=(x_min, y_min), pt2=(x_min, y_max)),
            dict(pt1=(x_min, y_max), pt2=(x_max, y_max)),
            dict(pt1=(x_max, y_max), pt2=(x_max, y_min)),
            dict(pt1=(x_max, y_min), pt2=(x_min, y_min)),
        ]
        for edge in edges:
            depth_line_img = cv2.line(depth_line_img, color=127, thickness=1, **edge)

    return depth_line_img


def estimate_empty_area_ratio(q_depth_line_points_list, container_box, pt0):
    empty_ratio_dict = dict(empty_ratio=0, left_empty_ratio=0, right_empty_ratio=0)
    x_min, y_min, x_max, y_max = container_box

    top_depth_line_points_list = (
        q_depth_line_points_list[0] + q_depth_line_points_list[1]
    )
    if top_depth_line_points_list:
        top_y = max([pt1[1] for pt1, _ in top_depth_line_points_list])
        top_ratio = (top_y - y_min) / (pt0[1] - y_min)

        bottom_left_line_points_list = q_depth_line_points_list[2]
        if bottom_left_line_points_list:
            bottom_left_x = max([pt1[0] for pt1, _ in bottom_left_line_points_list])
            bottom_left_ratio = max(0.0, (bottom_left_x - x_min) / (pt0[0] - x_min))
            left_empty_ratio = min(1.0, bottom_left_ratio / top_ratio)
            empty_ratio_dict["left_empty_ratio"] = left_empty_ratio

        bottom_right_line_points_list = q_depth_line_points_list[3]
        if bottom_right_line_points_list:
            bottom_right_x = min([pt1[0] for pt1, _ in bottom_right_line_points_list])
            bottom_right_ratio = max(0.0, (x_max - bottom_right_x) / (x_max - pt0[0]))
            right_empty_ratio = min(1.0, bottom_right_ratio / top_ratio)
            empty_ratio_dict["right_empty_ratio"] = right_empty_ratio

    empty_ratio_dict["empty_ratio"] = (
        empty_ratio_dict["left_empty_ratio"] + empty_ratio_dict["right_empty_ratio"]
    ) / 2

    return empty_ratio_dict


def flatten(ls):
    return list(itertools.chain.from_iterable(ls))
