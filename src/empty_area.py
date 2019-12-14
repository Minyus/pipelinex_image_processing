import math
import cv2
from pylsd.lsd import lsd
import numpy as np
from numpy.linalg import norm
import cmath
import itertools
import logging

from scipy.sparse import coo_matrix

log = logging.getLogger(__name__)


def detect_lines_and_estimate_empty_ratio(img, roi):

    intersection_img, line_points_list, tilted_line_points_list = detect_line_segments(
        img
    )
    pt0 = get_vanishing_point(intersection_img)
    depth_line_points_list = extract_depth_line_segments(tilted_line_points_list, pt0)
    depth_line_points_list = connect_line_segments(depth_line_points_list, pt0)
    q_depth_line_points_list = extract_q_line_segments(depth_line_points_list, pt0)
    container_box = estimate_container_box(q_depth_line_points_list, roi, pt0)
    front_ceiling_line_points_list = extract_front_ceiling_line_segments(
        line_points_list, pt0
    )
    empty_ratio_dict = estimate_empty_area_ratio(
        q_depth_line_points_list, container_box, pt0
    )
    report_img = draw_report_img(
        img,
        line_points_list,
        q_depth_line_points_list,
        front_ceiling_line_points_list,
        container_box,
    )
    return empty_ratio_dict, intersection_img, report_img


def detect_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    center_point = s / 2
    min_length = 0.01 * w
    line_points_list = []
    tilted_line_points_list = []
    for line in lines:
        pt1, pt2 = list_to_arrays(line, center_point)
        line_points_list.append((pt1, pt2))
        pt12 = pt2 - pt1
        if (
            min_length < norm(pt12) < 0.9 * norm(pt12, ord=1)
            and norm(pt12, ord=1) < (h + w) / 2
        ):
            tilted_line_points_list.append((pt1, pt2))

    line_angle_list = [
        cmath.phase(complex(*tuple(pt2 - pt1))) for pt1, pt2 in tilted_line_points_list
    ]

    ac_list = []
    for pt1, pt2 in tilted_line_points_list:
        pt12 = pt2 - pt1
        a_ = np.array([pt12[1], -pt12[0]])
        c_ = pt12[1] * pt1[0] - pt12[0] * pt1[1]
        ac_list.append((a_, c_))

    n_lines = len(ac_list)
    intersection_point_list = []
    for i in range(n_lines):
        for j in range(n_lines):
            if (
                i > j
                and 0.05 * np.pi
                < abs(line_angle_list[i] - line_angle_list[j])
                < 0.95 * np.pi
            ):
                a_i, c_i = ac_list[i]
                a_j, c_j = ac_list[j]
                a_2darr = np.stack([a_i, a_j])
                c_arr = np.array([c_i, c_j])
                intersection_point = np.matmul(np.linalg.inv(a_2darr), c_arr)

                dist_pt1_from_center = min(
                    norm(tilted_line_points_list[i][0] - center_point),
                    norm(tilted_line_points_list[j][0] - center_point),
                )
                intersection_close_enough_to_center_flag = (
                    norm(intersection_point - center_point) < 0.8 * dist_pt1_from_center
                )
                if (
                    intersection_close_enough_to_center_flag
                    and (w * 1 / 4) <= intersection_point[0] < (w * 3 / 4)
                    and (h * 1 / 4) <= intersection_point[1] < (h * 3 / 4)
                ):
                    intersection_point_list.append(intersection_point)

    intersection_arr = np.stack(intersection_point_list)
    intersection_x_arr = intersection_arr[:, 0]
    intersection_y_arr = intersection_arr[:, 1]

    center_point_2darr = np.expand_dims(center_point, axis=0)
    vals_arr = (
        np.exp(
            -0.5
            * np.square(norm(intersection_arr - center_point_2darr, axis=1) / (w / 4))
        )
        / intersection_arr.shape[0]
    )

    intersection_coo = coo_matrix(
        (vals_arr, (intersection_y_arr, intersection_x_arr)), shape=img.shape
    )
    intersection_img = intersection_coo.todense()
    intersection_img = cv2.GaussianBlur(intersection_img, ksize=(21, 21), sigmaX=10)
    return intersection_img, line_points_list, tilted_line_points_list


def visualize_lines_img(lines_img):
    pt0 = get_vanishing_point(lines_img)
    lines_img = (127 * (lines_img / np.max(lines_img, keepdims=True))).astype(np.uint8)
    lines_img = cv2.line(
        lines_img, pt1=tuple(pt0), pt2=tuple(pt0), color=255, thickness=7
    )

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


def get_vanishing_point(a):
    return argsmax(a)[::-1]


def extract_depth_line_segments(line_points_list, pt0):
    depth_line_points_list = []
    for line_points in line_points_list:
        pt1, pt2 = line_points
        v12 = pt2 - pt1
        nv12 = v12 / norm(v12)
        v01 = pt1 - pt0
        dist_vanishing_point = np.abs(nv12[0] * v01[1] - nv12[1] * v01[0])
        depth_flag = dist_vanishing_point < 20
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
                and (0 < (polar_pt1[0] - polar_pt2[0]) < 20)
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


def estimate_container_box(q_depth_line_points_list, roi, pt0):
    container_box = tuple(roi)
    ls_list = flatten(q_depth_line_points_list)
    if not ls_list:
        return container_box

    pt2_x_list = [pt2[0] for _, pt2 in ls_list]
    pt2_y_list = [pt2[1] for _, pt2 in ls_list]

    x_lower_list = [v for v in pt2_x_list if v < pt0[0]]
    y_lower_list = [v for v in pt2_y_list if v < pt0[1]]
    x_upper_list = [v for v in pt2_x_list if v > pt0[0]]
    y_upper_list = [v for v in pt2_y_list if v > pt0[1]]

    # if len(x_lower_list) >= 3:
    #     x_lower_list = [v for v in x_lower_list if v < np.mean(x_lower_list)]
    # if len(y_lower_list) >= 3:
    #     y_lower_list = [v for v in y_lower_list if v < np.mean(y_lower_list)]
    # if len(x_upper_list) >= 3:
    #     x_upper_list = [v for v in x_upper_list if v < np.mean(x_upper_list)]
    # if len(y_upper_list) >= 3:
    #     y_upper_list = [v for v in y_upper_list if v < np.mean(y_upper_list)]

    roi_weight = 2

    x_lower_list.extend([container_box[0]] * roi_weight)
    y_lower_list.extend([container_box[1]] * roi_weight)
    x_upper_list.extend([container_box[2]] * roi_weight)
    y_upper_list.extend([container_box[3]] * roi_weight)

    x_lower_list.sort()
    y_lower_list.sort()
    x_upper_list.sort(reverse=True)
    y_upper_list.sort(reverse=True)

    x_lower = int(np.mean(np.array(x_lower_list[:2])))
    y_lower = int(np.mean(np.array(y_lower_list[:2])))
    x_upper = int(np.mean(np.array(x_upper_list[:2])))
    y_upper = int(np.mean(np.array(y_upper_list[:2])))

    return x_lower, y_lower, x_upper, y_upper


def extract_front_ceiling_line_segments(line_points_list, pt0):
    front_ceiling_line_points_list = []
    for pt1, pt2 in line_points_list:
        pt12 = pt2 - pt1
        horizontal_flag = np.abs(pt12[1] / pt12[0]) < 0.05
        above_vanishing_point_flag = (
            -np.pi * 5 / 6 < cmath.phase(complex(*tuple(pt1 - pt0))) < -np.pi * 1 / 6
            and -np.pi * 5 / 6
            < cmath.phase(complex(*tuple(pt2 - pt0)))
            < -np.pi * 1 / 6
        )
        if horizontal_flag and above_vanishing_point_flag:
            front_ceiling_line_points_list.append((pt1, pt2))

    return front_ceiling_line_points_list


def draw_report_img(
    img,
    line_points_list=None,
    q_depth_line_points_list=None,
    front_ceiling_line_points_list=None,
    container_box=None,
):
    img_out = img // 8
    if line_points_list is not None:
        for pt1, pt2 in line_points_list:
            img_out = cv2.line(
                img_out, pt1=tuple(pt1), pt2=tuple(pt2), color=127, thickness=1
            )
    if q_depth_line_points_list is not None:
        depth_line_points_list = flatten(q_depth_line_points_list)
        for pt1, pt2 in depth_line_points_list:
            img_out = cv2.line(
                img_out, pt1=tuple(pt1), pt2=tuple(pt2), color=191, thickness=2
            )
    if front_ceiling_line_points_list is not None:
        for pt1, pt2 in front_ceiling_line_points_list:
            img_out = cv2.line(
                img_out, pt1=tuple(pt1), pt2=tuple(pt2), color=191, thickness=1
            )
    if container_box is not None:
        x_lower, y_lower, x_upper, y_upper = container_box
        edges = [
            dict(pt1=(x_lower, y_lower), pt2=(x_lower, y_upper)),
            dict(pt1=(x_lower, y_upper), pt2=(x_upper, y_upper)),
            dict(pt1=(x_upper, y_upper), pt2=(x_upper, y_lower)),
            dict(pt1=(x_upper, y_lower), pt2=(x_lower, y_lower)),
        ]
        for edge in edges:
            img_out = cv2.line(img_out, color=255, thickness=3, **edge)

    return img_out


def estimate_empty_area_ratio(q_depth_line_points_list, container_box, pt0):
    empty_ratio_dict = dict(empty_ratio=0, left_empty_ratio=0, right_empty_ratio=0)
    x_lower, y_lower, x_upper, y_upper = container_box

    top_depth_line_points_list = (
        q_depth_line_points_list[0] + q_depth_line_points_list[1]
    )
    if top_depth_line_points_list:
        top_y = max([pt1[1] for pt1, _ in top_depth_line_points_list])
        top_ratio = (top_y - y_lower) / (pt0[1] - y_lower)

        bottom_left_line_points_list = q_depth_line_points_list[2]
        if bottom_left_line_points_list:
            bottom_left_x = max([pt1[0] for pt1, _ in bottom_left_line_points_list])
            bottom_left_ratio = max(0.0, (bottom_left_x - x_lower) / (pt0[0] - x_lower))
            left_empty_ratio = min(1.0, bottom_left_ratio / top_ratio)
            empty_ratio_dict["left_empty_ratio"] = left_empty_ratio

        bottom_right_line_points_list = q_depth_line_points_list[3]
        if bottom_right_line_points_list:
            bottom_right_x = min([pt1[0] for pt1, _ in bottom_right_line_points_list])
            bottom_right_ratio = max(
                0.0, (x_upper - bottom_right_x) / (x_upper - pt0[0])
            )
            right_empty_ratio = min(1.0, bottom_right_ratio / top_ratio)
            empty_ratio_dict["right_empty_ratio"] = right_empty_ratio

    empty_ratio_dict["empty_ratio"] = (
        empty_ratio_dict["left_empty_ratio"] + empty_ratio_dict["right_empty_ratio"]
    ) / 2

    return empty_ratio_dict


def flatten(ls):
    return list(itertools.chain.from_iterable(ls))
