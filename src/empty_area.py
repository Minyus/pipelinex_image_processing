import math
from operator import itemgetter
import cv2
from pylsd.lsd import lsd
import logging
from scipy.sparse import coo_matrix
import cmath
import itertools
import numpy as np
from numpy.linalg import norm

log = logging.getLogger(__name__)


def detect_lines_and_estimate_empty_ratio(edge_img, roi, seg_edge_img, vis_img):

    intersection_img, line_points_list, tilted_line_points_list = detect_line_segments(
        edge_img
    )
    pt0 = get_vanishing_point(intersection_img)
    depth_line_points_list = extract_depth_line_segments(tilted_line_points_list, pt0)
    depth_line_points_list = connect_line_segments(depth_line_points_list, pt0)
    q_depth_line_points_list = extract_q_line_segments(depth_line_points_list, pt0)
    container_box = estimate_container_box(q_depth_line_points_list, roi, pt0)
    front_ceiling_line_points_list = extract_front_ceiling_line_segments(
        line_points_list, pt0
    )
    empty_region_points_list = estimate_empty_region(container_box, pt0, seg_edge_img)
    empty_ratio_dict, selected_ratio_lines_list = estimate_empty_area_ratio(
        q_depth_line_points_list, container_box, pt0, empty_region_points_list,
    )
    report_img = draw_report_img(
        vis_img,
        line_points_list,
        q_depth_line_points_list,
        front_ceiling_line_points_list,
        container_box,
        pt0,
        empty_region_points_list,
        selected_ratio_lines_list,
    )
    return empty_ratio_dict, intersection_img, report_img


def detect_line_segments(img):

    lines = lsd(img)
    h, w = img.shape
    s = np.array([w, h])
    center_point = s / 2
    min_length = 0.03 * (h + w) / 2
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

    line_angle_list = [phase(pt2 - pt1) for pt1, pt2 in tilted_line_points_list]

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


def get_vanishing_point(a):
    return argsmax(a)[::-1]


def extract_depth_line_segments(line_points_list, pt0):
    depth_line_points_list = []
    for line_points in line_points_list:
        depth_flag = (
            get_point_to_line_distance(line_points, pt0) < 20
            and angle_diff(line_points, pt0) < np.pi * 10 / 180
        )
        if depth_flag:
            depth_line_points_list.append(line_points)
    return depth_line_points_list


def connect_line_segments(depth_line_points_list, pt0):

    n_lines = len(depth_line_points_list)

    polar_pt1_list = [polar(pt1 - pt0) for pt1, _ in depth_line_points_list]
    polar_pt2_list = [polar(pt2 - pt0) for _, pt2 in depth_line_points_list]

    connected_line_points_list = []
    connected_index_list = []
    for i in range(n_lines):
        for j in range(n_lines):
            polar_pt1 = polar_pt1_list[i]
            polar_pt2 = polar_pt2_list[j]
            if (
                (i != j)
                and (0 < (polar_pt1[0] - polar_pt2[0]) < 50)
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

    roi_weight = 1

    x_lower_list.extend([container_box[0]] * roi_weight)
    y_lower_list.extend([container_box[1]] * roi_weight)
    x_upper_list.extend([container_box[2]] * roi_weight)
    y_upper_list.extend([container_box[3]] * roi_weight)

    x_lower_list.sort()
    y_lower_list.sort()
    x_upper_list.sort(reverse=True)
    y_upper_list.sort(reverse=True)

    x_lower = int(np.mean(np.array(x_lower_list[:3])))
    y_lower = int(np.mean(np.array(y_lower_list[:3])))
    x_upper = int(np.mean(np.array(x_upper_list[:3])))
    y_upper = int(np.mean(np.array(y_upper_list[:3])))

    return x_lower, y_lower, x_upper, y_upper


def extract_front_ceiling_line_segments(line_points_list, pt0):
    front_ceiling_line_points_list = []
    for pt1, pt2 in line_points_list:
        pt12 = pt2 - pt1
        horizontal_flag = np.abs(pt12[1] / pt12[0]) < 0.05
        above_vanishing_point_flag = (
            -np.pi * 5 / 6 < phase(pt1 - pt0) < -np.pi * 1 / 6
            and -np.pi * 5 / 6 < phase(pt2 - pt0) < -np.pi * 1 / 6
        )
        if horizontal_flag and above_vanishing_point_flag:
            front_ceiling_line_points_list.append((pt1, pt2))

    return front_ceiling_line_points_list


def estimate_empty_region(container_box, pt0, seg_edge_img):

    x_lower, y_lower, x_upper, y_upper = container_box

    points = np.stack(
        [np.array([x_lower, y_upper]), np.array([x_upper, y_upper]), pt0,]
    )

    mask_img = np.zeros_like(seg_edge_img)
    mask_img = cv2.fillConvexPoly(mask_img, points=points, color=1)
    masked_seg_edge_img = (seg_edge_img > 1).astype(np.uint8) * mask_img

    outline_y = last_argmax(masked_seg_edge_img)
    outline_points = enumerate(outline_y)
    empty_region_points = [
        (np.array([pt[0], np.clip(pt[1], pt0[1], y_upper)]), np.array([pt[0], y_upper]))
        for pt in outline_points
        if (x_lower < pt[0] < x_upper)
    ]

    return empty_region_points


def estimate_empty_area_ratio(
    q_depth_line_points_list, container_box, pt0, empty_region_points_list
):
    empty_ratio_dict = dict(empty_ratio=0, left_empty_ratio=0, right_empty_ratio=0)
    x_lower, y_lower, x_upper, y_upper = container_box
    box_dict = {
        0: [x_lower, y_lower],
        1: [x_upper, y_lower],
        2: [x_lower, y_upper],
        3: [x_upper, y_upper],
    }

    def phase_vp(pt):
        return phase(pt - pt0)

    def sin_vp(pt):
        return np.sin(phase_vp(pt))

    def ratio_line_x(pt, box):
        b_pt = pt.copy()
        b_pt[0] = box
        return pt, b_pt, (pt[0] - box) / (pt0[0] - box)

    def ratio_line_y(pt, box):
        b_pt = pt.copy()
        b_pt[1] = box
        return pt, b_pt, (pt[1] - box) / (pt0[1] - box)

    def get_lines_with_max_ratio(ratio_line_list):
        if ratio_line_list:
            return max(ratio_line_list, key=itemgetter(2))
        else:
            return None

    def get_ratio_line(pt, q_index):
        if q_index >= 4:
            return ratio_line_y(pt, y_upper)
        box = box_dict[q_index]
        y_flag = sin_vp(pt) > sin_vp(np.array(box))
        if y_flag:
            return ratio_line_y(pt, box[1])
        else:
            return ratio_line_x(pt, box[0])

    if empty_region_points_list is not None:
        q_depth_line_points_list.append(empty_region_points_list)

    q_ratio_line_list = []
    for i, line_points_list in enumerate(q_depth_line_points_list):
        ratio_line_list = [
            get_ratio_line(pt1, q_index=i) for pt1, _, in line_points_list
        ]
        q_ratio_line_list.append(ratio_line_list)

    top_ratio_line = get_lines_with_max_ratio(
        q_ratio_line_list[0] + q_ratio_line_list[1]
    )
    bottom_left_ratio_line = get_lines_with_max_ratio(q_ratio_line_list[2])
    bottom_right_ratio_line = get_lines_with_max_ratio(q_ratio_line_list[3])
    empty_region_ratio_line = get_line_mean(q_ratio_line_list[4])

    selected_ratio_lines_list = []
    if top_ratio_line is None:
        left_ratio = 1
        right_ratio = 1
    else:
        selected_ratio_lines_list.append(top_ratio_line)
        if bottom_left_ratio_line is None:
            left_ratio = 0
        else:
            left_ratio = clip(bottom_left_ratio_line[2] / top_ratio_line[2], 0, 1)
            selected_ratio_lines_list.append(bottom_left_ratio_line)
        if bottom_right_ratio_line is None:
            right_ratio = 0
        else:
            right_ratio = clip(bottom_right_ratio_line[2] / top_ratio_line[2], 0, 1)
            selected_ratio_lines_list.append(bottom_right_ratio_line)
        if empty_region_ratio_line is not None:
            empty_ratio = clip(empty_region_ratio_line[2] / top_ratio_line[2], 0, 1)
            selected_ratio_lines_list.append(empty_region_ratio_line)

    empty_ratio_dict["empty_ratio"] = empty_ratio
    empty_ratio_dict["left_empty_ratio"] = left_ratio
    empty_ratio_dict["right_empty_ratio"] = right_ratio

    return empty_ratio_dict, selected_ratio_lines_list


def draw_report_img(
    img,
    line_points_list=None,
    q_depth_line_points_list=None,
    front_ceiling_line_points_list=None,
    container_box=None,
    pt0=None,
    empty_region_points_list=None,
    selected_ratio_lines_list=None,
):
    color_flag = img.ndim == 3
    img_out = img // (2 if color_flag else 8)
    if line_points_list is not None:
        for pt1, pt2 in line_points_list:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt1),
                pt2=tuple(pt2),
                color=(0, 255, 255) if color_flag else 127,
                thickness=1,
            )
    if q_depth_line_points_list is not None:
        for pt1, pt2 in q_depth_line_points_list[4]:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt1),
                pt2=tuple(pt2),
                color=(0, 127, 255) if color_flag else 191,
                thickness=1,
            )
        depth_line_points_list = flatten(q_depth_line_points_list[:4])
        for pt1, pt2 in depth_line_points_list:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt1),
                pt2=tuple(pt2),
                color=(255, 0, 255) if color_flag else 191,
                thickness=2,
            )
    if front_ceiling_line_points_list is not None:
        for pt1, pt2 in front_ceiling_line_points_list:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt1),
                pt2=tuple(pt2),
                color=(255, 255, 0) if color_flag else 191,
                thickness=1,
            )
    if container_box is not None:
        x_lower, y_lower, x_upper, y_upper = container_box
        box_edges = [
            dict(pt1=(x_lower, y_lower), pt2=(x_lower, y_upper)),
            dict(pt1=(x_lower, y_upper), pt2=(x_upper, y_upper)),
            dict(pt1=(x_upper, y_upper), pt2=(x_upper, y_lower)),
            dict(pt1=(x_upper, y_lower), pt2=(x_lower, y_lower)),
        ]
        for edge in box_edges:
            img_out = cv2.line(
                img_out, color=(255, 0, 0) if color_flag else 255, thickness=3, **edge
            )
    if pt0 is not None:
        if container_box is not None:
            cross_edges = [
                dict(pt1=(pt0[0], y_lower), pt2=(pt0[0], y_upper)),
                dict(pt1=(x_lower, pt0[1]), pt2=(x_upper, pt0[1])),
            ]
            for edge in cross_edges:
                img_out = cv2.line(
                    img_out,
                    color=(0, 255, 0) if color_flag else 255,
                    thickness=2,
                    **edge
                )
        else:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt0),
                pt2=tuple(pt0),
                color=(0, 255, 0) if color_flag else 255,
                thickness=10,
            )
    # if empty_region_points_list is not None:
    #     n_points = len(empty_region_points_list)
    #     for i in range(n_points - 1):
    #         img_out = cv2.line(
    #             img_out,
    #             pt1=tuple(empty_region_points_list[i][0]),
    #             pt2=tuple(empty_region_points_list[i + 1][0]),
    #             color=(0, 127, 255) if color_flag else 255,
    #             thickness=2,
    #         )
    if selected_ratio_lines_list is not None:
        for pt1, b_pt, _ in selected_ratio_lines_list:
            img_out = cv2.line(
                img_out,
                pt1=tuple(pt1),
                pt2=tuple(b_pt),
                color=(0, 0, 255) if color_flag else 255,
                thickness=3,
            )

    return img_out


def flatten(ls):
    return list(itertools.chain.from_iterable(ls))


def phase(pt):
    return cmath.phase(complex(*tuple(pt)))


def polar(pt):
    return cmath.polar(complex(*tuple(pt)))


def angle_diff(points, pt0):
    pt1, pt2 = points
    return np.abs(phase(pt1 - pt0) - phase(pt2 - pt0))


def get_point_to_line_distance(line_points, pt0):
    pt1, pt2 = line_points
    v12 = pt2 - pt1
    nv12 = v12 / norm(v12)
    v01 = pt1 - pt0
    return np.abs(nv12[0] * v01[1] - nv12[1] * v01[0])


def extend(pt1, pt2, scale=1):
    pt2 = scale * (pt2 - pt1) + pt1
    return pt1, pt2


def argsmax(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)


def list_to_arrays(line, c):
    x1, y1, x2, y2 = map(int, line[:4])

    pt1 = np.array([x1, y1])
    pt2 = np.array([x2, y2])
    if norm(pt1 - c) > norm(pt2 - c):
        pt1, pt2 = pt2, pt1
    return pt1, pt2


def clip(a, *args, **kwargs):
    return float(np.clip(a, *args, **kwargs))


def last_argmax(a, axis=0):
    if axis == 0:
        rev = a[::-1, ...]
    if axis == 1:
        rev = a[:, ::-1]
    args = np.argmax(rev, axis=axis)
    argsrev = [a.shape[axis] - arg - 1 for arg in args]
    return argsrev


def unzip(zipped_list):
    return zip(*zipped_list)


def get_line_mean(tuple_list):
    unzipped = unzip(tuple_list)
    mean_list = [
        np.mean(np.stack(item), axis=0).astype(np.int64)
        if isinstance(item[0], np.ndarray)
        else np.mean(item)
        for item in unzipped
    ]
    return tuple(mean_list)
