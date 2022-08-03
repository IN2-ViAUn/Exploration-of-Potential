#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()

        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)

        line_thickness = 4 
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (x0, y0), (x1, y1)  # c1是目标框左上点坐标；c2是目标框右下点坐标
        w = x1 - x0  # 目标框的宽
        h = y1 - y0  # 目标框的高

        # 目标框 
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]   # 获得文字尺寸t_size = ((128, 15), 6)[0] = (128, 15)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # t_size[0]是文字的宽；t_size[1]是文字的高；c2代表文字框的右上点坐标
        # 文字框及文字
        # cv2.rectangle(image, start_point, end_point, color, thickness) 厚度-1像素将以指定的颜色填充矩形形状
        if c1[1] < t_size[1] + 3: # 如果目标框上方不够放置文字框
            # 放在目标框上边界的下方
            # c1 = c1[0], c1[1] + h
            c2 = c2[0], c2[1] + 2 * (t_size[1] + 3)

            # # 放在目标框下边界的下方
            # c1 = c1[0], c1[1] + h
            # c2 = c2[0], c2[1] + h + 2 * (t_size[1] + 3)

            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, text, (c1[0], c1[1] + t_size[1]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        else:
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA) # c1是文字框的左下点坐标；c2是文字框右上点坐标
            cv2.putText(img, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)  # 白色字
        
        # # 原
        # color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
