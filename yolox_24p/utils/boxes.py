
import numpy as np
import cv2

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]

def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

# 预测数据筛选处理
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    theta = torch.tensor(15*np.pi/180, device=prediction.device)
    theta_all = torch.arange(24, device=prediction.device) * theta
    cos_theta_all = theta_all * torch.cos(theta_all)
    sin_theta_all = theta_all * torch.sin(theta_all)

    # 网络的输出结果尺寸,[1,8400,131]
    box_corner = prediction.new(prediction.shape)

    #所有预测框的预测结果,非归一化结果，直接像素坐标系结果
    box_corner[:, :, :26] = prediction[:, :, :26] # 中心点+预测24点坐标赋值
    
    # 创建一个和prediction一样长度的None列表
    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        # image_pred是[8400, 85]尺寸，表示每一张图片的预测结果，8400个框数据
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        # 从80个类别中找到评分最高的项目【class_conf】，获取该项目的索引编号【class_pred】
        class_conf, class_pred = torch.max(image_pred[:, 27: 27 + num_classes], 1, keepdim=True)
        # print(class_conf)

        # 过滤掉评分太低的目标检测结果
        conf_mask = (image_pred[:, 26] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :27], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
    
        if not detections.size(0):
            continue
        
        # 为了使用NMS，需要将24点的外框取出
        cos_theta_all = cos_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)
        sin_theta_all = sin_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)

        p24_x = detections[:, 2:26] * cos_theta_all + detections[:, 0].unsqueeze(1).repeat(1, 24)
        p24_y = detections[:, 2:26] * sin_theta_all + detections[:, 1].unsqueeze(1).repeat(1, 24)


        p24_x_min = p24_x.min(dim=1).values
        p24_x_max = p24_x.max(dim=1).values
        p24_y_min = p24_y.min(dim=1).values
        p24_y_max = p24_y.max(dim=1).values
        
        p24_rect = torch.stack((p24_x_min, p24_y_min, p24_x_max, p24_y_max), dim=0).transpose(0, 1)

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                p24_rect,
                detections[:, 26] * detections[:, 27],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                p24_rect,
                detections[:, 26] * detections[:, 27],
                detections[:, 28],
                nms_thre,
            )

        detections = detections[nms_out_index]
        
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    # 输出筛选后的结果，类型是list，list长度与batch相同
    return output

# 返回两圆相交部分面积，圆心距离
def circle_inter(c_gtx, c_gty, gt_r, c_pdx, c_pdy, pd_r):

    torch_pi = torch.tensor(np.pi)
    # 转换格式
    # c_gtx, c_gty：[numb_gt]
    # c_pdx, c_pdy：[numb_pd]
    # gt_r_min :[numb_gt, 24]
    # pd_r_min :[numb_pd, 24]
    numb_gt = c_gtx.shape[0]
    numb_pd = c_pdx.shape[0]
    
    # 复制对齐对应关系[numb_gt*numb_pd, 1]
    c_gtx = c_gtx.reshape(numb_gt, 1).repeat_interleave(numb_pd, 0)
    c_gty = c_gty.reshape(numb_gt, 1).repeat_interleave(numb_pd, 0)
    c_pdx = c_pdx.reshape(numb_pd, 1).repeat(numb_gt, 1)
    c_pdy = c_pdy.reshape(numb_pd, 1).repeat(numb_gt, 1)
    # [numb_gt*numb_pd, 24]
    gt_rr = gt_r.repeat_interleave(numb_pd, 0)
    pd_rr = pd_r.repeat(numb_gt, 1)
    
    # 结果缓存 [numb, 24]
    res_inter = torch.zeros_like(gt_rr, device = gt_r.device) 

    # 圆心距离 [numb_gt*numb_pd, 1]
    # 所有预测结果和标签结果的距离
    dist = torch.sqrt((c_gtx - c_pdx)**2 + (c_gty - c_pdy)**2)

    # 正常计算，先找出预测和标签中每个位置中较小的半径[numb_gt*numb_pd, 24]
    min_circle_r, _ = torch.min(torch.stack((gt_rr, pd_rr), 0), 0)  
    max_circle_r, _ = torch.max(torch.stack((gt_rr, pd_rr), 0), 0)

    ac_min = (min_circle_r**2 + dist**2 - max_circle_r**2)/(2*min_circle_r*dist + 1e-8)
    ac_max = (max_circle_r**2 + dist**2 - min_circle_r**2)/(2*max_circle_r*dist + 1e-8)
    
    ac_min = torch.clip(ac_min, min=-0.99, max=0.99)
    ac_max = torch.clip(ac_max, min=-0.99, max=0.99)

    ang_min = torch.acos(ac_min)
    ang_max = torch.acos(ac_max)

    # 真值和预测结果交集面积[numb_gt*numb_pd, 24]
    inter = ang_min * min_circle_r**2 + ang_max * max_circle_r**2 - min_circle_r * dist * torch.sin(ang_min)       

    # 如果两圆半径差绝对值大于圆心距离，则结果为小圆面积
    min_idx = torch.abs(gt_rr - pd_rr) >= dist.repeat(1, 24)
    min_circle_s = torch_pi * (min_circle_r**2)
    # [numb_gt*numb_pd, 24]
    res_inter[min_idx] = min_circle_s[min_idx]

    # 如果圆心大于两半径和,结果为0
    area_0_idx = dist >= gt_rr + pd_rr
    res_inter[area_0_idx] = 0
    
    # 交集面积赋值
    inter_idx = ~(min_idx + area_0_idx)
    res_inter[inter_idx] = inter[inter_idx]

    # 所有真值和所有预测的面积
    # 面积[numb_gt*numb_pd, 24]
    # 所有真值和所有预测的距离
    # 距离[numb_gt*numb_pd, 24]
    return res_inter, dist.repeat(1, 24)
    
# bboxes_a是box的真值信息，bboxes_b是box的预测信息
def bboxes_iou(bboxes_a, bboxes_b, imgs=None):
    if bboxes_b.shape[1] != 26 or bboxes_a.shape[1] != 50:
        raise IndexError

    torch_pi = torch.tensor(np.pi)
    # pred:[numb_obj, 26]
    # target:[numb_pred_obj, 50]
    bboxes_b = bboxes_b.view(-1, 26)
    bboxes_a = bboxes_a.view(-1, 50)

    numb_gt = bboxes_a.shape[0]
    numb_pd = bboxes_b.shape[0]

    # 真值信息中心点
    gt_center_x = bboxes_a[:, 0].to(torch.float)
    gt_center_y = bboxes_a[:, 1].to(torch.float)
    pd_center_x = bboxes_b[:, 0].to(torch.float)
    pd_center_y = bboxes_b[:, 1].to(torch.float)
    
    # 真值信息24点坐标
    gt_24p_x = bboxes_a[:, 2::2].to(torch.float)
    gt_24p_y = bboxes_a[:, 3::2].to(torch.float)
    # 标签数据向量化
    gt_vect_x = gt_24p_x - gt_center_x.reshape((-1,1))
    gt_vect_y = gt_24p_y - gt_center_y.reshape((-1,1))
            
    # 合并x和y
    gt_vect_xy = torch.cat((gt_vect_x, gt_vect_y), 1).reshape(-1, 2, gt_vect_x.shape[1])
    
    # 计算模长（24点距离中心位置的距离）
    # 【注意！！！】：【信息排布：从右侧水平开始，顺时针一圈】
    scale_gt = torch.norm(gt_vect_xy, dim=1, out=None, keepdim=False)
    scale_pd = bboxes_b[:, 2:]

    # 计算圆形面积
    area_gt_circle = torch_pi * scale_gt**2
    area_pd_circle = torch_pi * scale_pd**2
    area_gt_circle = area_gt_circle.repeat_interleave(numb_pd, 0)
    area_pd_circle = area_pd_circle.repeat(numb_gt, 1)

    # 计算交集面积
    area_inter, circle_dist = circle_inter(gt_center_x, gt_center_y, scale_gt, pd_center_x, pd_center_y, scale_pd)

    # 下面计算均仅计算最小面积
    # 计算IOU
    iou_24 = area_inter / (area_gt_circle + area_pd_circle - area_inter + 1e-6)

    # 计算GIOU
    # 相离、相交、包含计算模式不一样
    # 如果两圆半径差绝对值大于圆心距离，则c_l结果为大圆直径
    scale_gt = scale_gt.repeat_interleave(numb_pd, 0)
    scale_pd = scale_pd.repeat(numb_gt, 1)
    c_l_mode = torch.abs(scale_gt - scale_pd) >= circle_dist
    max_circle_r, _ = torch.max(torch.stack((scale_gt, scale_pd), 0), 0) 

    giou_c_l = (scale_gt + scale_pd + circle_dist)/2
    giou_c_l[c_l_mode] = max_circle_r[c_l_mode]

    # 比例情况
    # [numb, 24]
    scale_gt_min, _ = torch.min(scale_gt, 1, True)
    scale_pd_min, _ = torch.min(scale_pd, 1, True)

    # [numb, 24]
    giou_c_s = torch_pi * giou_c_l**2

    # giou分子
    giou_top =  giou_c_s - (area_gt_circle + area_pd_circle - area_inter)
    # giou分母
    giou_24 = iou_24 - giou_top / giou_c_s
    
    # 几种损失函数
    loss_giou24 = (1 - giou_24).sum(1)/24
    # loss_giou范围是0-2，需要映射到0-1
    # 否则计算类别Loss会出错
    loss_giou24 = loss_giou24.reshape(numb_gt, numb_pd)/2    

    return loss_giou24

# 内角和法计算连通域面积
def gtpts2poly(bboxes_24p):
    # 标签中目标数量
    obj_numb = bboxes_24p.shape[0]
    # 取目标的坐标
    target_x = bboxes_24p[:,2::2]
    target_y = bboxes_24p[:,3::2]

    # 生成与拼接网格坐标匹配的目标坐标，尺寸[24, 8400], 每一个通道是一个x值，共24通道
    target_x_24 = target_x.repeat(8400, 1).permute(1, 0)
    target_y_24 = target_y.repeat(8400, 1).permute(1, 0)

    # 向量转换
    vect_start_x = target_x_24 - x_grid
    vect_start_y = target_y_24 - y_grid
    # 错位转换对齐
    vect_end_x = target_x_24.roll(-1, 0) - x_grid
    vect_end_y = target_y_24.roll(-1, 0) - y_grid

    # 格式转换成向量【x,y】,[24, 8400, 2]
    vect_start = torch.stack((vect_start_x, vect_start_y), 1).permute(0,2,1)
    vect_end = torch.stack((vect_end_x, vect_end_y), 1).permute(0,2,1)

    # 二维向量叉乘运算x1y2-x2y1
    vect_cross = torch.mul(vect_start_x, vect_end_y) - torch.mul(vect_end_x, vect_start_y) 
    vect_dot = torch.mul(vect_start, vect_end).sum(2)

    arctan_rad = torch.atan2(torch.abs(vect_cross), vect_dot)
    degree = torch.rad2deg(arctan_rad).sum(0)
    idx = degree >= 350
    print(degree[idx])        

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
