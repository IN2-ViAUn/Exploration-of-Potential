from matplotlib import scale
from loguru import logger

import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from zmq import device

from utils import bboxes_iou

class IOUloss(nn.Module):
    def __init__(self, reduction="none"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        # 常数参数
        self.torch_pi = torch.tensor(np.pi)
        self.theta_15 = torch.tensor(15*np.pi/180) #转为弧度

    # 返回两圆相交部分面积，圆心距离
    def circle_inter(self, c_gtx, c_gty, gt_r, c_pdx, c_pdy, pd_r):
        torch_pi = torch.tensor(np.pi)
        # 转换格式
        # c_gtx, c_gty：[numb_gt]
        # c_pdx, c_pdy：[numb_pd]
        # gt_r_min :[numb_gt, 24]
        # pd_r_min :[numb_pd, 24]
                
        # 结果缓存 [numb, 24]
        res_inter = torch.zeros_like(gt_r, device = gt_r.device) 

        # 圆心距离 [numb_gt*numb_pd, 1]
        # 所有预测结果和标签结果的距离
        dist = torch.sqrt((c_gtx - c_pdx)**2 + (c_gty - c_pdy)**2)
        dist = dist.unsqueeze(1).repeat(1, 24)
        
        # 正常计算，先找出预测和标签中每个位置中较小的半径[numb_gt*numb_pd, 24]
        if (gt_r.shape[0] == 0) or (pd_r.shape[0] == 0):
            # 如果是占位label，直接返回无效占位数据
            return res_inter,dist
        else:
            min_circle_r, _ = torch.min(torch.stack((gt_r, pd_r), 0), 0)  
            max_circle_r, _ = torch.max(torch.stack((gt_r, pd_r), 0), 0)
             
        ac_min = (min_circle_r**2 + dist**2 - max_circle_r**2)/(2*min_circle_r*dist + 1e-8)
        ac_max = (max_circle_r**2 + dist**2 - min_circle_r**2)/(2*max_circle_r*dist + 1e-8)
        
        ac_min = torch.clip(ac_min, min=-0.99, max=0.99)
        ac_max = torch.clip(ac_max, min=-0.99, max=0.99)

        ang_min = torch.acos(ac_min)
        ang_max = torch.acos(ac_max)

        # 真值和预测结果交集面积[numb_gt*numb_pd, 24]
        inter = ang_min * min_circle_r**2 + ang_max * max_circle_r**2 - min_circle_r * dist * torch.sin(ang_min)       

        # 如果两圆半径差绝对值大于圆心距离，则结果为小圆面积
        min_idx = torch.abs(gt_r - pd_r) >= dist
        min_circle_s = torch_pi * (min_circle_r**2)
        
        # [numb_gt*numb_pd, 24]
        res_inter[min_idx] = min_circle_s[min_idx]

        # 如果圆心大于两半径和,结果为0
        area_0_idx = dist >= gt_r + pd_r
        res_inter[area_0_idx] = 0
        
        # 交集面积赋值
        inter_idx = ~(min_idx + area_0_idx)
        res_inter[inter_idx] = inter[inter_idx]

        # 所有真值和所有预测的面积
        # 面积[numb_gt*numb_pd, 24]
        # 所有真值和所有预测的距离
        # 距离[numb_gt*numb_pd, 24]
        return res_inter, dist

    def forward(self, pred, target):
        if pred.shape[1] != 26 or target.shape[1] != 50:
            raise IndexError

        torch_pi = torch.tensor(np.pi)
        # pred:[numb_obj, 26]
        # target:[numb_pred_obj, 50]
        pred = pred.view(-1, 26)
        target = target.view(-1, 50)

        # 真值信息中心点
        gt_center_x = target[:, 0].to(torch.float)
        gt_center_y = target[:, 1].to(torch.float)
        pd_center_x = pred[:, 0].to(torch.float)
        pd_center_y = pred[:, 1].to(torch.float)
        
        # 真值信息24点坐标
        gt_24p_x = target[:, 2::2].to(torch.float)
        gt_24p_y = target[:, 3::2].to(torch.float)
        # 标签数据向量化
        gt_vect_x = gt_24p_x - gt_center_x.reshape((-1,1))
        gt_vect_y = gt_24p_y - gt_center_y.reshape((-1,1))
                
        # 合并x和y
        gt_vect_xy = torch.cat((gt_vect_x, gt_vect_y), 1).reshape(-1, 2, gt_vect_x.shape[1])
        
        # 计算模长（24点距离中心位置的距离）
        # 【注意！！！】：【信息排布：从右侧水平开始，顺时针一圈】
        scale_gt = torch.norm(gt_vect_xy, dim=1, out=None, keepdim=False)
        scale_pd = pred[:, 2:]

        if (scale_gt.shape[0] == 0) or (scale_pd.shape[0] == 0):
            # 如果是占位label，直接返回无效占位数据
            loss_giou24 = scale_gt.new_zeros(1, 24)
            draw_content = [pd_center_x.new_zeros(1, 24), pd_center_y.new_zeros(1, 24), scale_pd.new_zeros(1, 24)]
            return loss_giou24, draw_content

        # 计算圆形面积
        area_gt_circle = torch_pi * scale_gt**2
        area_pd_circle = torch_pi * scale_pd**2

        # 计算交集面积
        area_inter, circle_dist = self.circle_inter(gt_center_x, gt_center_y, scale_gt, pd_center_x, pd_center_y, scale_pd)

        # 下面计算均仅计算最小面积
        # 计算IOU
        iou_24 = area_inter / (area_gt_circle + area_pd_circle - area_inter + 1e-6)

        # 计算GIOU
        # 相离、相交、包含计算模式不一样
        # 如果两圆半径差绝对值大于圆心距离，则c_l结果为大圆直径
        c_l_mode = torch.abs(scale_gt - scale_pd) >= circle_dist
        max_circle_r, _ = torch.max(torch.stack((scale_gt, scale_pd), 0), 0) 

        giou_c_l = (scale_gt + scale_pd + circle_dist)/2
        giou_c_l[c_l_mode] = max_circle_r[c_l_mode]

        # 比例情况
        # [numb, 24]
        # scale_gt_min, _ = torch.min(scale_gt, 1, True)
        # scale_pd_min, _ = torch.min(scale_pd, 1, True)

        # [numb, 24]
        giou_c_s = torch_pi * giou_c_l**2

        # giou分子
        giou_top =  giou_c_s - (area_gt_circle + area_pd_circle - area_inter)
        # giou分母
        giou_24 = iou_24 - giou_top / giou_c_s
        
        # 损失函数
        loss_giou24 = (1 - giou_24)
        # loss = loss_giou24.sum(1)/24

        draw_content = [pd_center_x, pd_center_y, scale_pd]

        # loss
        return loss_giou24, draw_content

class Loss_Function(nn.Module):
    def __init__(self, num_classes):
        super(Loss_Function, self).__init__()
        # loss函数
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.num_classes = num_classes
        
        # 初始化
        self.last_iou_loss = 1.0
        self.last_obj_loss = 1.0
        self.last_cls_loss = 1.0

     # 定义loss损失函数
    def forward(self, outputs_train, labels):
    
        x_shifts = outputs_train[0]
        y_shifts = outputs_train[1]
        expanded_strides = outputs_train[2]
        outputs = outputs_train[3]
        origin_preds = outputs_train[4]

        # labels:[batch, max_numb_class(50), 51]
        # outputs: torch.Size([1, 8400, 26+1+80])
        bbox_preds = outputs[:, :, :26]  # [batch, n_anchors_all, 50]
        obj_preds = outputs[:, :, 26].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 27:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 50))
                l1_target = outputs.new_zeros((0, 50))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:]
                gt_classes = labels[batch_idx, :num_gt, 0]
                # 单张图片的回归框结果
                bboxes_preds_per_image = bbox_preds[batch_idx]

                gt_matched_classes,\
                fg_mask,\
                pred_ious_this_matching,\
                matched_gt_inds,\
                num_fg_img\
                = self.get_assignments(  # noqa
                batch_idx,
                num_gt,
                total_num_anchors,
                gt_bboxes_per_image,
                gt_classes,
                bboxes_preds_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                cls_preds,
                bbox_preds,
                obj_preds
                )
                    
                num_fg += num_fg_img
                
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                

                obj_target = fg_mask.unsqueeze(-1)
                
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 26)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(torch.float))
            
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        

        iou_value, draw_content = self.iou_loss(bbox_preds.view(-1, 26)[fg_masks], reg_targets)
        
        # 将任务分为24个小任务
        loss_iou = (
                iou_value
            ).sum(0) / num_fg

        # loss_iou = (
        #         iou_value
        #     ).sum() / num_fg
        
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 26)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # 计算变化的权重
        loss_iou_value = loss_iou.clone().detach()
        loss_obj_value = loss_obj.clone().detach()
        loss_cls_value = loss_cls.clone().detach()

        r_iou = loss_iou_value / (self.last_iou_loss + 1e-8) 
        r_obj = loss_obj_value / (self.last_obj_loss + 1e-8)
        r_cls = loss_cls_value / (self.last_cls_loss + 1e-8)


        r_iou = torch.clip(r_iou, 0, 2) 
        r_obj = torch.clip(r_obj, 0, 2)
        r_cls = torch.clip(r_cls, 0, 2)

        # 温度系数
        T = torch.tensor(20.0, device = r_iou.device, dtype=torch.float, requires_grad=False)

        # 分母
        denominator_w = torch.exp(r_iou/T).sum() + torch.exp(r_obj/T) + torch.exp(r_cls/T)

        
        reg_w = 26*torch.exp(r_iou/T) / (denominator_w)
        obj_w = 26*torch.exp(r_obj/T) / (denominator_w)
        cls_w = 26*torch.exp(r_cls/T) / (denominator_w)
        
        # 保存权重
        draw_content.append(reg_w)
        draw_content.append(obj_w)
        draw_content.append(cls_w)
        
        loss = (reg_w*loss_iou).sum() + obj_w*loss_obj + cls_w*loss_cls + loss_l1

        self.last_iou_loss = loss_iou_value
        self.last_obj_loss = loss_obj_value
        self.last_cls_loss = loss_cls_value


        
        return (
            loss,
            reg_w * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
            draw_content
        )

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds
    ):

        # 对单张图片的真值信息结果进行筛选，生成一个mask模板，告诉预测结果[8400, 50]哪个位置应该进行当前的预测
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors
        )

        # 从8400结果中选出应当预测当前数据的预测目标
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        # 总共应该有多少个位置产生预测结果
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        
        # 计算每个预测结果和每个真值的iou数值
        # gt_bboxes_per_image:[obj, 50](50:x,y + 24cord)
        # bboxes_preds_per_image:[pred, 50](50:x,y + 24cord)
        # pair_wise_ious:[obj, pred]
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
        # print(pair_wise_ious, "!!!!!!!!!")

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
            
        
        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )

        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)

        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss


        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # print(cost.sum(),"1")
        # print(pair_wise_ious.sum(),"2")
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # print(matching_matrix.shape, "#####")
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        # print(n_candidate_k, "$$$$$$$")
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # print(topk_ious, "@@@@@@@@@")
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        # dynamic_ks表示每个标签对应的目标在预测过程中分配多少个预测框合适
        dynamic_ks = dynamic_ks.tolist()
 
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # matching_matrix将cost评分表格中最低的前dynamic_ks项置1
        # anchor_matching_gt表示将matching_matrix中的1相加，如果某一列
        # 超过1，说明多个目标有共用的预测框，需要分离
        anchor_matching_gt = matching_matrix.sum(0)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        
        # 经过上述处理，matching_matrix中已经不存在共用预测框了
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        # 经过筛选处理后，最终应该有num_fg个预测框对应当前图片的预测结果
        num_fg = fg_mask_inboxes.sum().item()
        
        # fg_mask是网络预测的8400个结果中哪些位置应该是有效预测结果的mask
        # 因为原来mask中有同一个位置预测多个目标的情况，所以经过上面的筛选，
        # 让每个位置仅预测一个目标，得到最终的mask
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds    

    # 从预测结果中筛选哪些应该是预测信息
    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        
        
        # 输入图片中网格坐标中心点
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        )  
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        )

        # 判断网格坐标中心点哪些在标签标记的物体中【初步筛选1】
        is_in_boxes = self.pts_in_poly(gt_bboxes_per_image, x_centers_per_image, y_centers_per_image)
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        
        # 判断网格坐标中心点在标签中心5x5的方格内【初步筛选2】
        center_radius = 2.5
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )

        return is_in_boxes_anchor, is_in_boxes_and_center

    # 判断哪些点在多边形内, x_idx和y_idx是需要判断的点
    # 使用的方法是计算所有待判定点与各个顶点的连线夹角和是不是360
    def pts_in_poly(self, bboxes_24p, x_idx, y_idx):
        # 标签中目标物体数量
        target_numb = bboxes_24p.shape[0]
        # 要判断点的数量
        pts_numb = x_idx.shape[0]
        # 计算结果的缓存容器
        is_in_boxes = torch.zeros([target_numb, pts_numb], dtype=torch.bool, device='cuda:0')
        # 取目标的坐标
        target_x = bboxes_24p[:,2::2]
        target_y = bboxes_24p[:,3::2]
        # 循环处理标签中的目标
        for box_idx in torch.arange(0, target_numb, 1, device='cuda:0'):
            # 生成与拼接网格坐标匹配的目标坐标，尺寸[24, 8400], 每一个通道是一个x值，共24通道
            target_x_24 = target_x[box_idx].repeat(pts_numb, 1).permute(1, 0)
            target_y_24 = target_y[box_idx].repeat(pts_numb, 1).permute(1, 0)

            # 向量转换
            vect_start_x = target_x_24 - x_idx
            vect_start_y = target_y_24 - y_idx
            # 错位转换对齐
            vect_end_x = target_x_24.roll(-1, 0) - x_idx
            vect_end_y = target_y_24.roll(-1, 0) - y_idx

            # 格式转换成向量【x,y】,[24, 8400, 2]
            vect_start = torch.stack((vect_start_x, vect_start_y), 1).permute(0,2,1)
            vect_end = torch.stack((vect_end_x, vect_end_y), 1).permute(0,2,1)

            # 二维向量叉乘运算x1y2-x2y1
            vect_cross = torch.mul(vect_start_x, vect_end_y) - torch.mul(vect_end_x, vect_start_y) 
            vect_dot = torch.mul(vect_start, vect_end).sum(2)

            arctan_rad = torch.atan2(torch.abs(vect_cross), vect_dot)
            degree = torch.rad2deg(arctan_rad).sum(0)
            is_in_boxes_cur = degree >= 350
            is_in_boxes[box_idx] = is_in_boxes_cur
            
        # bool类型
        return is_in_boxes

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        # 每一列数据都需要除以stride
        stride = stride.unsqueeze(1).repeat(1, 24)
        gt_x = gt[:, 2::2]
        gt_y = gt[:, 3::2]
        gt_scale = torch.sqrt(gt_x ** 2 + gt_y ** 2)
        l1_target[:, 2:] = torch.log(gt_scale / stride + eps)

        return l1_target