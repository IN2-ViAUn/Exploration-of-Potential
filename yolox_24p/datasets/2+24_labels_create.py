"""
读取json文件,生成x,y和24点polygon
"""
import argparse
from numpy.ma import power
import yaml
import os
import json
import cv2

import numpy as np

from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO

class Polygon_24():
    # mode有Cord和Radius两种，Cord表示存储24点坐标，Radius表示存储24半径
    def __init__(self, mode = "Cord"):
        # 存储数据模式
        self.mode = mode
        # 标签json文件路径
        self.json_label_pth = "/home/gaoyu/COCO/json/instances_train2017.json"
        # 图片文件路径
        self.image_data_pth = "/home/gaoyu/COCO/images/train2017"
        # 处理后的标签保存路径
        self.new_label_pth = "./COCO_24p_label"
        # COCO API接口
        self.coco = COCO(self.json_label_pth)
        # json总数据
        self.json_dict = self.load_label_json()
        # annotations处理后的字典
        self.label_dict_cord24 = {}
        self.label_dict_radius = {}
        # label id的索引关系
        self.coco_id2idx = {'1': 0,   '2': 1,   '3': 2,   '4': 3,   '5': 4, 
                            '6': 5,   '7': 6,   '8': 7,   '9': 8,   '10': 9, 
                            '11': 10, '13': 11, '14': 12, '15': 13, '16': 14, 
                            '17': 15, '18': 16, '19': 17, '20': 18, '21': 19, 
                            '22': 20, '23': 21, '24': 22, '25': 23, '27': 24, 
                            '28': 25, '31': 26, '32': 27, '33': 28, '34': 29, 
                            '35': 30, '36': 31, '37': 32, '38': 33, '39': 34, 
                            '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, 
                            '46': 40, '47': 41, '48': 42, '49': 43, '50': 44, 
                            '51': 45, '52': 46, '53': 47, '54': 48, '55': 49, 
                            '56': 50, '57': 51, '58': 52, '59': 53, '60': 54, 
                            '61': 55, '62': 56, '63': 57, '64': 58, '65': 59, 
                            '67': 60, '70': 61, '72': 62, '73': 63, '74': 64, 
                            '75': 65, '76': 66, '77': 67, '78': 68, '79': 69, 
                            '80': 70, '81': 71, '82': 72, '84': 73, '85': 74, 
                            '86': 75, '87': 76, '88': 77, '89': 78, '90': 79}


    # 载入标签json文件
    def load_label_json(self):
        with open(self.json_label_pth,'r') as load_f:
            load_dict = json.load(load_f)
        return load_dict

    # 计算24次图像旋转mask与水平坐标轴的交界点
    def rotation_for_24p(self, center_x, center_y, mask):
        # 最终存储结果的容器（24点）
        cord_results = []
        # 最终存储结果的容器（半径R）
        radius_results = []
        # 获取图像的长宽
        img_h, img_w = mask.shape[0], mask.shape[1]
        # 图像对角线长度，作为旋转线
        max_line = int(np.sqrt(np.power(img_h,2) + np.power(img_w,2)))
        # 绘图模板初始化
        mask_pad = cv2.copyMakeBorder(mask.copy(), max_line, max_line, max_line, max_line,cv2.BORDER_CONSTANT,value= 0)
        # 获取有mask信息的坐标
        mask_x, mask_y = np.where(mask_pad != 0)
        # 初始化水平旋转线的坐标
        horizontal_cord_x = np.arange(0, max_line, 0.2)
        horizontal_cord_y = np.zeros_like(horizontal_cord_x)
        # 形成坐标
        rot_line = np.array([horizontal_cord_x,
                             horizontal_cord_y])
        for rot_time in range(24):
            # 每次旋转绘制一个模板
            template = cv2.copyMakeBorder(np.zeros_like(mask), max_line, max_line, max_line, max_line,cv2.BORDER_CONSTANT,value= 0)
            # 旋转一次15度
            theta_rad = rot_time*15*np.pi/180
            # 定义旋转矩阵
            M_rot = np.array([[np.cos(theta_rad), -1*np.sin(theta_rad)],
                              [np.sin(theta_rad),    np.cos(theta_rad)]])
            # 旋转结果
            rot_end = np.matmul(M_rot, rot_line).astype(np.int16)
            # 唯一化
            rot_end_uniq = rot_end[0,:] + rot_end[1,:]*1j
            _, idx = np.unique(rot_end_uniq, return_index=True)
            rot_end = rot_end[:, idx]
            # 平移到物体中心
            rot_end[0,:] = rot_end[0,:] + center_x + max_line
            rot_end[1,:] = rot_end[1,:] + center_y + max_line
            # 再截取mask
            template[rot_end[1,:], rot_end[0,:]] = 255
            template[mask_x, mask_y] = 0
            # 需要多加一圈像素，防止射线全部被mask掉
            mask_cut = template[max_line - 1:max_line + img_h + 1, max_line - 1:max_line + img_w + 1]
            # 找出端点
            marker_y, marker_x = np.where(mask_cut == 255)
            dist_center = np.sqrt(np.power(marker_x - center_x,2) + np.power(marker_y - center_y,2))
            final_idx = np.argmin(dist_center)
            # 最终的点如果在多加的一圈像素里，需要平移到原始图片大小的尺寸中
            x_final = np.clip(marker_x[final_idx], 0, img_w)
            y_final = np.clip(marker_y[final_idx], 0, img_h)
            # 坐标和半径记录容器
            cord = np.array([x_final, y_final])
            radius = dist_center[final_idx]
            # 存储数据
            cord_results.append(cord)
            radius_results.append(radius)

        return np.array(cord_results), np.array(radius_results)

    # 对annotations字段进行处理
    # area_t_low和area_t_high表示生成的24点标签数据和原始标签分割物体的面积比例接受阈值
    # 如果24点圈出的物体面积小于标签中标定的面积超过上面两个阈值，就抛弃当前标签
    # 532469
    def json_anno_process(self, area_t_low = 0.5, area_t_high = 1.5):        
        # 获取标签信息
        anno_info = self.json_dict["annotations"]
        numb_anno = len(anno_info)        
        # 进度条显示
        with tqdm(total=numb_anno) as pbar:
            # 对每一个标签进行处理
            for anno_numb, anno in enumerate(anno_info):
                cur_image_name = str(anno["image_id"]).zfill(12)
                pbar.set_description('Image_id: {}, anno_numb: {}'.format(cur_image_name, anno_numb))
                # 如果现有的存储容器中没有当前图片的处理标记结果，就创建该图片项目
                if cur_image_name in self.label_dict_cord24:
                    pass
                else:
                    self.label_dict_cord24[cur_image_name] = []

                if cur_image_name in self.label_dict_radius:
                    pass
                else:
                    self.label_dict_radius[cur_image_name] = []
                # 标签仅仅识别个体情况，放弃集群
                if (not anno["iscrowd"]):
                    # 读取标签中当前区域的面积，面积小于1的物体不做处理
                    label_area = anno["area"]
                    if (label_area < 1):
                        pbar.update(1)
                        continue
                    # 读取当前目标的类别信息
                    class_id = anno["category_id"]
                    # 需要对class和label做一个转换
                    label_id = np.array([self.coco_id2idx[str(class_id)]])
                    # 当前条目的信息处理
                    image_pth = Path(self.image_data_pth) / Path(cur_image_name + ".jpg")
                    # 如果当前路径下的文件存在,就读取该图片，如果不存在，就跳过该条目
                    if (os.path.exists(image_pth)):
                        image_ori = cv2.imread(str(image_pth))
                    else:
                        pbar.update(1)
                        continue
                    img_h, img_w = image_ori.shape[0], image_ori.shape[1]
                    img_diag = np.sqrt(np.power(img_h, 2) + np.power(img_w, 2))
                    # 读取目标左上角坐标，转换成中心坐标
                    obj_x = anno["bbox"][0] + anno["bbox"][2]/2
                    obj_y = anno["bbox"][1] + anno["bbox"][3]/2
                    np.clip(obj_x, 0, img_w)
                    np.clip(obj_y, 0, img_h)
                    # 计算当前条目的mask数据,活跃区mask数值为1，其余为0
                    cur_mask = self.coco.annToMask(anno)
                    # 计算当前mask相对于中心的24点边沿距离
                    cur_24p, cur_24r = self.rotation_for_24p(obj_x, obj_y, cur_mask)
                    # 将24r距离归一化，归一化尺寸是图片的对角线长度
                    cur_24r = cur_24r / img_diag
                    # 计算24点的凸包，并计算凸包的面积，如果面积和标注的差太多就抛弃
                    hull = cv2.convexHull(cur_24p)
                    hull_area = cv2.contourArea(hull)
                    # 比例设定
                    if (hull_area <= label_area * area_t_low) or (hull_area >= label_area * area_t_high):
                        pbar.update(1)
                        continue
                    else:
                        # 24个半径数据顺序：以x轴为起点，顺时针24个，每个间隔15°
                        obj_cord = np.array([obj_x/img_w, obj_y/img_h])
                        cur_24p = cur_24p.reshape(1, -1).squeeze(0).astype(np.float32)
                        cur_24p[0::2] = cur_24p[0::2]/img_w
                        cur_24p[1::2] = cur_24p[1::2]/img_h
                        label_info_cord24 = np.concatenate((label_id, obj_cord, cur_24p), axis = 0)
                        label_info_radius = np.concatenate((label_id, obj_cord, cur_24r), axis = 0)
                        self.label_dict_cord24[cur_image_name].append(label_info_cord24)  
                        self.label_dict_radius[cur_image_name].append(label_info_radius)
                    # 结果可视化
                    # self.show_24p(image_ori, hull, cur_24p)
                # 如果是集群，直接放弃当前这条标签 
                else:
                    pass
                # 更新进度条
                pbar.update(1)
        # 返回信息
        return self.label_dict_cord24, self.label_dict_radius

    # 可视化24点和凸包信息
    def show_24p(self, image, hull_points, points_24):
        length = len(hull_points)
        for i in range(len(hull_points)):
            cv2.line(image, tuple(hull_points[i][0]), tuple(hull_points[(i+1)%length][0]), (0,255,0), 2)

        for p in points_24:
            cv2.circle(image, p, 3, (255,0,0), -1)

        cv2.imshow("test1", image)
        cv2.waitKey(0)
    
    # 保存信息到txt文件
    def save_24r_to_txt(self):
        if (self.mode == "Cord"):
            label_dict = self.label_dict_cord24
            # 保存信息
            format_txt = ["%d"] + ["%0.4f"]*50
        else:
            label_dict = self.label_dict_radius
             # 保存信息
            format_txt = ["%d"] + ["%0.4f"]*26
        # 进度条显示
        numb_label = len(label_dict)
        with tqdm(total=numb_label) as pbar:   
            for image_numb in label_dict:
                # 创建txt标签的路径
                txt_pth = Path(self.new_label_pth) / Path(image_numb + ".txt")
                # 数组转换
                label_info = np.array(label_dict[image_numb])
                # 如果信息不是空
                if label_info.shape[0]:
                    np.savetxt(str(txt_pth), label_info, fmt=format_txt)
                else:
                    np.savetxt(str(txt_pth), label_info)
                pbar.update(1)

if __name__ == "__main__":
    polygon = Polygon_24()
    polygon.json_anno_process()
    polygon.save_24r_to_txt()