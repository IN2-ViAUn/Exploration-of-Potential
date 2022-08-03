"""
读取json文件,生成相应内容的mask
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
from matplotlib import pyplot as plt
from skimage import io

class Polygon_24():
    # mode有Cord和Radius两种，Cord表示存储24点坐标，Radius表示存储24半径
    def __init__(self, mode = "Cord"):
        # 存储数据模式
        self.mode = mode
        # 标签json文件路径
        self.json_label_pth = "/home/gaoyu/COCO/json/instances_val2017.json"
        # 图片文件路径
        self.image_data_pth = "/home/gaoyu/COCO/images/val2017_bk"
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
        # 颜色信息
        self.colors = np.array(
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

    # 载入标签json文件
    def load_label_json(self):
        with open(self.json_label_pth,'r') as load_f:
            load_dict = json.load(load_f)
        return load_dict

    # 对annotations字段进行处理, 输入图像名称
    def json_anno_process(self, img_name):        
        # 获取标签信息
        anno_info = self.json_dict["annotations"]
        # 当前image的id和对应mask缓存
        id_mask = {}
        # 对每一个标签进行处理
        for anno in anno_info:
            if img_name != str(anno["image_id"]).zfill(12):
                pass
            else:
                # 标签仅仅识别个体情况，放弃集群
                if (not anno["iscrowd"]):
                    # 读取标签中当前区域的面积，面积小于1的物体不做处理
                    label_area = anno["area"]
                    if (label_area < 1):
                        continue
                    # 读取当前目标的类别信息
                    class_id = anno["category_id"]
                    # 需要对class和label做一个转换
                    label_id = int(self.coco_id2idx[str(class_id)])
                    # 当前条目的信息处理
                    image_pth = Path(self.image_data_pth) / Path(img_name + ".jpg")
                    # 如果当前路径下的文件存在,就读取该图片，如果不存在，就跳过该条目
                    if (os.path.exists(image_pth)):
                        image_ori = cv2.imread(str(image_pth))
                    else:
                        continue
                    # 计算当前条目的mask数据,活跃区mask数值为1，其余为0
                    cur_mask = self.coco.annToMask(anno)*255
                    # 保存mask和id
                    if label_id in id_mask:
                        id_mask[label_id] += [cur_mask]
                    else:
                        id_mask[label_id] = [cur_mask]

        # 返回信息
        return image_ori, id_mask

    # 绘制mask边界信息
    def show_mask(self, image_ori, id_mask, name):
        for id in id_mask:
            for mask in id_mask[id]:
                # 寻找mask的边界信息
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
                cur_color = (self.colors[id] * 255).astype(np.uint8).tolist()
                # 在原图上绘制mask
                cv2.drawContours(image_ori, contours,-1, cur_color, 2)
                # cv2.imshow("contours", image_ori)
                # cv2.waitKey(0)
        cv2.imwrite("./" + name + ".jpg", image_ori)

if __name__ == "__main__":
    polygon = Polygon_24()
    name = "000000014473"    
    image, id_mask = polygon.json_anno_process(name)
    polygon.show_mask(image, id_mask, name)