import os
import torch
import torch.distributed as dist
import cv2
import numpy as np

from exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)

        self.num_classes = 80
        self.data_num_workers = 4
        self.exp_name = "yolox_24p"
        
    # 获取训练数据集
    def get_data_input(self, image_path):
        # 读取文件        
        image = cv2.imread(image_path)
        assert image is not None 
        height, width = image.shape[0], image.shape[1]
        # resize 图片
        self.ratio = min(self.input_size[0] / height, self.input_size[1] / width)
        resized_img = cv2.resize(image, (int(width * self.ratio), int(height * self.ratio)), interpolation=cv2.INTER_LINEAR,).astype(np.uint8)
        # padding 图片
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        # 将RGB图片赋值给padding图片
        padded_img[: int(height * self.ratio), : int(width * self.ratio)] = resized_img
        padded_img_trans = padded_img.transpose(2, 0 , 1)[np.newaxis, :]
        padded_img_trans = np.ascontiguousarray(padded_img_trans, dtype=np.float32)
        
        in_model_image = torch.tensor(padded_img_trans)                
        
        return in_model_image, self.ratio, image