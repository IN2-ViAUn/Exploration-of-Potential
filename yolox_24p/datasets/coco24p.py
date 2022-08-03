import os
from loguru import logger
import warnings
import cv2
import numpy as np
import torch

from pycocotools.coco import COCO


class COCO24PDataset(torch.utils.data.Dataset):
    """
    COCO 24P dataset class.
    """

    def __init__(self, img_size=(640, 640), preproc=None):
        super().__init__()
        
        self.data_dir = "/home/gaoyu/COCO/images/val2017"
        self.label_dir = "/home/gaoyu/COCO/labels/val2017_24XY"

        self.coco24p_dict, self.image_list = self.load_label_from_txt()
        self.item_numb = len(self.coco24p_dict)
        self.imgs = None
        self.resize_info = None
        self.imgs_shape = None
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return self.item_numb

    def __del__(self):
        del self.imgs

    # 从txt读数据
    def load_label_from_txt(self):
        label_dict = {}
        image_list = []
        logger.info("Loading txt files...")
        # 标签文件读取
        label_files = os.listdir(self.label_dir)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for label in label_files:
                info = np.loadtxt(os.path.join(self.label_dir, label), dtype=float)
                label_dict[label.split(".")[0]] = info
                image_list.append(label.replace("txt", "jpg"))
        logger.info("Loading txt files successed")
        return (label_dict, image_list)
    
    # 从图片存储路径中读取单张图片
    def load_image(self, img_name):
        img_file = os.path.join(self.data_dir, img_name)
        img = cv2.imread(img_file)
        assert img is not None 
        height, width = img.shape[0], img.shape[0]
        return img, height, width
    
    # 返回resize后的图像
    def load_resized_img(self, img_name):
        img, ori_h, ori_w = self.load_image(img_name)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img, r, ori_h, ori_w

    # 读取标签信息
    def load_anno(self, index):
        img_name = self.image_list[index]
        dict_key = img_name.split(".")[0]
        return self.coco24p_dict[dict_key]
            
    # 单条信息的获取
    def pull_item(self, index):
        # 获取图像的名字
        img_name = self.image_list[index]
        dict_key = img_name.split(".")[0]
        # 获取标签信息
        label_info = self.coco24p_dict[dict_key]
        # 如果尺度是1，就拓展一个维度
        if (len(label_info.shape) == 1):
            label_info = label_info[np.newaxis, :]

        # 图片id信息
        img_id = int(dict_key)

        # 如果用的是cache，
        if self.imgs is not None:
            pad_img = self.imgs[index]
            # 读取图像的原始尺寸
            img_info = (self.imgs_shape[index][0], self.imgs_shape[index][1])
            resized_info = (int(img_info[0] * self.resize_info[index]), int(img_info[1] * self.resize_info[index]))
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()

        # 其余情况直接接收resize图像 
        else:
            img, r, ori_h, ori_w = self.load_resized_img(img_name)
            img_info = (ori_h, ori_w)

        return img, label_info, img_info, np.array([img_id])

    # @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 27]`.
                each label consists of [class, xc, yc, r1, r2, ...., r24]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, [640,640])
        
        return img, target, img_info, img_id
