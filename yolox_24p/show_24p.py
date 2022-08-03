import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision

from exp import get_exp
from utils import save_checkpoint
from tqdm import tqdm
from models import Loss_Function

from torch.utils.tensorboard import SummaryWriter

count_iter = 0

class Evaluator:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args

        self.num_classes = self.exp.num_classes
        self.start_device = self.args.start_device
        self.numb_device = self.args.devices
        self.device = "cuda" 

        self.input_size = self.exp.input_size
        self.file_name = os.path.join(self.exp.output_dir, self.exp.exp_name)

        os.makedirs(self.file_name, exist_ok=True)

        self.file_list = os.listdir(self.args.load_path)
        
        self.model_weight_path = self.args.weights

        self.COLORS = np.array(
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

        self.COCO_CLASSES = (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        )

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        
        theta = torch.tensor(15*np.pi/180, device=prediction.device)
        theta_all = torch.arange(24, device=prediction.device) * theta
        cos_theta_all = theta_all * torch.cos(theta_all)
        sin_theta_all = theta_all * torch.sin(theta_all)

        box_corner = prediction.new(prediction.shape)

        box_corner[:, :, :26] = prediction[:, :, :26]

        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue
            class_conf, class_pred = torch.max(image_pred[:, 27: 27 + num_classes], 1, keepdim=True)


            conf_mask = (image_pred[:, 26] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :27], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
        
            if not detections.size(0):
                continue
            
            cos_theta_all = cos_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)
            sin_theta_all = sin_theta_all.unsqueeze(0).repeat(detections.shape[0], 1)

            p24_x = detections[:, 2:26] * cos_theta_all + detections[:, 0].unsqueeze(1).repeat(1, 24)
            p24_y = detections[:, 2:26] * sin_theta_all + detections[:, 1].unsqueeze(1).repeat(1, 24)


            p24_x_min = p24_x.min(dim=1).values
            p24_x_max = p24_x.max(dim=1).values
            p24_y_min = p24_y.min(dim=1).values
            p24_y_max = p24_y.max(dim=1).values
            
            p24_rect = torch.stack((p24_x_min, p24_y_min, p24_x_max, p24_y_max), dim=0).transpose(0, 1)

            nms_out_index = torchvision.ops.nms(
                p24_rect,
                detections[:, 26] * detections[:, 27],
                nms_thre,
            )

            detections = detections[nms_out_index]
            
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    @torch.no_grad()
    def eval(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        use_device = ''
        for gpu in range(self.start_device, self.start_device + self.numb_device):
            use_device = str(gpu) + use_device
        os.environ['CUDA_VISIBLE_DEVICES'] = use_device

        self.model = self.exp.get_model()

        weights_file = torch.load(self.model_weight_path, map_location=self.device)
        self.model.load_state_dict(weights_file["model"])

        logger.info("Evaluating start...")
        logger.info("\n{}".format(self.model))
       
        with tqdm(total = len(self.file_list), desc='processing:', ncols=70) as t:
            self.model.to(self.device)
            self.model.eval()

            current_time = time.localtime()
            self.save_folder = os.path.join(self.exp.output_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(self.save_folder, exist_ok=True)   

            for file in self.file_list:
                img_path = os.path.join(self.args.load_path, file)

                eval_img, ratio, ori_image = self.exp.get_data_input(img_path)

                images = eval_img.to(self.device)

                outputs = self.model(images)

                outputs = self.postprocess(outputs, self.num_classes, conf_thre=0.01, nms_thre=0.3)

                self.save_eval_results(ori_image, outputs[0], ratio, file, cls_conf=0.0001)    
                
                t.update()

    def save_eval_results(self, image, output, ratio, image_name, cls_conf=0.01):
        save_file_name = os.path.join(self.save_folder, os.path.basename(image_name))

        if output is None:
            logger.info("No Detection Results, Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, image)
        else:
            bboxes = output[:, 0:26]

            bboxes /= ratio
            cls = output[:, 28]
            scores = output[:, 26] * output[:, 27]
            
            vis_results = self.vis(image, bboxes, scores, cls, cls_conf, self.COCO_CLASSES)
            
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, vis_results)

    def vis(self, img, boxes, scores, cls_ids, conf, class_names=None):
        theta = 15*np.pi/180
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x_center = int(box[0])
            y_center = int(box[1])
            r_start = box[2:].to(torch.int)
            
            color = (self.COLORS[cls_id] * 255).astype(np.uint8).tolist()
            
            text = '{}'.format(class_names[cls_id])

            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
           
            cv2.circle(img, (x_center, y_center), 4, color, -1)

            for idx in range(len(r_start)):
                rot_theta = theta * idx
                cur_x = torch.tensor(x_center + r_start[idx] * np.cos(rot_theta))
                cur_y = torch.tensor(y_center + r_start[idx] * np.sin(rot_theta))

                cur_x = int(torch.clip(cur_x, 0, img.shape[1]))
                cur_y = int(torch.clip(cur_y, 0, img.shape[0]))             
                cv2.circle(img, (cur_x, cur_y), 2, color, -1)
                
                if idx == 0:
                    start_x = cur_x
                    start_y = cur_y
                else:
                    cv2.line(img, (cur_x, cur_y), (last_x, last_y), color, 2)
                
                last_x = cur_x
                last_y = cur_y

            cv2.line(img, (last_x, last_y), (start_x, start_y), color, 2)
            cv2.putText(img, text, (x_center + 3, y_center - txt_size[1]), font, 0.6, color, thickness=2)

        return img


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")    

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")

    parser.add_argument("-s", "--start_device", default=0, type=int, help="device for start count")
    
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file")

    parser.add_argument("-p", "--load_path", type=str, default=None, help="plz input your file path")

    parser.add_argument("-w", "--weights", type=str, default=None, help="plz input your weights path")

    return parser

@logger.catch
def main(exp, args):
    eval = Evaluator(exp, args)
    eval.eval()

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
        
