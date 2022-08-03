import argparse

import torch
import torch.backends.cudnn as cudnn
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from exp import get_exp
from utils import save_checkpoint
from tqdm import tqdm
from models import Loss_Function
from loguru import logger

from torch.utils.tensorboard import SummaryWriter

count_iter = 0

class Trainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args

        self.max_epoch = exp.max_epoch
        self.L1_epoch = exp.L1_epoch

        self.start_device = self.args.start_device
        self.numb_device = self.args.devices
        self.device = "cuda" 

        self.input_size = exp.input_size
        self.best_ap = 0
        self.file_name = os.path.join(exp.output_dir, exp.exp_name)

        self.current_total_loss = 0
        self.current_iou_loss = 0
        self.current_conf_loss = 0
        self.current_cls_loss = 0
        self.current_step = 0

        os.makedirs(self.file_name, exist_ok=True)

        self.train_loader = self.exp.get_data_loader(self.args.batch_size)

        self.loss_func = Loss_Function(self.exp.num_classes)
        
    def train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        use_device = ''
        for gpu in range(self.start_device, self.start_device + self.numb_device):
            use_device = str(gpu) + use_device
        os.environ['CUDA_VISIBLE_DEVICES'] = use_device

        model = self.exp.get_model()

        model.to(self.device)
        self.model = model

        self.optimizer = self.exp.get_optimizer(self.args.learn_rate)
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size)
        self.max_iter = len(self.train_loader)
        
        self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.model.train()
            running_loss = 0.0
            ave_loss = 0.0

            with tqdm(total = self.max_iter, desc='epoch:{}'.format(self.epoch), ncols=70) as t:
                for step, data in enumerate(self.train_loader, start=0):
                    self.current_step += 1

                    images, labels, img_info, img_id = data                

                    self.optimizer.zero_grad()

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    labels.requires_grad = False
                    images, labels = self.exp.preprocess(images, labels, self.input_size)

                    outputs = self.model(images, train = True) 

                    loss_all = self.loss_func.forward(outputs, labels)

                    total_loss = loss_all[0]

                    pred_res = loss_all[6]

                    total_loss.backward()

                    self.optimizer.step()

                    running_loss += total_loss.item()
                    ave_loss = running_loss/(step+1)

                    self.TB_data(loss_all)

                    t.set_postfix_str('AveLoss:{:^7.3f}'.format(ave_loss))
                    t.update()

                self.save_model()

    def TB_data(self, loss_all):
        self.current_total_loss = loss_all[0]
        self.current_iou_loss = loss_all[1]
        self.current_conf_loss = loss_all[2]
        self.current_cls_loss = loss_all[3]
        pred_res = loss_all[6]

        reg_w = pred_res[3]
        obj_w = pred_res[4]
        cls_w = pred_res[5]

        for i in range(self.current_iou_loss.shape[0]):
            self.tblogger.add_scalar("Loss/IOU_Loss_{}".format(i), self.current_iou_loss[i], self.current_step)
        
        self.tblogger.add_scalar("Loss/Total_Loss", self.current_total_loss, self.current_step)
        self.tblogger.add_scalar("Loss/Confi_Loss", self.current_conf_loss, self.current_step)
        self.tblogger.add_scalar("Loss/Class_Loss", self.current_cls_loss, self.current_step)

        for i in range(reg_w.shape[0]):
            self.tblogger.add_scalar("Weights/iou_w_{}".format(i), reg_w[i], self.current_step)
        
        self.tblogger.add_scalar("Weights/obj_w", obj_w, self.current_step)
        self.tblogger.add_scalar("Weights/cls_w", cls_w, self.current_step)
    
    def save_model(self):
        self.save_ckpt("last_epoch")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        save_model = self.model
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )

    def show_train_results(self, images, labels, pred_res, idx = 0, inter_step = 10):
        if (self.current_step % inter_step == 0):
            label = labels[idx]
            pd_center_x = pred_res[0][idx]
            pd_center_y = pred_res[1][idx] 
            scale_pd    = pred_res[2][idx]

            img_test = images[idx].permute(1,2,0).cpu().numpy().astype(np.uint8).copy()            
            label_test = label.cpu().numpy()[0]

            cv2.circle(img_test, (int(label_test[1]),  int(label_test[2])), 5, (0, 0, 255), -1)
            cv2.circle(img_test, (int(pd_center_x),  int(pd_center_y)), 3, (255, 0, 0), -1)

            label_x = label_test[3::2]
            label_y = label_test[4::2]

            for i in range(24):
                cv2.circle(img_test, (int(label_x[i]), int(label_y[i])), 5, (0, 0, 255), -1)
                cv2.circle(img_test, (int(pd_center_x), int(pd_center_y)), int(scale_pd[i]), (255, 0, 0), 2)

            img_test = img_test[:,:,::-1]
            self.tblogger.add_image("Image/test1", img_test, dataformats='HWC')


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")    
    
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="batch size")
    
    parser.add_argument("-l", "--learn_rate", type=float, default=0.001, help="learn rate")

    parser.add_argument("-s", "--start_device", default=0, type=int, help="device for start count")
    
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file")

    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")

    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")

    return parser

@logger.catch
def main(exp, args):
    trainer = Trainer(exp, args)
    trainer.train()

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
        
