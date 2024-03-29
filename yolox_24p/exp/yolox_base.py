
import os
import random

import torch
import torch.nn as nn

from .base_exp import BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.act = 'silu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 8
        self.input_size = (640, 640)  # (height, width)
        # 多尺度训练
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 100
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    # 获取并初始化模型
    def get_model(self):
        from models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    # 获取训练数据集
    def get_data_loader(self, batch_size):
        from datasets import COCO24PDataset, TrainTransform

        self.dataset = COCO24PDataset(
                        img_size=(640, 640),
                        preproc=TrainTransform(max_labels=50,
                            flip_prob=0.5)
                        )
                        
        train_loader = torch.utils.data.DataLoader(
                        self.dataset,
                        batch_size=batch_size,
                        num_workers=self.data_num_workers,
                        pin_memory=True
                        )

        return train_loader

    def random_resize(self, data_loader, epoch):
        tensor = torch.LongTensor(2).cuda()

        size_factor = self.input_size[1] * 1.0 / self.input_size[0]
        if not hasattr(self, 'random_size'):
            min_size = int(self.input_size[0] / 32) - self.multiscale_range
            max_size = int(self.input_size[0] / 32) + self.multiscale_range
            self.random_size = (min_size, max_size)
        size = random.randint(*self.random_size)
        size = (int(32 * size), 32 * int(size * size_factor))
        tensor[0] = size[0]
        tensor[1] = size[1]

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, lr):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum, nesterov=True)
        self.optimizer = optimizer
    
        return self.optimizer
    
    # def get_optimizer(self, batch_size):
    #     if "optimizer" not in self.__dict__:
    #         # warmup轮数
    #         if self.warmup_epochs > 0:
    #             lr = self.warmup_lr
    #         else:
    #             lr = self.basic_lr_per_img * batch_size
            
    #         pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            
    #         for k, v in self.model.named_modules():
    #             if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
    #                 pg2.append(v.bias)  # biases
    #             if isinstance(v, nn.BatchNorm2d) or "bn" in k:
    #                 pg0.append(v.weight)  # no decay
    #             elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
    #                 pg1.append(v.weight)  # apply decay

    #         optimizer = torch.optim.SGD(
    #             pg0, lr=lr, momentum=self.momentum, nesterov=True
    #         )
    #         optimizer.add_param_group(
    #             {"params": pg1, "weight_decay": self.weight_decay}
    #         )  # add pg1 with weight_decay
    #         optimizer.add_param_group({"params": pg2})
    #         self.optimizer = optimizer
    #     print(self.optimizer)
    #     return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    # def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.data import COCODataset, ValTransform

    #     valdataset = COCODataset(
    #         data_dir=self.data_dir,
    #         json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
    #         name="val2017" if not testdev else "test2017",
    #         img_size=self.test_size,
    #         preproc=ValTransform(legacy=legacy),
    #     )

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             valdataset, shuffle=False
    #         )
    #     else:
    #         sampler = torch.utils.data.SequentialSampler(valdataset)

    #     dataloader_kwargs = {
    #         "num_workers": self.data_num_workers,
    #         "pin_memory": True,
    #         "sampler": sampler,
    #     }
    #     dataloader_kwargs["batch_size"] = batch_size
    #     val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    #     return val_loader

    # def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.evaluators import COCOEvaluator

    #     val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
    #     evaluator = COCOEvaluator(
    #         dataloader=val_loader,
    #         img_size=self.test_size,
    #         confthre=self.test_conf,
    #         nmsthre=self.nmsthre,
    #         num_classes=self.num_classes,
    #         testdev=testdev,
    #     )
    #     return evaluator

    # def eval(self, model, evaluator, is_distributed, half=False):
    #     return evaluator.evaluate(model, is_distributed, half)
