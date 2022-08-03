
import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module

from utils import LRScheduler

# 定义抽象基类具备的基本功能
class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        # 输出实验数据的路径
        self.output_dir = "./YOLOX_outputs"
        # 间隔打印结果的轮数
        self.print_interval = 100
        # 间隔进行结果测试的轮数
        self.eval_interval = 10

    # 获取模型，返回类型为Module
    @abstractmethod
    def get_model(self) -> Module:
        pass
    
    # 获取数据加载器，返回类型为 Dict[str, torch.utils.data.DataLoader]
    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass
    
    # 获取优化器，返回类型为 torch.optim.Optimizer
    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass
    
    # 获取学习速率调整信息，返回类型为 LRScheduler
    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass
    
    # # 获取评估器，返回类型未指定
    # @abstractmethod
    # def get_evaluator(self):
    #     pass
    
    # # 评估函数
    # @abstractmethod
    # def eval(self, model, evaluator, weights):
    #     pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
