import os
import torch.distributed as dist
from exp import Exp as MyExp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        
        self.num_classes = 80
        self.max_epoch = 2000
        self.L1_epoch = 100
        self.data_num_workers = 4
        # self.eval_interval = 1
        self.exp_name = "yolox_24p"

    