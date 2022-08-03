import torch.nn as nn

from .yolo_head_24p import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, train=False):
        # fpn output content features of [dark3, dark4, dark5]
        # 备注：s/m/n/l会影响fpn的通道宽度
        # 以s网络为例
        # pan_out0:[1, 512, 20, 20](fpn_outs[2])
        # pan_out1:[1, 256, 40, 40](fpn_outs[1])
        # pan_out2:[1, 128, 80, 80](fpn_outs[0])
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs, train)
        
        return outputs
        