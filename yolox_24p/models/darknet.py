#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

# 新加入ResNet  对标yolox-l(256,512,1024)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1  # 最后FC层使用
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # 只有第一个卷积层才会有stride
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # copy一份x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  # 输入通道数，输出通道数
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        out_features=("dark3", "dark4", "dark5")
        self.out_features = out_features

        # 设置标准化层，默认为BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32   # 第一个卷积卷积核数量，后续各个stage，用planes控制通道数，64-128-256-512
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group # 64
        
        # stage: 1  7*7卷积 输入3，输出64，stride为2，缩小一半
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3*3池化，stride为2，缩小一半
        # stage: 2-5  planes代表输出通道数
        self.layer1 = self._make_layer(block, 32, layers[0])  # basic, 64, 2  # 输入64，输出64
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]) # basic, 128, 2, 2, false  # 输入64，输出128
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) # basic, 256, 2, 2, false  # 输入64，输出256
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]) # basic, 512, 2, 2, false  # 输入64，输出512
        
        # stage:6
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # block.expansion=1 
        
        # baseconv只有50/101/152需要
        self.baseconv1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.SiLU()
                                       )

        self.baseconv2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm2d(256),
                               nn.SiLU()
                               )

        self.baseconv3 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.SiLU()
                                       )  # 改掉YOLOX原来backbone中对应的即可

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer  #  nn.BatchNorm2d
        downsample = None  
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # shortcut connection 的B策略，当分辨率变化时，采用1*1卷积进行变换特征图分辨率
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        # 添加第一个building block，由于特征图分辨率下降在第一个building block中进行，因此这一个block比较特别，单独拿出来添加
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        
        # 更新 self.inplanes 
        self.inplanes = planes * block.expansion
        
        # 添加其余building block
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        outputs = {}
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)  # torch.Size([16, 64, 112, 112])
        outputs["stem"] = x
        
        x = self.maxpool(x)
        x = self.layer1(x)
        # print(x.shape)  # torch.Size([16, 256, 56, 56])
        outputs["dark2"] = x

        x = self.layer2(x)
        # print(x.shape)  # torch.Size([16, 512, 28, 28])
        outputs["dark3"] = x

        x = self.layer3(x)
        # print(x.shape)  # torch.Size([16, 1024, 14, 14])
        outputs["dark4"] = x
        
        x = self.layer4(x)
        # print(x.shape)  # torch.Size([16, 2048, 7, 7])
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        # return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model
    
def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs) 

# 新加入VGG  对标yolox-l(256,512,1024)
class ConvBNReLU(nn.Module):
    # **kwargs针对关键字参数,且不限制数量
    # *args针对非关键字参数，且不限制数量
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
    # forward方法会自动调用
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG(nn.Module):
    def __init__(self, layer):
        super(VGG, self).__init__()

        out_features=("dark3", "dark4", "dark5")
        self.out_features = out_features

        # 通道数变化过程：3 --> 64 --> 64*2 --> 64*4 --> 64*8 --> 64*16
        base_channels = 64
        
        # VGG中反复用到的卷积基本模块
        block = ConvBNReLU
        
        # _make_layer(基本模块，入通道数，出通道数，基本模块个数)
        self.conv_pool1 = self._make_layer(block, 3, base_channels, layer[0])
        self.conv_pool2 = self._make_layer(block, base_channels, base_channels*2, layer[1])
        self.conv_pool3 = self._make_layer(block, base_channels*2, base_channels*4, layer[2])
        self.conv_pool4 = self._make_layer(block, base_channels*4, base_channels*8, layer[3])
        self.conv_pool5 = self._make_layer(block, base_channels*8, base_channels*8, layer[4])
        
        # 保持vgg原版的卷积-池化结构不变，在最后加入一层1*1conv，只为了改变通道数512-->1024
        self.conv_add = block(base_channels*8, base_channels*16, kernel_size=1, bias=False)
    
    # VGG中的卷积-池化模块
    def _make_layer(self, block, in_channels, out_channels, layer):
        layers = []
        # 添加第一个卷积层，输入通道数和输出通道数是不一样的
        layers.append(block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        # 添加第二个至最后一个卷积层，输入通道数和输出通道数是一样的
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        # 添加最大池化层，将特征图尺寸缩小一半
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        outputs = {}
        x = self.conv_pool1(x)
        # print(x.shape)
        outputs["stem"] = x
        x = self.conv_pool2(x)
        # print(x.shape)
        outputs["dark2"] = x
        x = self.conv_pool3(x)
        # print(x.shape)
        outputs["dark3"] = x
        x = self.conv_pool4(x)
        # print(x.shape)
        outputs["dark4"] = x
        x = self.conv_pool5(x)
        # print(x.shape)
        x = self.conv_add(x)
        # print(x.shape)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
        # return x
        
def _vgg(layer, **kwargs):
    model = VGG(layer, **kwargs)
    return model 

def vgg16(**kwargs):
    return _vgg([2, 2, 3, 3, 3], **kwargs)

def vgg19(**kwargs):
    return _vgg([2, 2, 4, 4, 4], **kwargs)

# 新加入DenseNet
# 基本卷积模块（conv/bn/relu)
# 为了区别于line7的BaseConv，故命名为BaseConv_DN
class BaseConv_DN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv_DN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# DenseLayer中用到的卷积模块（bn/relu/conv）
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

# 定义过渡块，主要负责连接多个Denseblock
# 1*1卷积，2*2average池化
# 输出通道数是输入通道数的1/2
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.trans = nn.Sequential(ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                   nn.AvgPool2d(kernel_size=2, stride=2)
                                )

    def forward(self, x):
        return self.trans(x)

# DenseLayer: 1*1conv(out_c=4*k)  3*3conv(out_c=k)
class DenseLayer(nn.Module):
    def __init__(self, in_channels, drop_rate = 0):
        super(DenseLayer, self).__init__()
        # k=32
        self.growth_rate = 32
        # 系数因子
        self.bn_size = 4
        self.conv_block = nn.Sequential(ConvBlock(in_channels, self.bn_size*self.growth_rate, kernel_size=1, stride=1, bias=False),
                                        ConvBlock(self.bn_size*self.growth_rate, self.growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
                                        )
        self.drop_rate = float(drop_rate)
        self.dropout = nn.Dropout2d(self.drop_rate)

    def forward(self, x):
        x = self.conv_block(x)
        if (self.drop_rate) > 0:
            x = self.dropout(x)
        return x

# Denseblock块，num_layers表示每个块中DenseLayer的个数
# in_channels表示Transition块输出的通道数
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, drop_rate = 0):
        super(DenseBlock, self).__init__()
        # k=32
        self.growth_rate = 32
        # num_layers = 6, 12, 24, 16     
        layers = []  
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i*self.growth_rate, drop_rate=drop_rate))
        self.denseblock = nn.Sequential(*layers)
            
    # 在正向传播过程中定义cat操作
    def forward(self, x):
        for layer in self.denseblock:
            x_ = layer(x)
            x = torch.cat((x, x_), dim=1)
        return x

# growth_rate = 32 (k=32)
# block_config = [6, 12, 24, 16] (DenseBlock块中DenseLayer的个数)
# num_init_features = 64 (初始层卷积核个数)
class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_layer):
        super(DenseNet, self).__init__()

        out_features=("dark3", "dark4", "dark5")
        self.out_features = out_features
        
        #定义网络的基本参数
        self.growth_rate = growth_rate
        self.block_config = block_layer
        self.num_init_channels = 64
        
        # 7*7 conv,3*3 max pool 四倍下采样：224*224*3 --> 112*112*64 --> 56*56*64
        self.stem = nn.Sequential(BaseConv_DN(3, self.num_init_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )

        # 每个Denseblock块 Transition块 之间通道数的计算桥梁
        num_channels = self.num_init_channels
        
        # Transition块的输入通道数
        T1_channels = num_channels + self.block_config[0] * self.growth_rate  # 64+6*32=256
        T2_channels = T1_channels//2 + self.block_config[1] * self.growth_rate  # 128+12*32=512
        T3_channels = T2_channels//2 + self.block_config[2] * self.growth_rate  # 256+24*32=1024

        self.D1 = DenseBlock(self.block_config[0], num_channels, drop_rate=0.3)  # 64-->256
        self.T1 = Transition(T1_channels, T1_channels//2)  # 256-->128=256/2
        self.D2 = DenseBlock(self.block_config[1], T1_channels//2, drop_rate=0.3)  # 128-->512
        self.T2 = Transition(T2_channels, T2_channels//2)  # 512-->256=512/2
        self.D3 = DenseBlock(self.block_config[2], T2_channels//2, drop_rate=0.3)  # 256-->1024
        self.T3 = Transition(T3_channels, T3_channels//2)  # 1024-->512=1024/2
        self.D4 = DenseBlock(self.block_config[3], T3_channels//2, drop_rate=0.3)  # 512-->1024

        self.baseconv1 = BaseConv_DN(T2_channels, T2_channels//2, kernel_size=1, bias=False)
        self.baseconv2 = BaseConv_DN(T3_channels, T3_channels//2, kernel_size=1, bias=False)

    def forward(self, x):
        outputs = {}
        # print(x.shape)
        x = self.stem(x)
        # print(x.shape)
        outputs["stem"] = x
        x = self.D1(x)
        # print(x.shape)
        outputs["dark2"] = x
        x = self.T1(x)
        # print(x.shape)
        x = self.D2(x)
        # print(x.shape)
        x1 = self.baseconv1(x)
        # print(x1.shape,'************')
        outputs["dark3"] = x1
        x = self.T2(x)
        # print(x.shape)
        x = self.D3(x)
        # print(x.shape)
        x2 = self.baseconv2(x)
        # print(x2.shape,'************')
        outputs["dark4"] = x2
        x = self.T3(x)
        # print(x.shape)
        x = self.D4(x)
        # print(x.shape,'************')
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
        # return x

# 输入参数为：增长率（K），网络denseblock中包含的层数
def _densenet(growth_rate, block_layer):
    model = DenseNet(growth_rate, block_layer)
    return model

def densenet121():
    return _densenet(32, [6, 12, 24, 16])








