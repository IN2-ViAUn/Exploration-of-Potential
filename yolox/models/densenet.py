import torch
from torch import nn

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

