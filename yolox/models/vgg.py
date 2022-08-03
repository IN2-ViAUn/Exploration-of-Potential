from torch import nn

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

def vgg19(**kwargs):
    return _vgg([2, 2, 4, 4, 4], **kwargs)
