import torch
import torch.nn as nn

class BasicBlock(nn.Module):  # 定义一个BasicBlock类，继承nn.Module, 适用于resnet18、34的残差结构
    expansion = 1  # 定义扩展系数为1，主分支的卷积核个数不改变

    # 初始化函数，定义网络的层和参数
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 传入输入通道数，输出通道数，卷积核大小默认为3
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入x, 便于后面进行残差连接
        if self.downsample is not None:  # 如果需要下采样，则对输入进行下采样得到捷径分支的输出
            identity = self.downsample(x)  # 对输入x进行下采样

        out = self.conv1(x)  # 主分支的第一层卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 激活函数ReLU

        out = self.conv2(out)  # 主分支的第二层卷积
        out = self.bn2(out)  # 批归一化
        out += identity  # 将输出与残差连接相加
        out = self.relu(out)  # 激活函数ReLU

        return out  # 返回输出
    

class Bottleneck(nn.Module):  # 定义一个Bottleneck类，继承nn.Module, 适用于resnet50及以上的残差结构
    expansion = 4  # 定义扩展系数为4，主分支的卷积核个数最后一层会变为原来的四倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 传入输入通道数，输出通道数，卷积核大小默认为3,定义一个1*1的卷积核用于降维
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample # 下采样层，如果输入和输出的尺寸不匹配，则对其进行下采样

    def forward(self, x):
        identity = x  # 保存输入x, 便于后面进行残差连接
        if self.downsample is not None:  # 如果需要下采样，则对输入进行下采样得到捷径分支的输出
            identity = self.downsample(x)  # 对输入x进行下采样

        out = self.conv1(x)  # 主分支的第一层卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 激活函数ReLU

        out = self.conv2(out)  # 主分支的第二层卷积
        out = self.bn2(out)  # 批归一化
        out = self.relu(out)  # 激活函数ReLU

        out = self.conv3(out)  # 主分支的第三层卷积， 将通道数扩展为原来的四倍
        out = self.bn3(out)  # 批归一化

        out += identity  # 将输出与残差连接相加
        out = self.relu(out)  # 激活函数ReLU

        return out  # 返回输出
    
class ResNet(nn.Module):  # 定义ResNet类，继承nn.Module
    def __init__(self, block, blocks_num, include_top=True, num_classes=10):  # block为残差块类型，blocks_num为每层残差块数量，include_top表示是否包含顶层分类器，num_classes为分类数
        # block为对应网络选取，比如resnet18、34选BasicBlock，resnet50及以上选Bottleneck
        # blocks_num残差结构的数目，比如resnet18=[2,2,2,2], resnet34=[3,4,6,3], resnet50=[3,4,6,3]
        # num_classes分类数，CIFAR-10为10类
        # include_top是否包含顶层分类器
        super(ResNet, self).__init__()
        self.include_top = include_top # 分类头
        self.in_channels = 64  # 初始输入通道数为64
        # 第一层卷积层, 输入通道数为3,若为灰度图像则输入通道数为1
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)  # 第一层卷积，输入通道数为3
        self.bn1 = nn.BatchNorm2d(self.in_channels)  # 批归一化
        self.relu = nn.ReLU()  # 激活函数ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        # 定义四个残差层 block_num在resnet18中为[2,2,2,2]
        self.layer1 = self._make_layer(block, 64, blocks_num[0]) # 创建四个残差层，分别对应resnet的四个stage
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:  # 如果包含顶层分类器
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层，下采样，输出大小为1x1
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，输入特征数为512*block.expansion，输出特征数为num_classes
        
        for m in self.modules():  # 初始化卷积层权重
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, block_num, stride=1): # 创建一个残差层
        # block为对应网络深度来选取
        # channel 为残差结构中第一个卷积层的个数
        # block_num 该层包含多少个残差结构
        downsample = None  # 初始化下采样层为None
        if stride != 1 or self.in_channels != out_channels * block.expansion:  # 如果步幅不为1或者输入通道数不等于输出通道数*扩展系数,则需要下采样
            # 对于layer1的构建，使用resnet18， 不满足条件， downsample=None
            # 对于resnet 50 101 满足条件， 进行下采样，通道数由64变为256
            downsample = nn.Sequential(  # 定义下采样层
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []  # 初始化残差层列表
        # block为选取的basicblock或bottleneck
        layers.append(block(
            self.in_channels, 
            out_channels, 
            stride=stride, 
            downsample=downsample))  # 添加第一个残差块，可能包含下采样
        self.in_channels = out_channels * block.expansion  # 更新输入通道数。特征图已经经过了一残差结构，对18，34来说，通道数没有变化， 对于50，101来说，通道数变为原来的4倍
        
        # 通过循环将一系列实线的残差结构写入进去，无论是resnet18，34还是50，101，从第二层开始都是实线的残差结构
        # 传入输入特征图的通道数和残差结构主分支 上第一层卷积的卷积核的个数
        for _ in range(1, block_num):  # 添加剩余的残差块
            layers.append(block(self.in_channels, out_channels))  # 后续残差块不需要下采样
        return nn.Sequential(*layers)  # 返回一个包含所有残差块的序列容器，将一系列的残差结构组合在一起，得到layer1
    

    def forward(self, x):
        x = self.conv1(x)  # 输入通过第一层卷积
        x = self.bn1(x)  # 批归一化
        x = self.relu(x)  # 激活函数ReLU
        x = self.maxpool(x)  # 最大池化

        x = self.layer1(x)  # 通过第一个残差层
        x = self.layer2(x)  # 通过第二个残差层
        x = self.layer3(x)  # 通过第三个残差层
        x = self.layer4(x)  # 通过第四个残差层

        if self.include_top:  # 如果包含顶层分类器
            x = self.avgpool(x)  # 自适应平均池化
            x = torch.flatten(x, 1)  # 展平为(batch_size, 特征数)
            x = self.fc(x)  # 全连接层分类

        return x  # 返回输出
    
def resnet18(num_classes=10, include_top=True, Pretrained=False):
    return ResNet(BasicBlock, [2,2,2,2],  num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=10, include_top=True, Pretrained=False):
    return ResNet(BasicBlock, [3,4,6,3],  num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=10, include_top=True, Pretrained=False):
    return ResNet(Bottleneck, [3,4,6,3],  num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=10, include_top=True, Pretrained=False):
    return ResNet(Bottleneck, [3,4,23,3],  num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=10, include_top=True, Pretrained=False):
    return ResNet(Bottleneck, [3,8,36,3],  num_classes=num_classes, include_top=include_top)


# ==================== CIFAR-10 专用 ResNet ====================
# 针对 32x32 小图像优化：使用 3x3 卷积代替 7x7，移除初始 MaxPool
class ResNetCIFAR(nn.Module):
    """CIFAR-10 专用 ResNet，适配 32x32 小图像"""
    
    def __init__(self, block, blocks_num, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 64
        
        # CIFAR-10 专用：使用 3x3 卷积，stride=1，无 MaxPool
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# CIFAR-10 工厂函数
def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50_cifar(num_classes=10):
    return ResNetCIFAR(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101_cifar(num_classes=10):
    return ResNetCIFAR(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152_cifar(num_classes=10):
    return ResNetCIFAR(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

