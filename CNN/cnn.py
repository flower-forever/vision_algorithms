import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self, num_class): # num_class分类数
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), #保持图像大小不变 16*32*32 (CIFAR-10)
            nn.ReLU(), #卷积之后接上激活函数，增加非线性特征
            nn.MaxPool2d(kernel_size=2, stride=2),  #池化之后变为16*16*16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), #保持图像大小不变 32*16*16
            nn.ReLU(), #卷积之后接上激活函数，增加非线性特征
            nn.MaxPool2d(kernel_size=2, stride=2)  #池化之后变为32*8*8
        )

        #定义全连接层，做分类
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 128), #输入特征数32*8*8=2048 (适配CIFAR-10)，输出特征数128
            nn.ReLU(),
            nn.Linear(128, num_class) #输出分类数
        )

    def forward(self, x):
        x = self.features(x) # 提取特征
        x = x.view(x.size(0), -1) # 展平为(batch_size, 32*8*8=2048)
        x = self.fc(x) # 分类
        return x
    