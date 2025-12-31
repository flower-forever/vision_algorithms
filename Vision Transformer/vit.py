import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

def drop_path(x, drop_prob: float = 0., training: bool = False):
    ''' 
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    随机深度每个样本在残差块的主路径中应用
    这个实现类似于 DropConnect，用于EfficientNet和其他网络架构中，但名字不同，
    DropConnect更常用于指代对权重进行随机丢弃，而DropPath则是对整个路径进行丢弃。
    :param x: 输入张量
    :param drop_prob: 丢弃概率
    :param training: 是否在训练模式下

    :return: 如果不在训练模式或丢弃概率为0，则返回输入张量，否则返回经过drop path处理的张量
    '''
    if drop_prob == 0. or not training: #如果丢弃概率为0或者不在训练模式下，直接返回输入张量
        return x
    keep_prob = 1 - drop_prob #保持路径的概率
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与 x 的维度匹配的形状，除了第一个维度（batch size）外，其他维度为1
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # 生成一个与 x 形状匹配的随机张量
    random_tensor.floor_()  # binarize  将随机张量中的值向下取整，得到0或1的二值张量
    output = x.div(keep_prob) * random_tensor #将输入 x 缩放并与随机张量相乘，实现随机丢弃路径的效果
    return output # 返回经过drop path处理的张量

class Drop_path(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(Drop_path, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        # img——size: 输入图像的大小
        # patch_size: 每个patch的大小
        # 图像是224像素，除以16像素的patch，得到14x14的patch网格
        # in_chans: 输入图像的通道数
        # embed_dim: 每个patch嵌入的维度
        super().__init__()
        img_size = (img_size, img_size) #将输入图像的大小变为二维元组，目的是为了处理高度和宽度
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) #也就是14x14，patch的网格大小
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) #图像转换为3，224，224 -> 768，14，14
        #因为kernel_size=patch_size=16，则卷积核大小为16*16*3=768
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 如果提供了归一化层，则使用它，否则使用恒等映射


    def forward(self, x): #x是特征图
        B, C, H, W = x.shape # 获取输入图像的批量大小、高度和宽度
        # 输入图像的形状为(B, 3, 224, 224)
        # 通过卷积层将图像划分为patches并嵌入到高维空间
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 输出形状为(B, 768, 14, 14)
        x = x.flatten(2)  # 将空间维度展平，输出形状为(B, 768, 196)
        x = x.transpose(1, 2)  # 转置以匹配Transformer的输入格式，输出形状为(B, 196, 768)
        x = self.norm(x)  # 应用归一化层（如果有的话）
        return x  # 返回嵌入后的patches，形状为(B, num_patches, embed_dim)
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        '''
        __init__ 的 Docstring
        
        :param dim: 输入的token维度，为768
        :param num_heads: 注意力的头数，为8
        :param qkv_bias: 是否在qkv线性层中使用偏置，默认为False
        :param qk_scale: qk缩放因子，默认为None
        :param attn_drop: 注意力权重的丢弃率，默认为0.0
        :param proj_drop: 输出投影的丢弃率，默认为0.0
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads #每个头的维度是768/8=96
         #缩放因子，如果没有提供，则默认为head_dim的平方根的倒数
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #通过全连接层生成QKV，为了并行计算，提升计算效率，参数更少
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入dim
        self.proj = nn.Linear(dim, dim)


    def forward(self, x):
        B, N, C = x.shape # B是batch size，N是num_patchs+1，1是class token，C是embed_dim
        # 通过线性层生成QKV，并调整形状以适应多头注意力机制
        # 原先的形状是(B, N, 3*dim)，reshape后变为(B, N, 3, num_heads, C//self.num_heads)
        # permute函数的作用是改变张量的维度顺序,这里将qkv的维度调整为(3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale #计算注意力得分
         # 对注意力得分进行softmax归一化，得到注意力权重
         # 计算qk的点积 【B, num_heads, N, C//num_heads】 -> 【B, num_heads, C//num_heads, N】
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)

        # 计算加权值，注意力权重对V进行加权求和
        #reshape将多头注意力的输出重新调整为(B, N, C)的形状
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # then通过线性变换映射回原本的嵌入dim
        x = self.proj(x)
        x = self.proj_drop(x) # 防止过拟合
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        '''
        __init__ 的 Docstring
        
        :param in_features: 输入维度
        :param hidden_features: 隐藏层维度
        :param out_features: 输出层维度
        :param drop: 丢弃率
        '''
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) # 全连接层1
        x = self.act(x) # 激活函数
        x = self.drop(x) # 丢弃层
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''
        :param dim: 输入的token维度
        :param num_heads: 注意力的头数
        :param mlp_ratio: MLP隐藏层维度与输入维度的比例, 计算hidden_features大小，为输入的四倍
        :param qkv_bias: 是否在qkv线性层中使用偏置
        :param qk_scale: qk缩放因子
        :param drop: 丢弃率，多头自注意力机制最后的linear后使用
        :param attn_drop: 注意力权重的丢弃率，生成qkv后
        :param drop_path: 随机深度丢弃率，用在encoder block中，残差连接
        :param act_layer: 激活函数类型
        :param norm_layer: 归一化层类型
        '''
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) # transformer block中的第一个LayerNorm
         # 多头自注意力机制,实例化
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # 随机深度丢弃，用在残差连接中
        # 如果drop_path大于0，则使用Drop_path，否则不做任何操作
        self.drop_path = Drop_path(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim) # transformer block中的第二个LayerNorm
        mlp_hidden_dim = int(dim * mlp_ratio) # 计算mlp第一个全连接层的节点个数
        # 定义MLP部分。传入dim = mlp_hidden_dim
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 残差连接和注意力机制
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接和MLP
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer =PatchEmbed,
                 norm_layer=None):
        '''
        :param img_size: 输入图像的大小
        :param patch_size: 每个patch的大小
        :param in_chans: 输入图像的通道数
        :param num_classes: 分类任务的类别数
        :param embed_dim: 每个patch嵌入的维度
        :param depth: Transformer编码器块的数量
        :param num_heads: 注意力的头数
        :param mlp_ratio: MLP隐藏层维度与输入维度的比例
        :param qkv_bias: 是否在qkv线性层中使用偏置
        :param qk_scale: qk缩放因子
        :param representation_size: 表示层的大小
        :param distilled: 是否使用蒸馏token
        :param drop_rate: 丢弃率
        :param attn_drop_rate: 注意力权重的丢弃率
        :param drop_path_rate: 随机深度丢弃率
        :param embed_layer: Patch嵌入层的类型
        :param norm_layer: 归一化层类型
        '''
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features用于表示特征维度, embed_dim赋值给self.embed_dim
        self.num_tokens = 2 if distilled else 1 #如果使用蒸馏token，则num_tokens为2，否则为1
         # 归一化层，如果没有提供，则使用LayerNorm

        # 设置一个较小的参数防止除0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #如果没有提供归一化层，则使用LayerNorm
        act_layer = act_layer or nn.GELU()
        # Patch embedding层，将图像划分为patches并嵌入到高维空间
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches #得到patch的数量
        #使用nn.Parameter定义可学习的参数,用零矩阵初始化，第一个为batch维度

        # 分类token，作为Transformer输入序列的第一个token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 位置嵌入，添加位置信息以保留空间结构  pos_embed大小与conccat拼接后的大小一致，197，768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate) #位置嵌入后的丢弃层
        # 创建Transformer编码器块的列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # 使用nn.Sequential将列表中的所有模块打包为一个整体
        self.blocks = nn.ModuleList(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
                for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) # 最后的归一化层

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() #不做任何处理

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None 
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # 初始化参数
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=.02)
        
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)


    def forward_features(self, x):
        # B C H W -> B num_patches embed_dim
        x = self.patch_embed(x)  # 将图像划分为patches并嵌入到高维空间
        # 1,1,768 -> B,1,768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展分类token以匹配批量大小
        
        # 如果dist_token存在，则拼接dist_token和cls_token，否则只拼接cls_token
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # 拼接分类token B 197 768
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)  # 拼接分类token和蒸馏token

        x = self.pos_drop(x + self.pos_embed)  # 添加位置嵌入并进行丢弃
        x = self.blocks(x)  # 通过Transformer编码器块
        x = self.norm(x)  # 最后的归一化层
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  # dist_token不存在，提取cls_token对应的输出
        else:
            return x[:, 0], x[:, 1]  # 返回分类token和蒸馏token的表示
        
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            # 分别通过head和head_dist进行预测
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # 分别通过分类头和蒸馏头
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                # 则返回两个人头部的预测结果
                return x, x_dist
        else:
            x = self.head(x) #  最后的linear全连接层
        return x


def _init_vit_weights(m):
    """ ViT weight initialization
    """
    # 判断模块m是否是nn.Linear的实例
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None: # 如果偏置存在，则初始化为零
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out') # 对卷积层的权重做一个初始化，适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias) # 对LayerNorm的偏置初始化为零
        nn.init.ones_(m.weight) # 对LayerNorm的权重初始化为1

# Vision Transformer模型的实现
def vit_base_patch16_224(num_classes:int=1000, pretrained=False):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, representation_size=None, num_classes=num_classes)
    return model