"""
通用模型训练脚本
支持 Vision Transformer (ViT) 和 CNN 模型
包含训练过程可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 导入数据集加载函数
from download_dataset import get_dataset_info

# 导入模型
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Vision Transformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CNN'))

try:
    from vit import VisionTransformer, PatchEmbed
except ImportError:
    print("警告: 无法导入 vit 模块，请检查路径")
    VisionTransformer = None
    PatchEmbed = None


# ==================== 配置区域 ====================
CONFIG = {
    'model_name': 'vit',  # 可选: 'vit', 'cnn'
    'num_classes': 10,
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'plot_dir': './plots',
    'seed': 42
}


# ==================== 设置随机种子 ====================
def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 简单 CNN 模型定义 ====================
class SimpleCNN(nn.Module):
    """简单的 CNN 模型用于 CIFAR-10"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== ViT 适配 CIFAR-10 ====================
# 直接使用用户编写的 VisionTransformer 类，适配 32x32 的 CIFAR-10 图像
def create_vit_for_cifar10(num_classes=10):
    """创建适配 CIFAR-10 的 Vision Transformer"""
    model = VisionTransformer(
        img_size=32,  # CIFAR-10 图像大小
        patch_size=4,  # 使用 4x4 的 patch，得到 8x8=64 个patches
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,  # 减小嵌入维度以适应小图像
        depth=6,  # 减少层数
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    )
    return model


# ==================== 获取模型 ====================
def get_model(model_name, num_classes):
    """根据名称获取模型"""
    if model_name.lower() == 'vit':
        if VisionTransformer is None:
            raise ImportError("无法导入 VisionTransformer，请检查 vit.py 文件路径")
        print("使用 Vision Transformer 模型 (适配 CIFAR-10)")
        print("基于用户的 vit.py 实现")
        model = create_vit_for_cifar10(num_classes=num_classes)
    elif model_name.lower() == 'cnn':
        print("使用 Simple CNN 模型")
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model


# ==================== 数据加载 ====================
def get_data_loaders(batch_size):
    """获取数据加载器"""
    # 数据增强和归一化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    import torchvision
    train_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=False, transform=train_transform
    )
    
    # 划分训练集和验证集
    from torch.utils.data import random_split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 为验证集设置测试时的 transform
    val_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=False, transform=test_transform
    )
    val_subset.dataset = val_dataset
    
    # 测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=False, transform=test_transform
    )
    
    # 创建数据加载器
    # Windows 下建议使用 num_workers=0 避免多进程问题
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ==================== 训练一个 epoch ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='训练中')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ==================== 验证 ====================
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='验证中'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ==================== 绘制训练曲线 ====================
def plot_training_curves(history, save_path):
    """绘制训练和验证的损失、准确率曲线"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 绘制损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='验证损失', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 绘制准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存至: {save_path}")
    plt.close()


# ==================== 绘制混淆矩阵 ====================
def plot_confusion_matrix(model, test_loader, classes, device, save_path):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='生成混淆矩阵'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': '样本数量'},
                annot_kws={'fontsize': 10})
    plt.xlabel('预测类别', fontsize=13, fontweight='bold')
    plt.ylabel('真实类别', fontsize=13, fontweight='bold')
    plt.title('混淆矩阵', fontsize=15, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存至: {save_path}")
    plt.close()


# ==================== 主训练函数 ====================
def main():
    """主训练流程"""
    print("=" * 70)
    print(" " * 20 + "模型训练开始")
    print("=" * 70)
    
    # 设置随机种子
    set_seed(CONFIG['seed'])
    
    # 创建保存目录
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)
    
    # 设备信息
    device = CONFIG['device']
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # 加载数据
    print("加载数据集...")
    train_loader, val_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
    train_dataset_len = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'Unknown'
    val_dataset_len = len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else 'Unknown'
    test_dataset_len = len(test_loader.dataset) if hasattr(test_loader.dataset, '__len__') else 'Unknown'
    print(f"✓ 训练集: {train_dataset_len} 张图像")
    print(f"✓ 验证集: {val_dataset_len} 张图像")
    print(f"✓ 测试集: {test_dataset_len} 张图像\n")
    
    # 创建模型
    print("创建模型...")
    model = get_model(CONFIG['model_name'], CONFIG['num_classes'])
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}\n")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 开始训练
    print("=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(
                CONFIG['save_dir'], 
                f"{CONFIG['model_name']}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, save_path)
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    # 训练结束
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"训练完成! 总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print("=" * 70 + "\n")
    
    # 绘制训练曲线
    print("绘制训练曲线...")
    plot_path = os.path.join(
        CONFIG['plot_dir'], 
        f"{CONFIG['model_name']}_training_curves.png"
    )
    plot_training_curves(history, plot_path)
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试集 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    cm_path = os.path.join(
        CONFIG['plot_dir'],
        f"{CONFIG['model_name']}_confusion_matrix.png"
    )
    
    try:
        plot_confusion_matrix(model, test_loader, classes, device, cm_path)
    except ImportError:
        print("警告: 需要安装 scikit-learn 和 seaborn 才能绘制混淆矩阵")
        print("运行: pip install scikit-learn seaborn")
    
    print("\n" + "=" * 70)
    print("所有任务完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
