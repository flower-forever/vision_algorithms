"""
通用模型训练脚本
支持 Vision Transformer (ViT) 和 CNN 模型
包含训练过程可视化

# ==================== 注册 MobileNet 模型 ====================
try:
    from mobilenet import mobilenet_v2_cifar  # 你的模型文件
    register_model('mobilenet', mobilenet_v2_cifar, "MobileNet-V2 (CIFAR-10 专用)")
except ImportError:
    print("警告: 无法导入 mobilenet 模块")

# ==================== 新模型接口规范 ====================
def your_model_cifar(num_classes=10):
    ###
    Args:
        num_classes: 分类数量
    Returns:
        nn.Module 实例
    ###
        
    return YourModel(num_classes=num_classes)

如果参数名不同，请使用 lambda 包装以统一接口，例如：
register_model('mymodel', lambda nc: mymodel(num_class=nc), "描述")
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
import warnings

# 配置 matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 导入模型
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Vision Transformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CNN'))


# ==================== 模型注册表 ====================
# 用于统一管理所有模型，新增模型只需在此注册
MODEL_REGISTRY = {}

def register_model(name, model_fn, description=""):
    """
    注册模型到注册表
    Args:
        name: 模型名称（小写）
        model_fn: 模型创建函数，接受 num_classes 参数
        description: 模型描述
    """
    MODEL_REGISTRY[name.lower()] = {
        'fn': model_fn,
        'description': description
    }

def list_models():
    """列出所有可用模型"""
    print("\n可用模型列表:")
    print("-" * 50)
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:15s} - {info['description']}")
    print("-" * 50)


# ==================== 注册 ViT 模型 ====================
try:
    from vit import VisionTransformer, PatchEmbed
    
    def create_vit_for_cifar10(num_classes=10):
        """创建适配 CIFAR-10 的 Vision Transformer"""
        return VisionTransformer(
            img_size=32, patch_size=4, in_chans=3, num_classes=num_classes,
            embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.,
            qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1
        )
    
    register_model('vit', create_vit_for_cifar10, "Vision Transformer (CIFAR-10 适配)")
except ImportError:
    print("警告: 无法导入 vit 模块")


# ==================== 注册 CNN 模型 ====================
try:
    from cnn import simplecnn
    # simplecnn 使用 num_class 参数，包装一下统一接口
    register_model('cnn', lambda nc: simplecnn(num_class=nc), "SimpleCNN 基础卷积网络")
except ImportError:
    print("警告: 无法导入 cnn 模块")


# ==================== 注册 ResNet 模型 ====================
try:
    from resnet import resnet18_cifar, resnet34_cifar, resnet50_cifar, resnet101_cifar, resnet152_cifar
    register_model('resnet18', resnet18_cifar, "ResNet-18 (CIFAR-10 专用)")
    register_model('resnet34', resnet34_cifar, "ResNet-34 (CIFAR-10 专用)")
    register_model('resnet50', resnet50_cifar, "ResNet-50 (CIFAR-10 专用)")
    register_model('resnet101', resnet101_cifar, "ResNet-101 (CIFAR-10 专用)")
    register_model('resnet152', resnet152_cifar, "ResNet-152 (CIFAR-10 专用)")
except ImportError:
    print("警告: 无法导入 resnet 模块")


# ==================== 配置区域 ====================
CONFIG = {
    'model_name': 'resnet18',  # 可选: 'vit', 'cnn', 'resnet18/34/50/101/152' (运行时输入 list_models() 查看全部)
    'num_classes': 10,
    'epochs': 20,
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'plot_dir': './plots',
    'seed': 42
}


# ==================== 工具函数 ====================
def get_unique_filename(directory, base_name, extension):
    """
    生成唯一的文件名，如果文件存在则自动添加编号
    
    Args:
        directory: 文件所在目录
        base_name: 基础文件名（不含扩展名）
        extension: 文件扩展名（如 '.pth', '.png'）
    
    Returns:
        完整的唯一文件路径
    """
    # 确保扩展名以点开头
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # 原始文件路径
    file_path = os.path.join(directory, base_name + extension)
    
    # 如果文件不存在，直接返回
    if not os.path.exists(file_path):
        return file_path
    
    # 文件存在，添加编号
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}{extension}"
        new_path = os.path.join(directory, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


# ==================== 设置随机种子 ====================
def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ==================== 获取模型 ====================
def get_model(model_name, num_classes):
    """
    根据名称从注册表获取模型
    
    Args:
        model_name: 模型名称
        num_classes: 分类数
    Returns:
        模型实例
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"不支持的模型: {model_name}\n可用模型: {available}")
    
    model_info = MODEL_REGISTRY[model_name_lower]
    print(f"使用 {model_info['description']}")
    
    return model_info['fn'](num_classes)


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
    from torch.utils.data import random_split, Subset
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_indices_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 验证集使用测试时的 transform
    val_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True, download=False, transform=test_transform
    )
    val_subset = Subset(val_dataset, val_indices_subset.indices)
    
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
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 绘制准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
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
        for inputs, labels in tqdm(test_loader, desc='Generating Confusion Matrix'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Sample size'},
                annot_kws={'fontsize': 10})
    plt.xlabel('Predicted Class', fontsize=13, fontweight='bold')
    plt.ylabel('True Class', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
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
    best_model_path = None  # 记录最佳模型的保存路径
    
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
        
        # 保存最佳模型（第一次创建文件，之后覆盖）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 只在第一次保存时生成唯一文件名，之后都覆盖同一个文件
            if best_model_path is None:
                best_model_path = get_unique_filename(
                    CONFIG['save_dir'],
                    f"{CONFIG['model_name']}_best",
                    '.pth'
                )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, best_model_path)
            print(f"✓ 保存最佳模型至: {best_model_path} (验证准确率: {val_acc:.2f}%)")
    
    # 训练结束
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"训练完成! 总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print("=" * 70 + "\n")
    
    # 绘制训练曲线（自动处理重名）
    print("绘制训练曲线...")
    plot_path = get_unique_filename(
        CONFIG['plot_dir'],
        f"{CONFIG['model_name']}_training_curves",
        '.png'
    )
    plot_training_curves(history, plot_path)
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试集 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n")
    
    # 绘制混淆矩阵（自动处理重名）
    print("绘制混淆矩阵...")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cm_path = get_unique_filename(
        CONFIG['plot_dir'],
        f"{CONFIG['model_name']}_confusion_matrix",
        '.png'
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
