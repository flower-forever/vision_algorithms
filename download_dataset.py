"""
CIFAR-10 数据集下载脚本
数据集信息：
- 大小：约 170MB
- 类别：10 个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
- 图像尺寸：32x32 RGB
- 训练集：50,000 张图像
- 测试集：10,000 张图像
- 验证集：从训练集中划分 5,000 张
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

def download_cifar10(root_dir='./dataset'):
    """
    下载 CIFAR-10 数据集并组织为 train/val/test 结构
    
    Args:
        root_dir: 数据集保存的根目录
    """
    print("=" * 60)
    print("开始下载 CIFAR-10 数据集...")
    print("=" * 60)
    
    # 创建数据集目录
    os.makedirs(root_dir, exist_ok=True)
    
    # 定义数据转换（基础转换，用于下载）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载训练集
    print("\n[1/2] 下载训练集...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=True,
        transform=transform
    )
    print(f"✓ 训练集下载完成！共 {len(train_dataset)} 张图像")
    
    # 下载测试集
    print("\n[2/2] 下载测试集...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=True,
        transform=transform
    )
    print(f"✓ 测试集下载完成！共 {len(test_dataset)} 张图像")
    
    # 从训练集中划分验证集（10%）
    print("\n从训练集中划分验证集...")
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子以保证可复现性
    )
    
    print(f"✓ 数据集划分完成！")
    print(f"  - 训练集: {len(train_subset)} 张图像")
    print(f"  - 验证集: {len(val_subset)} 张图像")
    print(f"  - 测试集: {len(test_dataset)} 张图像")
    
    # 显示类别信息
    classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
    print(f"\n类别信息：")
    for i, class_name in enumerate(classes):
        print(f"  {i}: {class_name}")
    
    print("\n" + "=" * 60)
    print(f"数据集已保存到: {os.path.abspath(root_dir)}")
    print("=" * 60)
    
    # 创建示例 DataLoader
    print("\n创建数据加载器示例...")
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"✓ DataLoader 创建完成！")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 验证批次数: {len(val_loader)}")
    print(f"  - 测试批次数: {len(test_loader)}")
    
    return train_subset, val_subset, test_dataset, classes


def get_dataset_info(root_dir='./dataset'):
    """
    获取已下载数据集的信息
    """
    if not os.path.exists(os.path.join(root_dir, 'cifar-10-batches-py')):
        print("数据集未找到，请先运行下载脚本！")
        return None
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=root_dir, train=True, download=False, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root_dir, train=False, download=False, transform=transform
    )
    
    # 划分验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_subset, val_subset, test_dataset


if __name__ == '__main__':
    # 下载数据集到 dataset 文件夹
    train_data, val_data, test_data, classes = download_cifar10(root_dir='./dataset')
    
    print("\n" + "=" * 60)
    print("数据集下载完成！可以开始训练模型了。")
    print("=" * 60)
    
    # 使用示例
    print("\n使用示例代码：")
    print("-" * 60)
    print("""
from download_dataset import get_dataset_info
from torch.utils.data import DataLoader

# 加载数据集
train_data, val_data, test_data = get_dataset_info('./dataset')

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 开始训练...
    """)
    print("-" * 60)
