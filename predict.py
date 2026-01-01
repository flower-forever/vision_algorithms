"""
模型预测脚本
支持单张图像预测、批量预测和可视化
可以加载训练好的 ViT 或 CNN 模型
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import warnings

# 配置 matplotlib 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 导入模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Vision Transformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CNN'))

try:
    from vit import VisionTransformer
except ImportError:
    print("警告: 无法导入 vit 模块")
    VisionTransformer = None

try:
    from cnn import simplecnn
except ImportError:
    print("警告: 无法导入 cnn 模块")
    simplecnn = None

# 从 train.py 导入辅助函数
from train import create_vit_for_cifar10


# ==================== 配置区域 ====================
CONFIG = {
    'model_name': 'cnn',  # 可选: 'vit', 'cnn'
    'model_path': './checkpoints/cnn_best.pth',  # 模型权重路径
    'num_classes': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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


# ==================== 加载模型 ====================
def load_model(model_path, model_name, num_classes, device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 创建模型
    if model_name.lower() == 'vit':
        model = create_vit_for_cifar10(num_classes=num_classes)
    elif model_name.lower() == 'cnn':
        model = simplecnn(num_class=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功!")
    if 'val_acc' in checkpoint:
        print(f"✓ 验证准确率: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"✓ 训练轮数: {checkpoint['epoch']}")
    
    return model


# ==================== 图像预处理 ====================
def get_transform():
    """获取图像预处理变换"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 图像大小
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform


# ==================== 单张图像预测 ====================
def predict_single_image(model, image_path, class_names, device, show_plot=True):
    """预测单张图像"""
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"错误: 无法加载图像 {image_path}")
        print(f"详细信息: {e}")
        return None
    
    # 预处理
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # 获取所有类别的概率
    all_probs = probabilities[0].cpu().numpy() * 100
    
    print(f"\n预测结果:")
    print(f"图像路径: {image_path}")
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {confidence_score:.2f}%")
    print(f"\n各类别概率:")
    for i, (name, prob) in enumerate(zip(class_names, all_probs)):
        print(f"  {name}: {prob:.2f}%")
    
    # 可视化
    if show_plot:
        visualize_prediction(image, predicted_class, confidence_score, 
                           class_names, all_probs, image_path)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'probabilities': all_probs
    }


# ==================== 批量预测 ====================
def predict_batch(model, image_dir, class_names, device, max_display=10):
    """批量预测文件夹中的图像"""
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # 获取所有图像文件
    image_dir = Path(image_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    if len(image_files) == 0:
        print(f"错误: 在 {image_dir} 中没有找到图像文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图像")
    print("=" * 70)
    
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_path.name}")
        result = predict_single_image(
            model, str(image_path), class_names, device, show_plot=False
        )
        if result:
            results.append({
                'path': str(image_path),
                'filename': image_path.name,
                **result
            })
    
    # 显示前 N 张图像的预测结果
    if results and max_display > 0:
        visualize_batch_predictions(
            results[:max_display], image_files[:max_display], class_names
        )
    
    return results


# ==================== 可视化单张预测结果 ====================
def visualize_prediction(image, predicted_class, confidence, class_names, 
                        probabilities, image_path):
    """可视化单张图像的预测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 显示原图
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(
        f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%',
        fontsize=14, fontweight='bold', pad=10
    )
    
    # 显示概率条形图
    colors = ['green' if i == np.argmax(probabilities) else 'skyblue' 
              for i in range(len(class_names))]
    bars = axes[1].barh(class_names, probabilities, color=colors)
    axes[1].set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Prediction Probabilities', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlim(0, 100)
    
    # 在条形图上添加数值标签
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        axes[1].text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存预测结果（自动处理重名）
    save_dir = './predictions'
    os.makedirs(save_dir, exist_ok=True)
    save_path = get_unique_filename(save_dir, f'pred_{Path(image_path).stem}', '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 预测结果已保存至: {save_path}")
    
    plt.show()
    plt.close()


# ==================== 可视化批量预测结果 ====================
def visualize_batch_predictions(results, image_paths, class_names):
    """可视化批量预测结果"""
    n_images = len(results)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (result, image_path) in enumerate(zip(results, image_paths)):
        try:
            image = Image.open(image_path).convert('RGB')
            axes[idx].imshow(image)
        except:
            axes[idx].text(0.5, 0.5, '图像加载失败', ha='center', va='center')
        
        axes[idx].axis('off')
        title = f"{result['predicted_class']}\n{result['confidence']:.1f}%"
        color = 'green' if result['confidence'] > 80 else 'orange' if result['confidence'] > 50 else 'red'
        axes[idx].set_title(title, fontsize=11, fontweight='bold', color=color, pad=8)
    
    # 隐藏多余的子图
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Batch Prediction Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存结果（自动处理重名）
    save_dir = './predictions'
    os.makedirs(save_dir, exist_ok=True)
    save_path = get_unique_filename(save_dir, 'batch_predictions', '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 批量预测结果已保存至: {save_path}")
    
    plt.show()
    plt.close()


# ==================== 从测试集随机预测 ====================
def predict_from_test_set(model, class_names, device, num_samples=10):
    """从 CIFAR-10 测试集随机选择图像进行预测"""
    import torchvision
    from torch.utils.data import DataLoader
    
    # 加载测试集（不进行归一化，用于显示）
    test_dataset_raw = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=False, transform=transforms.ToTensor()
    )
    
    # 加载测试集（带归一化，用于预测）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=False, transform=test_transform
    )
    
    # 随机选择样本
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # 创建子图
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    results = []
    
    for idx, sample_idx in enumerate(indices):
        # 获取原始图像（用于显示）
        image_raw, true_label = test_dataset_raw[sample_idx]
        
        # 获取预处理后的图像（用于预测）
        image_norm, _ = test_dataset[sample_idx]
        
        # 预测
        with torch.no_grad():
            input_tensor = image_norm.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        true_class = class_names[true_label]
        confidence_score = confidence.item() * 100
        
        # 判断预测是否正确
        is_correct = (predicted.item() == true_label)
        
        # 显示图像
        image_np = image_raw.permute(1, 2, 0).numpy()
        axes[idx].imshow(image_np)
        axes[idx].axis('off')
        
        title = f"True: {true_class}\nPred: {predicted_class}\n{confidence_score:.1f}%"
        color = 'green' if is_correct else 'red'
        axes[idx].set_title(title, fontsize=10, fontweight='bold', color=color, pad=8)
        
        results.append({
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'correct': is_correct
        })
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    # 计算准确率
    accuracy = sum(r['correct'] for r in results) / len(results) * 100
    
    plt.suptitle(f'Test Set Random Predictions (Accuracy: {accuracy:.1f}%)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存结果（自动处理重名）
    save_dir = './predictions'
    os.makedirs(save_dir, exist_ok=True)
    save_path = get_unique_filename(save_dir, 'test_set_predictions', '.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 测试集预测结果已保存至: {save_path}")
    
    plt.show()
    plt.close()
    
    return results


# ==================== 主函数 ====================
def main():
    """主预测流程"""
    print("=" * 70)
    print(" " * 25 + "模型预测")
    print("=" * 70)
    
    # 设备信息
    device = CONFIG['device']
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # 加载模型
    try:
        model = load_model(
            CONFIG['model_path'],
            CONFIG['model_name'],
            CONFIG['num_classes'],
            device
        )
    except Exception as e:
        print(f"\n错误: 无法加载模型")
        print(f"详细信息: {e}")
        print(f"\n请确保:")
        print(f"1. 模型路径正确: {CONFIG['model_path']}")
        print(f"2. 已经训练并保存了模型")
        return
    
    print("\n" + "=" * 70)
    print("选择预测模式:")
    print("1. 单张图像预测")
    print("2. 批量预测（文件夹）")
    print("3. 测试集随机预测")
    print("=" * 70)
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == '1':
        # 单张图像预测
        image_path = input("请输入图像路径: ").strip()
        if os.path.exists(image_path):
            predict_single_image(
                model, image_path, CONFIG['class_names'], device
            )
        else:
            print(f"错误: 文件不存在 {image_path}")
    
    elif choice == '2':
        # 批量预测
        image_dir = input("请输入图像文件夹路径: ").strip()
        if os.path.exists(image_dir):
            predict_batch(
                model, image_dir, CONFIG['class_names'], device
            )
        else:
            print(f"错误: 文件夹不存在 {image_dir}")
    
    elif choice == '3':
        # 测试集随机预测
        num_samples = input("请输入预测数量 (默认 10): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 10
        predict_from_test_set(
            model, CONFIG['class_names'], device, num_samples
        )
    
    else:
        print("无效的选项!")
    
    print("\n" + "=" * 70)
    print("预测完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()