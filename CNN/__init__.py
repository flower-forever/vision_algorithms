from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet import resnet18_cifar, resnet34_cifar, resnet50_cifar, resnet101_cifar, resnet152_cifar
from .cnn import simplecnn

# 标准 ResNet（适用于 ImageNet 等大图像）
model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

# CIFAR-10 专用 ResNet（适用于 32x32 小图像）
cifar_model_dict = {
    'resnet18': resnet18_cifar,
    'resnet34': resnet34_cifar,
    'resnet50': resnet50_cifar,
    'resnet101': resnet101_cifar,
    'resnet152': resnet152_cifar,
    'cnn': simplecnn,
}

def create_model(model_name, num_classes=10, for_cifar=True):
    """
    创建模型
    Args:
        model_name: 模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'cnn')
        num_classes: 分类数
        for_cifar: 是否使用 CIFAR-10 专用版本（32x32 小图像）
    """
    if for_cifar and model_name in cifar_model_dict:
        if model_name == 'cnn':
            return cifar_model_dict[model_name](num_class=num_classes)
        return cifar_model_dict[model_name](num_classes=num_classes)
    elif model_name in model_dict:
        return model_dict[model_name](num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
