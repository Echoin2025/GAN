import torchvision as tv
from torch.utils.data import DataLoader

# 检查数据集
dataset = tv.datasets.ImageFolder('data/')
print(f"数据集大小: {len(dataset)}")
print(f"类别数: {len(dataset.classes)}")

# 查看一张图片
img, label = dataset[0]
print(f"图片尺寸: {img.size}")