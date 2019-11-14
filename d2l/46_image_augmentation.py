import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import utils
import math
import torchvision
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('打开测试图像')
img = Image.open('data/Image/pika.jpg')
plt.imshow(img)
plt.show()

print('定义辅助函数，多次进行 aug 操作并展示所有结果')


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    utils.show_images(Y, num_rows, num_cols, scale)


print('反转，上下反转不如左右反转通用')
apply(img, torchvision.transforms.RandomHorizontalFlip())
print('裁剪')
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
print('亮度变化')
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
print('色调变化')
apply(img, torchvision.transforms.ColorJitter(hue=0.5))
print('对比度变化')
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
print('全部随机调整')
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)
apply(img, color_aug)
print('叠加多个方法')
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug
])
apply(img, augs)
