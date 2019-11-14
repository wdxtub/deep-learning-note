import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('载入数据')
data_dir = './data/hotdog'
print(os.listdir(data_dir))
train_imgs = ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'test'))

print('展示 16 张图片')
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
utils.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

print('做和预训练模型一样的预处理操作')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

print('定义和初始化模型')
print('预训练模型会默认下载到 ~/.cache/torch/checkpoints 文件夹，如果需要变换，可以修改 $TORCH_MODEL_ZOO 环境变量')
pretrained_net = models.resnet18(pretrained=True)

'''
源码中地址如下，我是用自己的 NAS 下载的
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
'''
print('打印源模型的 fc 层')
print(pretrained_net.fc)
print('原来的是分成 1000 类，所以我们需要修改为 2 类')
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
print('针对不同的层使用不同的学习率')
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([
    {'params': feature_params},
    {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}
], lr=lr, weight_decay=0.001)

print('微调模型')


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=4):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs), batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


train_fine_tuning(pretrained_net, optimizer)

print('定义一个结构一样但是随机化初始化')
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)

print('可以看到微调模型精度更高')


'''
打印源模型的 fc 层
Linear(in_features=512, out_features=1000, bias=True)
原来的是分成 1000 类，所以我们需要修改为 2 类
Linear(in_features=512, out_features=2, bias=True)
针对不同的层使用不同的学习率
微调模型
training on  cpu
epoch 1, loss 4.0836, train acc 0.674, test acc 0.922, time 456.7 sec
epoch 2, loss 0.1991, train acc 0.904, test acc 0.934, time 474.0 sec
epoch 3, loss 0.1026, train acc 0.915, test acc 0.927, time 464.1 sec
epoch 4, loss 0.0606, train acc 0.921, test acc 0.920, time 463.7 sec
定义一个结构一样但是随机化初始化
training on  cpu
epoch 1, loss 2.4249, train acc 0.654, test acc 0.776, time 458.2 sec
epoch 2, loss 0.2222, train acc 0.804, test acc 0.787, time 430.5 sec
epoch 3, loss 0.1286, train acc 0.841, test acc 0.814, time 429.5 sec
epoch 4, loss 0.1015, train acc 0.815, test acc 0.838, time 474.2 sec
可以看到微调模型精度更高
'''