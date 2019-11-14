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

print('载入训练数据')
num_workers = 4
all_images = torchvision.datasets.CIFAR10(train=True, root='./data/CIFAR-10', download=True)
# 每个元素都是 (image, label)
utils.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

print('定义 aug 操作')
flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def load_cifar10(is_train, augs, batch_size, root='data/CIFAR-10'):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=False)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


print('训练模型')


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, utils.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=1)


print('只训练一轮')
train_with_data_aug(flip_aug, no_aug)

'''
只训练一轮
training on  cpu
epoch 1, loss 1.4007, train acc 0.494, test acc 0.478, time 372.4 sec
'''