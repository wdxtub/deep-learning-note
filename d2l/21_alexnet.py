import time
import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch import nn, optim
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义 AlexNet')


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kerner_size, stride
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，padding 为 2 保证输入输出尺寸一致，增大输出通道
            nn.Conv2d(96, 256, 5, 1, 2), # in_channels, out_channels, kerner_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续 3 个卷积层，更小的窗口，继续增加通道数
            # 前两个卷积层后不用池化层，减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 使用 Dropout 缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 我们用 Fashion-MNIST，所以最后为 10
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


print('查看网络结构')
net = AlexNet()
print(net)

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)


print('开始训练，这里我只训练 2 轮')
lr, num_epochs = 0.001, 2
optimizer = optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

'''
开始训练，这里我只训练 2 轮
training on cpu
epoch 1, loss 0.0028, train acc 0.718, test acc 0.830, time 4233.4 sec
epoch 2, loss 0.0014, train acc 0.871, test acc 0.884, time 4219.5 sec
'''