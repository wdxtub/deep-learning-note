import time
import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch import nn, optim
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('定义 LeNet')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


print('看下每层形状')
net = LeNet()
print(net)

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

print('训练一下')
lr, num_epochs = 0.002, 10
optimizer = optim.Adam(net.parameters(), lr=lr)
utils.train_cnn(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
print('在我的电脑上，CPU 相对 GPU 时间要 x10，供大家参考')
print('计算会复杂一些，所以比之前 MLP 慢不少，准确率也需要大概 10 轮左右，才能达到比较高')

'''
training on cpu
epoch 1, loss 0.0058, train acc 0.441, test acc 0.673, time 12.0 sec
epoch 2, loss 0.0029, train acc 0.721, test acc 0.730, time 13.1 sec
epoch 3, loss 0.0024, train acc 0.755, test acc 0.766, time 13.8 sec
epoch 4, loss 0.0021, train acc 0.783, test acc 0.784, time 14.2 sec
epoch 5, loss 0.0020, train acc 0.801, test acc 0.807, time 14.0 sec
epoch 6, loss 0.0018, train acc 0.819, test acc 0.818, time 13.8 sec
epoch 7, loss 0.0017, train acc 0.831, test acc 0.830, time 13.5 sec
epoch 8, loss 0.0017, train acc 0.840, test acc 0.824, time 13.5 sec
epoch 9, loss 0.0016, train acc 0.849, test acc 0.843, time 13.5 sec
epoch 10, loss 0.0015, train acc 0.856, test acc 0.841, time 13.5 sec
'''