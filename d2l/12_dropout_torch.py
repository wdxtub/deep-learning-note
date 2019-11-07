import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import utils


print('定义模型参数')
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
drop_prob1, drop_prob2 = 0.2, 0.5

net = nn.Sequential(
    utils.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, 10)
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)


print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

num_epochs, lr, batch_size = 5, 100.0, 256 # 这里学习率这么大，是因为我们自己实现的时候没有除以 batchsize
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss = torch.nn.CrossEntropyLoss()
utils.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

'''
获取和读取数据
epoch 1, loss 0.0043, train acc 0.575, test acc 0.679
epoch 2, loss 0.0023, train acc 0.788, test acc 0.799
epoch 3, loss 0.0019, train acc 0.825, test acc 0.817
epoch 4, loss 0.0017, train acc 0.839, test acc 0.819
epoch 5, loss 0.0016, train acc 0.849, test acc 0.837
'''