import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import utils

print('获取和读取数据')
batch_size = 256
num_worker = 4
mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transforms.ToTensor())
train_iter = Data.DataLoader(mnist_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_worker)
test_iter = Data.DataLoader(mnist_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_worker)

print('定义模型参数')
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

print('定义激活函数')
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

print('定义模型')
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

print('定义损失函数')
loss = torch.nn.CrossEntropyLoss()

print('训练模型，直接借用 softmax 一样的')
num_epochs, lr = 10, 100.0
utils.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)