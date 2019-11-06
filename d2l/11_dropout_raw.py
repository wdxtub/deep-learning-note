import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import utils


print('定义 dropout')
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob


print('定义模型参数')
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens1)), dtype=torch.float)
b1 = torch.zeros(num_hiddens1, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens1, num_hiddens2)), dtype=torch.float)
b2 = torch.zeros(num_hiddens2, dtype=torch.float)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens2, num_outputs)), dtype=torch.float)
b3 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.requires_grad_(requires_grad=True)

print('定义模型')
drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)
    return torch.matmul(H2, W3) + b3


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

num_epochs, lr, batch_size = 5, 100.0, 256 # 这里学习率这么大，是因为我们自己实现的时候没有除以 batchsize
loss = torch.nn.CrossEntropyLoss()
utils.train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
