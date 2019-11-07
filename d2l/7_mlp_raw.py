import torch
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import utils

print('获取和读取数据')
batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

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

'''
训练模型，直接借用 softmax 一样的
epoch 1, loss 0.0030, train acc 0.713, test acc 0.799
epoch 2, loss 0.0019, train acc 0.824, test acc 0.801
epoch 3, loss 0.0017, train acc 0.845, test acc 0.782
epoch 4, loss 0.0015, train acc 0.854, test acc 0.853
epoch 5, loss 0.0014, train acc 0.865, test acc 0.833
epoch 6, loss 0.0014, train acc 0.869, test acc 0.836
epoch 7, loss 0.0013, train acc 0.875, test acc 0.848
epoch 8, loss 0.0013, train acc 0.880, test acc 0.856
epoch 9, loss 0.0012, train acc 0.886, test acc 0.869
epoch 10, loss 0.0012, train acc 0.885, test acc 0.852
'''