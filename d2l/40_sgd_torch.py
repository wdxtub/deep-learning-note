import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.optim as optim
import utils


print('读取数据')
features, labels = utils.get_nasa_data()
print(features.shape)

print('小批量梯度下降')
utils.train_opt_pytorch(optim.SGD, {'lr': 0.05}, features, labels, 10)



