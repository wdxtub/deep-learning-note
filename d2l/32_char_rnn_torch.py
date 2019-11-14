import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('载入数据')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = utils.load_data_jay_lyrics()

print('定义模型，单隐藏层，隐藏单元 256')
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

print('输出形状为（时间步数，批量大小，输入个数）')
print('隐藏状态 h 的形状为（层数，批量大小，隐藏单元个数')
num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)

print('实现完整 RNN，请参考 utils.py 中对应函数')

model = utils.RNNModel(rnn_layer, vocab_size).to(device)
print(utils.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

print('进行训练，并创作歌词（相邻采样）')
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['可爱女人', '龙卷风']
utils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)

'''
❯ python 32_char_rnn_torch.py
载入数据
定义模型，单隐藏层，隐藏单元 256
输出形状为（时间步数，批量大小，输入个数）
隐藏状态 h 的形状为（层数，批量大小，隐藏单元个数
torch.Size([35, 2, 256]) 1 torch.Size([2, 256])
实现完整 RNN，请参考 utils.py 中对应函数
分开活月月月月月月月月月
进行训练，并创作歌词（相邻采样）
epoch 50, perplexity 3.222136, time 0.96 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 都能承受我已无处可躲 我不要再想 我不能再想 我不 我不 我不能再想你 不能再想 我不去 不容我们
epoch 100, perplexity 1.108461, time 0.92 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱
 - 龙卷风 离吹开暴风 在小村外的溪边河口默默等 我对你 依旧 旧的温柔 让我心疼的可爱女人 透明的让我感动的
epoch 150, perplexity 1.043189, time 0.92 sec
 - 可爱女人海漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 离开开了风 在那个人在过了的泪少汗重何当我爱你的是一像无能 安用铅笔的家乡 我很透了空 我想要在这
epoch 200, perplexity 1.020109, time 1.03 sec
 - 可爱女人海漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 都能承受 荣耀的背后刻着一道孤独 仁慈的父我已坠入 看不见罪的国度 请原谅我的自负 没人能说没人可
epoch 250, perplexity 1.021810, time 0.87 sec
 - 可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 都能为喝们看不见 不要我这样打我妈妈 我说你怎么打我妈 你说啊 你怎么打我手 你说啊 是不是你不想
'''