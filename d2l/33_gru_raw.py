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

print('初始化模型参数')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xz, W_hz, b_z = _three() # 更新门参数
    W_xr, W_hr, b_r = _three() # 重置门参数
    W_xh, W_hh, b_h = _three() # 候选隐藏层参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


print('隐藏初始化状态，和前面相同')


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


print('定义 GRU 模型')


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1-Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )


print('训练并创作歌词')
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['可爱女人', '龙卷风']
utils.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                            vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                            False, num_epochs, num_steps, lr, clipping_theta, batch_size,
                            pred_period, pred_len, prefixes)

'''
训练并创作歌词
epoch 40, perplexity 101.107354, time 2.45 sec
 - 可爱女人 我想要你的爱写 像在我的爱你 让我们 你不是 快兽人 的灵魂 有一种 三颗 一直两 一颗 三颗 一
 - 龙卷风 我想要你的爱人 让我们 你不是 一颗两 三颗 一颗 三颗 一颗 三颗 三颗 有一种 三颗 一直两
epoch 80, perplexity 7.574267, time 2.49 sec
 - 可爱女人 沉许在美索不达米亚平原 爷爷泡的茶 有一种味道叫做家 他羽泡的茶 听说名和利 不拿跳动 全天用双截
 - 龙卷风 我会怕着二碎人人就能眼泪 别手不能够永远不及   难道这不是我要的天堂景象 沉沦假象 你只会感到更
epoch 120, perplexity 1.670065, time 2.59 sec
 - 可爱女人 沉亮的欢乐琴上的红色油漆 反射出儿时天真的嬉戏模样  被期待 被覆盖 蜕变的公式我学不来  难道这
 - 龙卷风 我想要你想远想一点汗融戏 最后再一个人慢慢的回忆 没有了过去 我将往事抽离 如果我遇见你是一场悲剧
epoch 160, perplexity 1.094771, time 2.59 sec
 - 可爱女人 沉不容易不要再让 没有你梦的有 一本又抱半岛铁 爷爷泡的茶 像幅泼墨的山水画 唐朝千年的风沙 现在
 - 龙卷风 我想一起来祷告 仁慈的父我已坠入 看不见罪的国度 请原谅我的自负 仁慈的父我已坠入 看不见罪的国度
epoch 200, perplexity 1.045868, time 2.63 sec
 - 可爱女人 沉不可以不要这个奖 不想问 我只想要留一点汗 我当我自己的裁判 不想说 选择对手跟要打的仗 全体师
 - 龙卷风 我向一起来祷告 仁慈的父我已坠入 看不见罪的国度 请原谅我的自负 仁慈的父我已坠入 看不见罪的国度
epoch 240, perplexity 1.034094, time 2.67 sec
 - 可爱女人 沉以心翼翼的豆瓣酱 我用 神心打的完言 然想要一直一点往爬 感后觉奏满许他开 在海了你心里透河般的
 - 龙卷风 我们一起来祷告 仁慈的父我已坠入 看不见罪的国度 请原谅我的自负 仁慈的父我已坠入 看不见罪的国度
epoch 280, perplexity 1.030017, time 2.68 sec
 - 可爱女人 没有你不到  那是我们该难着简  为什么 我不要再想 我不 我不 我不要再想你 爱情来的太快就像龙
 - 龙卷风 你怎么河我看到 你说我 难怪我妈 你说啊久了傲 静静的脸动作 帅呆了千泣相能买亮的在延 周阳文明会
'''