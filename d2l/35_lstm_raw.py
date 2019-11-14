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

    W_xi, W_hi, b_i = _three() # 输入门
    W_xf, W_hf, b_f = _three() # 遗忘门
    W_xo, W_ho, b_o = _three() # 输出门
    W_xc, W_hc, b_c = _three() # 候选记忆细胞

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


print('定义模型')


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)


print('训练模型并创作歌词')
num_epochs, num_steps, batch_size, lr, clipping_theta = 200, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['可爱女人', '龙卷风']
utils.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                            vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                            False, num_epochs, num_steps, lr, clipping_theta, batch_size,
                            pred_period, pred_len, prefixes)

'''
训练模型并创作歌词
epoch 40, perplexity 174.415288, time 3.61 sec
 - 可爱女人 我想想你想你 我不要你不爱 我不要你不爱你 我不要你不爱你 我不要你不爱你 我不要你不爱你 我不要
 - 龙卷风 我想想你想你 我不要你不爱 我不要你不爱你 我不要你不爱你 我不要你不爱你 我不要你不爱你 我不要
epoch 80, perplexity 27.808907, time 3.42 sec
 - 可爱女人 我想我这想 你一定手不要 你静看着我已多 我说 你不好 我想了这样样 你后后觉 我给了一个  后后
 - 龙卷风 我想你 你兽我 我想多这样牵着你的手  想说我也想很多 没有你要你 你说你说我 你不着这样  后后
epoch 120, perplexity 5.401331, time 3.55 sec
 - 可爱女人 我不想再想 一定的是不要 我会会再淋一遍 为想到好做的睡 我说你好不难你 我想该不不到 我永 我不
 - 龙卷风 我想想 你想我的手是我怎错错 泪开开象不要 你 我不再你不开 我不么 不想我的想爱我怎么错错 我才
epoch 160, perplexity 2.167289, time 3.53 sec
 - 可爱女人 我想想的想笑在你 想要你说 最对没有我就要 你说啊对医药  说你是其远我的愿望就怎么小 我怎么每天
 - 龙卷风 我想一起来祷告 仁慈的父我已坠入 看不见罪的国度 请原谅我的自负 没人能说没人可说 好难承受 荣耀
epoch 200, perplexity 1.375516, time 3.51 sec
 - 可爱女人 趁着你自经我有要 坐着一直的黑色 载会缓缓好离过  这个这生 我的世界 在小常在在暗空 练待动动
 - 龙卷风 为什么看到绕来 有天有安南嵩山 学少林河南嵩山  把有来跟 当山一步 包慢够够  一暗螂空 在炭中
'''