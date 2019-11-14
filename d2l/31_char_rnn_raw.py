import time
import math
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import utils
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('载入歌词数据')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = utils.load_data_jay_lyrics()

print('定义 one_hot 函数，方便把数据转化成向量')


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


print('简单测试一下')
x = torch.tensor([0, 2])
print(one_hot(x, vocab_size))
print('批量转换 X 为 one_hot')


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


print('测试一下')
X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)


print('初始化模型参数')
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))

    b_h= torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


print('返回初始化的隐藏状态，返回一个元组，方便处理包含多个 NDArray 的情况')


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


print('定义 RNN 函数，返回输出和隐藏状态')


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # inptus 和 outputs 都是 num_steps 个形状为 (batch_size, vocab_size) 的矩阵
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )


print('简单观察下数据结果的个数（即时间步数）以及形状')
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

print('定义预测函数，基于前缀 prefix 来预测接下来的 num_chars 个字符')


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是 prefix 里的字符或者当前最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


print('简单测试一下')
print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx))
print('结果比较混乱，因为现在模型还没有开始训练')

print('裁剪梯度，避免爆炸')


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


print('定义模型训练函数，评价模型使用困惑度，迭代参数前裁剪梯度，根据不同采样方法变更初始化方法')


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = utils.data_iter_random
    else:
        data_iter_fn = utils.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            utils.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


# 这里用前 20000 个字符制作词典
print('进行训练，并创作歌词（随机采样）')
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['可爱女人', '龙卷风']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device,
                      corpus_indices, idx_to_char, char_to_idx, True, num_epochs, num_steps,
                      lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)

print('进行训练，并创作歌词（相邻采样）')
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device,
                      corpus_indices, idx_to_char, char_to_idx, False, num_epochs, num_steps,
                      lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)


'''
❯ python 31_char_rnn_raw.py
载入歌词数据
定义 one_hot 函数，方便把数据转化成向量
简单测试一下
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.]])
批量转换 X 为 one_hot
测试一下
5 torch.Size([2, 1447])
初始化模型参数
will use cpu
返回初始化的隐藏状态，返回一个元组，方便处理包含多个 NDArray 的情况
定义 RNN 函数，返回输出和隐藏状态
简单观察下数据结果的个数（即时间步数）以及形状
5 torch.Size([2, 1447]) torch.Size([2, 256])
定义预测函数，基于前缀 prefix 来预测接下来的 num_chars 个字符
简单测试一下
分开层剧J膀弱愁符多又门
结果比较混乱，因为现在模型还没有开始训练
裁剪梯度，避免爆炸
定义模型训练函数，评价模型使用困惑度，迭代参数前裁剪梯度，根据不同采样方法变更初始化方法
进行训练，并创作歌词（随机采样）
epoch 50, perplexity 51.194474, time 1.63 sec
 - 可爱女人 透坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 我想拳这样 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我不要再想 我
epoch 100, perplexity 7.627060, time 1.68 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风 一直一直东鸠 我们将打分了离 化身为龙 把山河重新移动 填平裂缝 将东方 的日情调剩下一种 等待英
epoch 150, perplexity 3.371430, time 1.69 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风涌落的伤戏面 经爱盯在什么就一天杨风 你回那里 在小村外的溪边 默默等待 娘子 在箩却的茶 有些事觉
epoch 200, perplexity 2.414131, time 1.64 sec
 - 可爱女人 坏坏的让我心狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风空落的伤不有跳 我却怕爸讯号 干经的没用 到在一元热粥 配上我遇见你是一场悲剧 我可以让生命就这样的
epoch 250, perplexity 2.124895, time 1.68 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 透明的让我感动的可爱女人 坏明的让我感动的可爱女人 坏明的让我感动的可爱
 - 龙卷风空落 铁永的最都有人暴弹哭 石铁路在门父千开始 一天走愁孤迹天一切东想 轻地的爱写笑多的喊 在等待
epoch 300, perplexity 2.042052, time 1.69 sec
 - 可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯动的可爱女人 坏坏的让我疯动的可爱女人 坏坏的让我疯动的可爱
 - 龙卷风空落 为什笔的有迹依然顶晰可说 在回玩不去阻着你的门候 泪笑那明去地安记解的脸言 从说文明只剩下欢解
 
进行训练，并创作歌词（相邻采样）
epoch 50, perplexity 37.366468, time 1.70 sec
 - 可爱女人 爱不的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱
 - 龙卷风  爷有这不是我要多 他不着这想 我不要再想 我不能再想 我不 我想 我不能再想 我不 我不 我不能
epoch 100, perplexity 5.333635, time 1.69 sec
 - 可爱女人 一场我没多受 让我们 半兽人 的灵魂 单纯 停止古婪 永无止尽的战争 让我们 半兽人 的灵魂 单纯
 - 龙卷风 看见无 和你的手式心言暴力 我不到过 你就有回够我 不知不觉 我跟了这节奏 后知后觉 你过好好 你
epoch 150, perplexity 2.658893, time 1.66 sec
 - 可爱女人 一遍汗  你想多陪难始我要要  说难的可以 已在完在还动 你静已引分不睡 化身为人 把始没人新意到
 - 龙卷风 看见这 小你的话的想言  语的让娘时间的战 等待上 我怎么快不好口 手发跳 一颗我 印地安  分期
epoch 200, perplexity 2.159146, time 1.69 sec
 - 可爱女人 因风我没说汉多 我根耍的想模有样 什么兵器最喜欢  什么我梦爱 我的将界来比摧色 也 我 靠你是
 - 龙卷风 看见么这样打 你在这起我妈一定没现 想爱你看去盯着不能 在原的真悟只我的嬉 在等风雨注来临变 泽到
epoch 250, perplexity 2.387836, time 1.65 sec
 - 可爱女人道一点汗乒一 得许时间年只有世满   翻录的最爸栓的的和 这样方果开始临在沼泽 灰狼啃醒不去阻止你
 - 龙卷风 安见我没了难久  你想起有你离找难熬想 走地你不了白 我不要你说到你开 这想就手 荣耀的背后刻着一
epoch 300, perplexity 4.162204, time 1.64 sec
 - 可爱女人 你却我想很难 让我们 半兽人 的灵魂 单滚 对远古恨 永无止尽的过程 让我们 半兽人 的灵魂 单纯
 - 龙卷风 漂怪在战想要世 我什么这慢走堡 为像是你 经不自 别表的公子下机  你在你的之 有一个味利太 小看
'''