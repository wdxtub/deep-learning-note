import collections
import math
import random
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import utils


print('处理数据集')
with open('./data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]
print('句子数目', len(raw_dataset))
print('打印前 3 个句子，句尾为 <eos>，生僻词用 <unk> 表示，数字替换成 N')
for st in raw_dataset[:3]:
    print('token count', len(st), st)

print('建立词语索引，只保留至少出现 5 次的词')
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
print('将词映射到索引')
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print('token count', num_tokens)
print('二次采样，越高频的词越可能被丢弃')


def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens
    )


subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('token count', sum([len(st) for st in subsampled_dataset]))
print('比较高频词的采样')


def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]
    ))


print(compare_counts('the'))
print(compare_counts('a'))
print(compare_counts('join'))
print(compare_counts('china'))
print('提取中心词和背景词')


def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2: # 句子太短则不要
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i) # 排除中心词
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


print('用样例数据集展示提取效果')
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

print('生成真实数据的中心词和背景词')
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

print('使用负采样进行近似训练，随机 K 个噪声词')


def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重随机生成 k 个词的索引
                i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

print('读取数据')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)


print('实现小批量读取数据函数')


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels))


print('测试读取')
batch_size = 512
num_workers = 4

dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify,
                            num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break

print('使用嵌入层和小批量乘法实现 skip-gram')
print('嵌入层测试')
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(embed.weight)
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
print(embed(x))
print('小批量乘法测试')
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)

print('skip-gram 前向计算')


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


print('定义模型损失函数 - 二元交叉熵函数')


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)


loss = SigmoidBinaryCrossEntropyLoss()

print('测试 loss 函数')
pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
print(loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1))

print('初始化模型参数')

embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)
)

print('定义训练函数')


def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])

            # 使用掩码变量mask来避免填充项对损失函数计算的影响
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()  # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


print('开始训练')
train(net, 0.01, 10)

print('应用词嵌入模型')


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


print(get_similar_tokens('chip', 3, net[0]))
print(get_similar_tokens('china', 3, net[0]))

'''
开始训练
train on cpu
epoch 1, loss 1.96, time 96.43s
epoch 2, loss 0.62, time 96.94s
epoch 3, loss 0.45, time 96.74s
epoch 4, loss 0.40, time 97.31s
epoch 5, loss 0.37, time 95.05s
epoch 6, loss 0.35, time 94.25s
epoch 7, loss 0.34, time 94.01s
epoch 8, loss 0.33, time 94.24s
epoch 9, loss 0.32, time 95.18s
epoch 10, loss 0.32, time 94.91s
应用词嵌入模型
cosine sim=0.518: intel
cosine sim=0.512: rolled
cosine sim=0.482: bricks
None
cosine sim=0.474: kong
cosine sim=0.428: expressed
cosine sim=0.416: hong
None
'''