import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import utils
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集下载地址 http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

print('定义一维卷积层')


def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


print('测试一下')
X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
print(corr1d(X, K))

print('定义时序最大池化层')


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)


print('获取数据集')
train_data, test_data = utils.read_imdb('train'), utils.read_imdb('test')
print('获取词表')
vocab = utils.get_vocab_imdb(train_data)
print('vocab count', len(vocab))

print('创建数据迭代器')
batch_size = 64
train_set = Data.TensorDataset(*utils.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*utils.preprocess_imdb(train_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

print('定义 TextCNN 模型')


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = 2*embed_size,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)), dim=2) # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


print('构建网络')
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)

print('加载预训练词向量')
glove = Vocab.GloVe(name='6B', dim=100, cache='./data/glove')
net.embedding.weight.data.copy_(
    utils.load_pretrained_embedding(vocab.itos, glove))
net.constant_embedding.weight.data.copy_(
    utils.load_pretrained_embedding(vocab.itos, glove))
net.constant_embedding.weight.requires_grad = False

print('训练并评价模型')
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

print('尝试预测')
print(utils.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
print(utils.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))

'''
training on  cpu
epoch 1, loss 0.4834, train acc 0.759, test acc 0.874, time 427.7 sec
epoch 2, loss 0.1655, train acc 0.860, test acc 0.937, time 424.7 sec
epoch 3, loss 0.0700, train acc 0.917, test acc 0.971, time 416.7 sec
epoch 4, loss 0.0301, train acc 0.957, test acc 0.991, time 427.4 sec
epoch 5, loss 0.0129, train acc 0.979, test acc 0.997, time 460.8 sec
尝试预测
positive
negative
'''