import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集下载地址 http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

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

print('看一批数据')
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('batch count', len(train_iter))

print('构建模型')

class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 初始时间步和最终时间步的隐藏状态作为全连接层输入

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

print('加载预训练词向量')
glove = Vocab.GloVe(name='6B', dim=100, cache='./data/glove')


def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 0
    if oov_count > 0:
        print("There are %d oov words.")
    return embed


net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

print('训练并评价模型，只训练一轮')

lr, num_epochs = 0.01, 1
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

print('尝试预测')
print(utils.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
print(utils.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))

'''
training on  cpu
epoch 1, loss 0.6503, train acc 0.593, test acc 0.790, time 1577.5 sec
尝试预测
positive
negative
'''
