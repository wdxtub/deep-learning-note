import torch
import torchtext.vocab as vocab

print('查看已支持的预训练模型')
print(vocab.pretrained_aliases.keys())

print('下载 glove.6B.50d，比较大 800m，会下载全部 glove，但其实我们要的是 100m')
# 下载地址 http://nlp.stanford.edu/data/glove.6B.zip
glove = vocab.GloVe(name='6B', dim=50, cache='./data/glove')
print('词汇数量', len(glove.stoi))
print('测试获取索引与反差词汇')
print(glove.stoi['china'], glove.itos[132])
print('用 knn 来搜索近义词')


def knn(W, x, k):
    # 添加 1e-9 保证数值稳定
    cos = torch.matmul(W, x.view((-1,))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt()
    )
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]


print('用预训练词向量搜索近义词')


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]): # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))


print('测试 chip')
get_similar_tokens('chip', 3, glove)
print('测试 china')
get_similar_tokens('china', 3, glove)

print('求类比词，比如 man vs women 等于 son vs daughter')


def get_analogy(token_a, token_b, token_c, embed):
    vecs = [embed.vectors[embed.stoi[t]] for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    return embed.itos[topk[0]]


print('验证 man woman, son 类比')
print(get_analogy('man', 'woman', 'son', glove))
print('验证 beijing china, tokyo 类比')
print(get_analogy('beijing', 'china', 'tokyo', glove))
print('验证 bad worst, big 类比')
print(get_analogy('bad', 'worst', 'big', glove))
print('验证 do did, go 类比')
print(get_analogy('do', 'did', 'go', glove))
