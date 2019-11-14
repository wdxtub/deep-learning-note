import torch
import random
import zipfile

print('读取 zip 文件')
with zipfile.ZipFile('./data/JayChou/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print('查看前 40 个字符')
print(corpus_chars[:40])

print('替换换行符为空格，方便处理')
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
print('取用部分字符，这个根据自己机器情况而定')
print('总长度', len(corpus_chars))
corpus_chars = corpus_chars[:10000]

print('建立字符索引')
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print('字典大小', vocab_size)

print('将训练集转化为索引')
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)