import numpy as np
import string

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

print("单词级别")
print("构建标记索引，为每个单词指定唯一索引，从 1 开始")
word_token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in word_token_index:
            word_token_index[word] = len(word_token_index) + 1

print("对样本进行分词，只考虑前 10 个单词")
word_max_length = 10
word_results = np.zeros(shape=(len(samples), word_max_length, max(word_token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:word_max_length]:
        index = word_token_index.get(word)
        word_results[i, j, index] = 1
    print("sample %d:" % i, word_results[i, :, :])

print("===========")
print("字符级别")
characters = string.printable
char_token_index = dict(zip(range(1, len(characters) + 1), characters))
char_max_length = 50
char_results = np.zeros((len(samples), char_max_length, max(char_token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = char_token_index.get(character)
        char_results[i, j, index] = 1
    print("sample %d:" % i, char_results[i, :, :])
print("===========")
print("Keras 实现单词级别的 one-hot 编码")
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_result = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))