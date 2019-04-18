import keras
import numpy as np
from keras import layers
import random
import sys

print("学习尼采的写作风格和主题")
print("下载并解析文本文件，保存在 ~/.keras/dataset 中")
path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print("语料长度", len(text))
print("提取长度为 maxlen 的序列，进行 one-hot 编码，打包成为 (sequences, maxlen, unique_characters) 三维数组")
print("准备数组 y，其中包含对应目标，即在每一个序列之后出现的字符（进行 one-hot 编码）")
maxlen = 60 # 提取 60 个字符组成序列
step = 3 # 每 3 个字符采样一个新序列
sentences = [] # 保存所提取的序列
next_chars = [] # 保存目标（即下一个字符）

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i + maxlen])

print("Number of sequences:", len(sentences))
print("添加语料中唯一字符列表")
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
print("构造字符的字典")
char_indices = dict((char, chars.index(char)) for char in chars)

print("Vectorization...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print("构建网络，单层 LSTM")
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
print("One-hot 编码，所以使用 categorical_crossentropy")
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # 给定模型预测，采样下一个字符的函数
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print("文本生成循环")
for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1) # 将模型在数据上拟合一次
    start_index = random.randint(0, len(text) - maxlen - 1) # 随机选择一个文本中字
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"') 
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1. # 对目前生成的字符进行 one-hot 编码
            
            preds = model.predict(sampled, verbose=0)[0]
            # 对下一个字符进行采样
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)