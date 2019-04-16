from keras.models import Model
from keras import layers
from keras import Input

from keras.utils import to_categorical

import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

print("两个输入，一个是问题，一个是用于找答案的文章；一个输出，是问题的答案，一个阅读理解应用")

print("文本输出是一个长度可变的整数序列")
text_input = Input(shape=(None,), dtype='int32', name='text')
print("将输入嵌入长度为 64 的向量")
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
print("利用 LSTM 将向量编码为单个向量")
encoded_text = layers.LSTM(32)(embedded_text)

print("对问题进行相同的处理")
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

print("将编码后的问题和文本连接起来")
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
print("再添加一个 softmax 分类器")
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

print("模型实例化时指定输入和输出")
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

print("将数据输入到多输入模型中")
num_samples = 1000
max_length = 100

print("生成虚构的 Numpy 数据")
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
print("回答是 one_hot 编码")
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = to_categorical(answers, answer_vocabulary_size)

input_has_name = True
if input_has_name:
    model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
else:
    model.fit([text, question], answers, epochs=10, batch_size=128)