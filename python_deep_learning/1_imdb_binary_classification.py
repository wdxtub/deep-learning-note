from keras.datasets import imdb
import numpy as np
from keras import models, layers, optimizers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 加载数据，只保留前 10000 个最常出现的单次
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("train_data[0]")
print(train_data[0])
print("train_labels[0]")
print(train_labels[0])
print("单词索引不超过 10000")
print(max([max(sequence) for sequence in train_data]))
print("解码 train_data[0]")
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 减去 3 是因为 0 1 2 是保留索引，分别是 padding, start of sequence, unknown
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_review)

print("将 train_data 和 test_data 进行 one-hot 编码")
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("one-hot 编码后的 train_data[0]")
print(x_train[0])

print("将 train_labels 和 test_labels 向量化")
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print("使用 Dense 构建网络，在二分类上效果很好")
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print("编译模型")
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("留出验证集")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("训练模型")
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

print("绘制训练损失和验证损失")
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("绘制训练精度和验证精度")
plt.clf() # 清空图像
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("注意这个模型是过拟合的，因为验证集的 loss 不断在上升")

print("重新训练一个模型")
new_model = models.Sequential()
new_model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
new_model.add(layers.Dense(16, activation='relu'))
new_model.add(layers.Dense(1, activation='sigmoid'))

new_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
new_model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print("最终结果", results)