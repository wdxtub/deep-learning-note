from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models, layers

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("加载路透社数据集")
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print("训练样本个数", len(train_data))
print("测试样本个数", len(test_data))

print("将 train_data 和 test_data 进行 one-hot 编码")
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("将 train_label 和 test_label 进行 one-hot 编码")
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, lable in enumerate(labels):
        results[i, labels] = 1.
    return results
print("keras 有内置的，我们可以直接用")
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print("构建模型")
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

print("编译模型")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("留出验证集")
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val =one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

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
new_model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
new_model.add(layers.Dense(64, activation='relu'))
new_model.add(layers.Dense(46, activation='sigmoid'))

# 注，如果 label 没有用 one_hot 编码，对于整数标签，用 sparse_categorical_crossentropy 即可
new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print("最终结果", results)