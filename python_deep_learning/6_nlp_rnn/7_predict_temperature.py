import os
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data_dir = 'data/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
print("这个数据集是每十分钟记录 14 个不同的量，来自德国 jena 气象站的记录")

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))
print("解析数据")
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

print("数据标准化")
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

print("生成时间序列样本及其目标的生成器")
print('''
data - 标准化后的原始数据
lookback - 输入数据应该包括多少个时间步
delay - 目标在多少个时间步之后
step - 数据采样的周期，6 = 6 个十分钟即一小时一次
''')
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size
            )
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

print("准备训练生成器、验证生成器和测试生成器")
lookback = 1440
step = 6
delay = 144 # 24 小时后
batch_size = 128

train_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=200000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300000,
                    max_index=None,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size
print("验证集 step 数目", val_steps)
print("测试集 step 数目", test_steps)

print("我们使用平均绝对误差 MAE 来评估")
print("第一个方法-常识假设：总是假设 24 小时之后的温度等于现在的温度")
print("计算常识假设的 MAE")
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    final_mae = np.mean(batch_maes)
    print(final_mae)
    return final_mae
mae = evaluate_naive_method()
print("将误差转换成即使温度差，即 mae x 数据的标准差")
celsius_mae = mae * std[1]
print("预测的温度和实际温度平均相差", celsius_mae, "摄氏度")

print("第二个方法-基本的机器学习，两层 Dense")
naive_ml_model = Sequential()
naive_ml_model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
naive_ml_model.add(layers.Dense(32, activation='relu'))
naive_ml_model.add(layers.Dense(1)) # 不用激活函数
naive_ml_model.compile(optimizer=RMSprop(), loss='mae')
naive_ml_model_history = naive_ml_model.fit_generator(train_gen, 
                                steps_per_epoch=500, 
                                epochs=20, 
                                validation_data=val_gen,
                                validation_steps=val_steps)
print("绘制结果，这个结果只是跟第一个方法接近，并且很容易过拟合")
loss = naive_ml_model_history.history['loss']
val_loss = naive_ml_model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Dense Model Training and validation loss')
plt.legend()
plt.show()

print("第三个方法-GRU 模型，添加 dropout，所以要多训练一些轮数")
gru_model = Sequential()
gru_model.add(layers.GRU(32, 
            dropout=0.2,
            recurrent_dropout=0.2,
            input_shape=(None, float_data.shape[-1])))
gru_model.add(layers.Dense(1))

gru_model.compile(optimizer=RMSprop(), loss='mae')
gru_model_history = gru_model.fit_generator(train_gen, 
                                steps_per_epoch=500, 
                                epochs=40, 
                                validation_data=val_gen,
                                validation_steps=val_steps)
print("绘制结果")
loss = gru_model_history.history['loss']
val_loss = gru_model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('GRU Model Training and validation loss')
plt.legend()
plt.show()

print("第四个方法-堆叠 GRU 模型")
stack_gru_model = Sequential()
stack_gru_model.add(layers.GRU(32, 
            dropout=0.1,
            recurrent_dropout=0.5,
            return_sequences=True,
            input_shape=(None, float_data.shape[-1])))
stack_gru_model.add(layers.GRU(64, activation='relu',
            dropout=0.1,
            recurrent_dropout=0.5 ))
stack_gru_model.add(layers.Dense(1))

stack_gru_model.compile(optimizer=RMSprop(), loss='mae')
stack_gru_model_history = stack_gru_model.fit_generator(train_gen, 
                                steps_per_epoch=500, 
                                epochs=40, 
                                validation_data=val_gen,
                                validation_steps=val_steps)
print("绘制结果")
loss = stack_gru_model_history.history['loss']
val_loss = stack_gru_model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Stack GRU Model Training and validation loss')
plt.legend()
plt.show()

print("第五个方法-双向 GRU 模型")
bi_gru_model = Sequential()
bi_gru_model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
bi_gru_model.add(layers.Dense(1))

bi_gru_model.compile(optimizer=RMSprop(), loss='mae')
bi_gru_model_history = bi_gru_model.fit_generator(train_gen, 
                                steps_per_epoch=500, 
                                epochs=40, 
                                validation_data=val_gen,
                                validation_steps=val_steps)
print("绘制结果")
loss = bi_gru_model_history.history['loss']
val_loss = bi_gru_model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Bidirectional GRU Model Training and validation loss')
plt.legend()
plt.show()

print("第六个方法-CNN+GRU")
cnn_gru_model = Sequential()
cnn_gru_model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
cnn_gru_model.add(layers.MaxPooling1D(3))
cnn_gru_model.add(layers.Conv1D(32, 5, activation='relu'))
cnn_gru_model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
cnn_gru_model.add(layers.Dense(1))

print(cnn_gru_model.summary())

cnn_gru_model.compile(optimizer=RMSprop(), loss='mae')
cnn_gru_model_history = cnn_gru_model.fit_generator(train_gen, 
                                steps_per_epoch=500, 
                                epochs=20, 
                                validation_data=val_gen,
                                validation_steps=val_steps)
print("绘制结果")
loss = cnn_gru_model_history.history['loss']
val_loss = cnn_gru_model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Bidirectional GRU Model Training and validation loss')
plt.legend()
plt.show()