from keras.datasets import boston_housing
from keras import models, layers
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("获取数据集")
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print("训练数据大小", train_data.shape)
print("测试数据大小", test_data.shape)
print("训练目标是房屋价格中位数，单位是千美元")
print(train_targets)

print("标准化数据，特征的平均值为 0，标准差为 1")
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
print("在工作流程中，不能使用在测试数据上计算得到的任何结果，标准化也不行")
test_data -= mean
test_data /= std

print("构建网络，较小的网络可以降低过拟合")
print("MSE - 均方误差，回归问题常用")
print("MAE - 平均绝对误差，比如 MAE 0.5 在这里意味着平均价格相差 500 美元")
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

print("在小样本上利用 K Fold 验证")
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print("处理 Fold #", i)
    val_start = i * num_val_samples
    val_end = val_start + num_val_samples
    # 验证数据
    val_data = train_data[val_start: val_end]
    val_targets = train_targets[val_start: val_end]
    # 训练数据
    partial_train_data = np.concatenate(
        [train_data[:val_start],
        train_data[val_end:]],axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:val_start],
        train_targets[val_end:]],axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, 
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

print("计算所有轮次中 K 折验证分数平均值")
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print("绘制验证分数")
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

print("重新绘制以看清规律。删除前十个点；将每个数据点替换为前面数据点的指数移动平均值")
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE after Smoothing')
plt.show()
print("从图中可得 MAE 在 80 轮后不再降低，之后开始过拟合")

print("重新训练")
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("Test MSE", test_mse_score)
print("Test MAE", test_mae_score)