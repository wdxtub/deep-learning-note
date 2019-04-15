import os
import numpy as np

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
print("绘制温度时间序列")
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()

print("绘制前十天的温度时间序列")
plt.plot(range(1440), temp[:1440])
plt.show()