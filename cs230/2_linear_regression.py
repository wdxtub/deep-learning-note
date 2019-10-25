import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ui

print("PART 1 单变量线性回归")
ui.split_line1()

path = 'data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print("查看前 5 行数据")
print(data.head())
ui.split_line2()
print("查看数据集统计信息")
print(data.describe())
ui.split_line2()
print("绘制数据分布")
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()
ui.split_line2()
print("设置代价函数")

def computeCost(X, y, theta):
    inner = np.power((X * theta.T - y), 2)
    return np.sum(inner) / (2 * len(X))

ui.split_line2()
print("在训练集中添加一列，便于我们以向量化的方法计算 Cost 和梯度")
data.insert(0, 'Ones', 1)
print("进行变量初始化")
cols = data.shape[1]
print("X 是所有的行，去掉最后一列")
X = data.iloc[:, 0:cols-1]
print("X 的前五行")
print(X.head())
print("y 是最后一列")
y = data.iloc[:, cols-1:cols]
print("y 的前五行")
print(y.head())
ui.split_line2()
print("我们的代价函数需要传入 numpy 矩阵，先进行转换，并初始化 theta")
X = np.matrix(X.values)
y = np.matrix(y.values)
print("theta 是一个 (1,2) 矩阵")
theta = np.matrix(np.array([0, 0]))
print("X, y, theta 的维度")
print(X.shape, theta.shape, y.shape)
print("计算初始状态的 Cost")
print(computeCost(X, y, theta))
ui.split_line2()
print("批量梯度下降函数")

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

print("初始化学习率与迭代次数")
alpha = 0.01
iters = 1000
print("运行梯度下降算法")
g, cost = gradientDescent(X, y, theta, alpha, iters)
print("最后得到的 theta 为", g)
print("再次计算代价函数")
print(computeCost(X, y, g))
ui.split_line2()
print("绘制线性模型及数据，查看拟合情况")
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
print("绘制 Cost 下降的趋势")
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

ui.split_line1()
print("PART 2 多变量线性回归")
ui.split_line1()

path = 'data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print("查看数据前 5 行")
print(data2.head())
print("进行特征归一化，查看归一化后的数据")
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())
print("对新数据应用前面一样的套路")
data2.insert(0, 'Ones', 1)
# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)
print("查看训练过程")
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

ui.split_line1()
print("PART 3 用 Scikit Learn 回归一次")
ui.split_line1()
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)
print("scikit-learn 预测表现")
x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

ui.split_line1()
print("PART 4 直接解方程得到最佳 theta")
ui.split_line1()
print("注意：如果特征数量较大， normal equation 因为计算复杂度较高，所以无法使用")

def normalEqn(X, y):
    #X.T@X等价于X.T.dot(X)
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

final_theta2 = normalEqn(X, y)
print("直接计算得到的 theta", final_theta2)
print("第一步中梯度下降得到的 theta 为", g)
print("可以看到还是有一定差别的")