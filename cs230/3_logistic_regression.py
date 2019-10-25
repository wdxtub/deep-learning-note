import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ui

print("构建一个逻辑回归模型来预测，某个学生是否被大学录取。设想你是大学相关部分")
print("的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。现在你拥有")
print("之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有")
print("他们两次测试的评分和最后是被录取的结果。为了完成这个预测任务，我们准备构")
print("建一个可以基于两次测试评分来评估录取可能性的分类模型。")
ui.split_line1()
print("载入数据并查看分布")
path = 'data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print("查看数据前五行")
print(data.head())
print("创建两个分数的散点图，并使用颜色编码来可视化")
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
ui.split_line2()
print("sigmoid 函数")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print("输出函数图像")
nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()
ui.split_line2()
print("定义代价函数")

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

ui.split_line2()
print("设置 X, theta, y")
# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
print("检查 X, theta, y 的维度")
print(X.shape, theta.shape, y.shape)
print("计算优化前的代价函数")
print(cost(theta, X, y))
ui.split_line1()
print("定义梯度下降函数，注意，这里指计算出梯度步长，实际由 SciPy 的 optimize 函数计算 Cost 和梯度")

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

print("第一次执行梯度下降的结果")
print(gradient(theta, X, y))
print("用SciPy's truncated newton（TNC）实现寻找最优参数")
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print("结果为", result)
print("重新计算 Cost")
print(cost(theta, X, y))
ui.split_line2()
print("编写预测函数")

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

print("计算当前 theta 在训练集上的准确率")
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

ui.split_line1()
print("设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。对于这两次测试，你想")
print("决定是否芯片要被接受或抛弃。为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，")
print("从其中你可以构建一个逻辑回归模型")
ui.split_line2()
print("载入数据并可视化")
path = 'data/ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print("查看数据前 5 行")
print(data2.head())
print("绘制数据分布")
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
print("构造一组多项式特征")
degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
print("查看新数据的前 5 行")
print(data2.head())
ui.split_line2()
print("定义带正则化项的 Cost 函数")

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

print("定义梯度下降函数")

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad

print("准备好训练需要用的数据")
# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
print("设置初始学习率")
learningRate = 1
print("优化前的 Cost 为")
print(costReg(theta2, X2, y2, learningRate))
print("第一步的梯度步长为")
print(gradientReg(theta2, X2, y2, learningRate))
print("计算优化后的结果")
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
print(result2)
print("预测准确率")
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

ui.split_line1()
print("用 Scikit-Learn 进行优化")
from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
print("最终的准确率为", model.score(X2, y2))
print("可以看到这个准确率比我们之前优化的低不少")