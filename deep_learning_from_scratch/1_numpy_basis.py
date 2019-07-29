import numpy as np

print('生成 NumPy 数组')
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

print('NumPy 算术运算')
y = np.array([2.0, 4.0, 6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print('广播')
print(x / 2.0)


print('N 维数组')
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)
B = np.array([[3, 0], [0, 6]])
print(A+B)
print(A*B)
print(A*10) # 也是广播

print('访问元素')
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

print('用 for 访问')
for row in X:
    print(row)

print('转化为 1 维数组')
X = X.flatten()
print(X)
print('获取索引为 0 2 4 的元素')
X[np.array([0, 2, 4])]
print('取出大于 15 的元素')
print(X[X>15])