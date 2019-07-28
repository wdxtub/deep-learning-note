import numpy as np

print('一维数组')
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)

print('二维数组')
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

print('矩阵乘法')
A1 = np.array([[1,2,3], [4,5,6]])
print(A1.shape)
B1 = np.array([[1,2], [3,4], [5,6]])
print(B1.shape)
print(np.dot(A1, B1))

