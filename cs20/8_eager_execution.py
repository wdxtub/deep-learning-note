import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tf.compat.v1.enable_eager_execution()

# 可以直接计算
x = [[2.]]
m = tf.matmul(x, x)
print(m)

# 也不用担心 lazy loading
x = tf.random.uniform([2, 2])
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        print(x[i, j])

# 可以像 Numpy Array 一样用
x = tf.constant([1.0, 2.0, 3.0])
assert type(x.numpy()) == np.ndarray
squared = np.square(x)
# Tensors 可以遍历
for i in x:
    print(i)

# 也可以直接进行梯度计算
def square(x):
    return x**2

grad = tfe.gradients_function(square)
print(square(3.))
print(grad(3.))

x = tf.Variable(2.0)
def loss(y):
    return (y - x ** 2) ** 2

grad = tfe.implicit_gradients(loss)
print(loss(7.))
print(grad(7.))
