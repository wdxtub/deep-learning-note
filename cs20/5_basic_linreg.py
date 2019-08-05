import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import numpy as np

DATA_FILE = 'sample/birth_life_2010.txt'

# Step 1: 从 txt 文件中读取数据
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: 为 X（出生率） 和 Y（预期寿命） 创建占位符
# 注意 X 和 Y 都是 float 类型的标量
X = tf.compat.v1.placeholder(tf.float32, name='X')
Y = tf.compat.v1.placeholder(tf.float32, name='Y')

# Step 3: 创建 weight 和 bias 并初始化为 0.0
# 我们要使用 tf.get_variable 来创建
w = tf.compat.v1.get_variable('weights', initializer=tf.constant(0.0))
b = tf.compat.v1.get_variable('bias', initializer=tf.constant(0.0))

w1 = tf.compat.v1.get_variable('weights_1', initializer=tf.constant(0.0))
u1 = tf.compat.v1.get_variable('weights_1_2', initializer=tf.constant(0.0))
b1 = tf.compat.v1.get_variable('bias_1', initializer=tf.constant(0.0))

w2 = tf.compat.v1.get_variable('weights_2', initializer=tf.constant(0.0))
b2 = tf.compat.v1.get_variable('bias_2', initializer=tf.constant(0.0))

# Step 4: 构造用来预测 Y 的模型
# 把前面的 w, X, b 都用上
Y_predicted = w * X + b
Y_predicted_1 = w1 * X * X + u1 * X + b1
Y_predicted_2 = w2 * X + b2

# Step 5: 损失函数用 MSE，也可以使用 Huber loss
loss = tf.square(Y - Y_predicted, name='loss')
loss_1 = tf.square(Y - Y_predicted_1, name='loss_1')
loss_2 = utils.huber_loss(Y, Y_predicted_2)

# Step 6: 使用梯度下降方法来最小化 loss，学习率是 0.001
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
optimizer_1 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_1)
optimizer_2 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_2)

start = time.time()

# 写入 graph
writer = tf.summary.FileWriter('data/graphs/linear_reg', tf.compat.v1.get_default_graph())

with tf.compat.v1.Session() as sess:
    # Step 7: 初始化所有的变量
    sess.run(tf.compat.v1.global_variables_initializer())

    # Step 8: 进行 100 个 epoch 的模型训练，一个 epoch 我们会把所有的数据计算都计算一次
    for i in range(100):
        total_loss = 0
        total_loss_1 = 0
        total_loss_2 = 0
        for x, y in data:
            # 执行训练的动作，并且获取 loss 的值
            # 这里我们要把训练数据喂给 optimizer
            _, _, _, _loss, _loss_1, _loss_2 = sess.run([optimizer, optimizer_1, optimizer_2, loss, loss_1, loss_2], feed_dict={X: x, Y: y})
            total_loss += _loss
            total_loss_1 += _loss_1
            total_loss_2 += _loss_2

        print('1st Loss Epoch {0}: {1}'.format(i, total_loss / n_samples))
        print('2nd Loss Epoch {0}: {1}'.format(i, total_loss_1 / n_samples))
        print('3rd Loss Epoch {0}: {1}'.format(i, total_loss_2 / n_samples))

    # 用完之后关闭 writer
    writer.close()

    # Step 9: 输出最终得到的 w 和 b 的值
    w_out, b_out = sess.run([w, b])
    w_1_out, u_1_out, b_1_out = sess.run([w1, u1, b1])
    w_2_out, b_2_out = sess.run([w2, b2])


print('Took: %f seconds' % (time.time() - start))
print(f'W {w_out}; b {b_out}')
print(f'W1 {w_1_out}; U1 {u_1_out}; b1 {b_1_out}')
print(f'W1 {w_2_out}; b2 {b_2_out}')

plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='linear reg + MSE')
plt.plot(data[:, 0], data[:, 0] * w_2_out + b_2_out, 'y', label='linear reg + Huber')
# 排序，方便显示
show_x = np.sort(data[:, 0], axis=0)
plt.plot(show_x, w_1_out * show_x ** 2 + show_x * u_1_out + b_1_out, 'g', label='L2 reg + MSE')
plt.legend()
plt.show()
