import numpy as np
import tensorflow as tf

import matplotlib as mpl
from matplotlib import pyplot as plt

HIDDEN_SIZE = 30 # LSTM 隐藏节点个数
NUM_LAYERS = 2

TIMESTEPS = 10 # 序列训练长度
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01 # 采样间隔，因为 RNN 预测是离散值


def generate_data(seq):
    X = []
    y = []
    # 输入是 第 i 项和后面 TIMESTEPS-1 项合在一起
    # 输出是 第 i+TIEMSTEPS 项
    # 即用 sin 函数前面 TIMESTEPS 个点的信息，预测第 i+TIMESTEMPS 个点的函数值
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training):
    # 多层 LSTM
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    # 将多层 LSTM 结构连接成 RNN 网络并计算其前向结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs 是顶层 LSTM 在每一步的输出，维度是 [batch_size, time, HIDDEN_SIZE]
    # 我们只关注最后一个时刻
    output = outputs[:, -1, :]

    # 再加一层全连接层并计算损失，平方差损失函数
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    # 只在训练时计算损失函数和优化步骤
    if not is_training:
        return predictions, None, None

    # 计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 创建优化器
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                               optimizer="Adagrad", learning_rate=0.1)
    return predictions, loss, train_op


def train(sess, train_X, train_Y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))


def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
        predictions = []
        labels = []
        for i in range(TESTING_EXAMPLES):
            p, l = sess.run([prediction, y])
            predictions.append(p)
            labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels) ** 2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES+TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TRAINING_EXAMPLES+TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32
)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32
)))

with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)
