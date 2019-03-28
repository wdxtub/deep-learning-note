import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关常数
INPUT_NODE = 784 # 图片像素数目
OUTPUT_NODE = 10 # 因为是 0~9，一共十个数字

# 神经网络参数
LAYER1_NODE = 500 # 只使用一个隐层，有 500 个节点
BATCH_SIZE = 100 # 一个 batch 的数量。数字越小，越接近随机梯度下降；越大，越接近梯度下降
LEARNING_RATE_BASE = 0.8 # 基础学习率
LEARNING_RATE_DECAY = 0.99 # 学习率的衰减率
REGULARIZATION_RATE = 0.0001 # 正则化的系数
TRAINING_STEPS = 30000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算前向传播结果，是一个 ReLU 三层全连接
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 没有提供滑动平均类，直接使用当前值
    if avg_class is None:
        # 计算隐藏层的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输入层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播结果，不用参数值的滑动平均
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 记录训练轮数
    global_step = tf.Variable(0, trainable=False)

    # 初始滑动平均衰减率
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有的参数上使用滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算前向传播结果，使用参数值的滑动平均
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前 batch 的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算 L2 正则化损失参数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵 + 正则化
    loss = cross_entropy_mean + regularization

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 更新参数与滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用滑动平均模型的前向结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 验证数据，判断停止的条件和评判训练的效果
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 实际测试数据
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation|test accuracy using avg model is %g | %g " %
                      (i, validate_acc, test_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束之后，验证最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using avg model is %g" % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
