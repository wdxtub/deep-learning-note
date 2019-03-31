import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 处理输入数据的计算都放在 “input” 命名空间下
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的相关计算都放在 moving_average 命名空间下
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数放在 loss_function 命名空间下
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率等信息放在 train_step 命名空间下
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    writer = tf.summary.FileWriter("log", tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i % 1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_: ys},
                                               options=run_options,
                                               run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % i), global_step=i)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

    writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
