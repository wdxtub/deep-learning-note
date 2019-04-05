import tensorflow as tf
import mnist_inference
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

# 定义模型参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 日志文件相关
LOG_DIR= 'log/projector'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'
TENSOR_NAME = 'FINAL_LOGITS'

# 这里需要返回输出层矩阵
def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
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

    # 训练模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], 
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g" % (i, loss_value))
        
        # 计算 MNIST 测试数据对应的输出层矩阵
        final_result = sess.run(y, feed_dict={x: mnist.test.images})
    
    return final_result


# 生成可视化最终输出层向量所需要的日志文件
def visualisation(final_result):
    # 使用一个新的变量来保存最终输出层向量的结果
    # 因为 embedding 是通过 Tensorflow 中的变量完成的，所以 PROJECTOR 可视化的都是 TF 变量
    y = tf.Variable(final_result, name = TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # 通过 PROJECTOR 生成日志
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    # 指定 embedding 对应的原始数据信息
    embedding.metadata_path = META_FILE

    # 指定 sprite 图像及大小
    embedding.sprite.image_path = SPRITE_FILE
    embedding.sprite.single_image_dim.extend([28, 28])

    # 写入日志
    projector.visualize_embeddings(summary_writer, config)

    # 生成会话，写入文件
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


# 主函数先调用模型训练，再处理测试数据，最后将输出矩阵输出到 PROJECTOR 需要的日志文件中
def main(argv=None):
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)

if __name__ == '__main__':
    main()
