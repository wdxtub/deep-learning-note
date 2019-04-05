import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMAYR_DIR = "log/histogram"

BATCH_SIZE = 100
TRAIN_STEPS = 3000

# 生成变量监控信息并定义生成监控信息日志的操作
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # tf.summary.histogram 函数不会立即执行，在 sess.run 中明确调用才会执行
        tf.summary.histogram(name, var)
        # 计算变量的平均值，给定命名空间
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# 生成一层全链接层神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 权重及监控变量
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name+'/weights')

        with tf.name_scope('biases'):
            # 偏置及监控变量
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络输出节点在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        
        activations = act(preactivate, name='activation')
        # 记录神经网络输出节点在经过激活函数之后的分布
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main(_):
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    # 定义输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 将输入向量还原图片的像素矩阵，并通过 tf.summary.image 函数定义将当前的图片信息写入日志
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)
    
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    # 计算交叉熵并监控交叉熵
    #logits=y, labels=tf.argmax(y_, 1)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross entropy', cross_entropy)
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 计算正确率与添加监控，如果是 batch，则是整个 batch 的正确率
    # 如果给定的数据为验证或者测试数据
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 合并所有 summary
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMAYR_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(summary, i)
        
    summary_writer.close()

if __name__ == '__main__':
    tf.app.run()