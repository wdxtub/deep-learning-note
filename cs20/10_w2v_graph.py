import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utils
import w2v_utils

# 模型超参数
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # 词嵌入向量的维度
SKIP_WINDOW = 1             # 上下文的大小，1 表示使用前 1 个词和后 1 个词
NUM_SAMPLED = 64            # 使用 negative examples 数目
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'data/visualization'
SKIP_STEP = 5000

# 下载数据的参数
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 3000        # 用来可视化的 token 数量


def word2vec(dataset):
    """ 构建 W2V 的计算图，并且进行训练 """
    # Step 1: 从 dataset 获取数据
    with tf.name_scope('data'):
        iterator = dataset.make_initializable_iterator()
        center_words, target_words = iterator.get_next()

    """ Step 2 + 3: 定义 weight 和 embedding 的查找方式
    在 W2V 中，我们只关心 weights 
    """
    with tf.name_scope('embed'):
        embed_matrix = tf.get_variable('embed_matrix',
                                       shape=[VOCAB_SIZE, EMBED_SIZE],
                                       initializer=tf.random_uniform_initializer())
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embedding')

    # Step 4: 创建 NCE 相关变量并定义损失函数
    with tf.name_scope('loss'):
        nce_weight = tf.compat.v1.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
                                     initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
        nce_bias = tf.compat.v1.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

        # 定义损失函数
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE), name='loss')

    # Step 5: 定义优化器
    with tf.name_scope('optimizer'):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    utils.safe_mkdir('data/checkpoints')

    with tf.compat.v1.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.compat.v1.global_variables_initializer())

        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.compat.v1.summary.FileWriter('data/graphs/word2vec_simple', sess.graph)

        for index in range(NUM_TRAIN_STEPS):
            try:
                loss_batch, _ = sess.run([loss, optimizer])
                total_loss += loss_batch
                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        writer.close()


def gen():
    yield from w2v_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE,
                                   BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)


def main():
    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    word2vec(dataset)


if __name__ == '__main__':
    main()
