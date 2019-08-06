import tensorflow as tf
import tensorflow.contrib.eager as tfe

import w2v_utils

tf.enable_eager_execution()

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


class Word2Vec(object):
    def __init__(self, vocab_size, embed_size, num_sampled=NUM_SAMPLED):
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.embed_matrix = tf.Variable(tf.random.uniform(
                                          [vocab_size, embed_size]))
        self.nce_weight = tf.Variable(tf.random.truncated_normal(
                                        [vocab_size, embed_size],
                                        stddev=1.0 / (embed_size ** 0.5)))
        self.nce_bias = tf.Variable(tf.zeros([vocab_size]))

    def compute_loss(self, center_words, target_words):
        """Computes the forward pass of word2vec with the NCE loss."""
        embed = tf.nn.embedding_lookup(self.embed_matrix, center_words)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                             biases=self.nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=self.num_sampled,
                                             num_classes=self.vocab_size))
        return loss


def gen():
    yield from w2v_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES,
                                   VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW,
                                   VISUAL_FLD)


def main():
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([BATCH_SIZE]),
                                              tf.TensorShape([BATCH_SIZE, 1])))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
    model = Word2Vec(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE)
    grad_fn = tfe.implicit_value_and_gradients(model.compute_loss)
    total_loss = 0.0
    num_train_steps = 0
    while num_train_steps < NUM_TRAIN_STEPS:
        for center_words, target_words in tfe.Iterator(dataset):
            if num_train_steps >= NUM_TRAIN_STEPS:
                break
            loss_batch, grads = grad_fn(center_words, target_words)
            total_loss += loss_batch
            optimizer.apply_gradients(grads)
            if (num_train_steps + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(
                    num_train_steps, total_loss / SKIP_STEP
                ))
                total_loss = 0.0
            num_train_steps += 1


if __name__ == '__main__':
    main()

