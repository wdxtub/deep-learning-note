import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.Variable(tf.random_normal([100, 50]), name='h1_weights')
    b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]
    w2 = tf.Variable(tf.random_normal([50, 10]), name='h2_weights')
    b2 = tf.Variable(tf.zeros([10]), name='2_biases')
    logits = tf.matmul(h1, w2) + b2
    return logits


logits1 = two_hidden_layers(x1)
logits2 = two_hidden_layers(x2)

writer = tf.summary.FileWriter('data/graphs/no_sharing', tf.get_default_graph())
writer.close()
