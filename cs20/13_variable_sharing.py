import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def two_hidden_layers_2(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable('h1_weights', [100, 50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('h1_biases', [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]
    w2 = tf.get_variable('h2_weights', [50, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('h2_biases', [10], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(h1, w2) + b2
    return logits


with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers_2(x1)
    scope.reuse_variables()
    logits2 = two_hidden_layers_2(x2)


writer = tf.summary.FileWriter('data/graphs/variable_sharing', tf.get_default_graph())
writer.close()
