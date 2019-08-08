import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')


def fully_connected(x, output_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable('weights', [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b


def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')


with tf.variable_scope('two_layers') as scope:
    two_hidden_layers(x1)
    two_hidden_layers(x2)


writer = tf.summary.FileWriter('data/graphs/fully_connected', tf.get_default_graph())
writer.close()
