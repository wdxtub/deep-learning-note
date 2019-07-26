# -*- coding: UTF-8 -*-

import tensorflow as tf

a = tf.constant(1, dtype=tf.float32, shape=[1, 1], name='a')
b = tf.constant(2, dtype=tf.float32, shape=[1, 1], name='b')
c = tf.constant(3, dtype=tf.float32, shape=[1, 1], name='c')

s = tf.add(a, b, name='sum')
re = tf.multiply(c, s, name='mul')

sess = tf.Session()
print sess.run(s)
print sess.run(re)

