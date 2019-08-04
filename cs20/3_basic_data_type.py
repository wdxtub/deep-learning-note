import tensorflow as tf
import numpy as np

# 标量会被认为是 0 维张量
t0 = 19
# 1 维数组会被认为是 1 维张量
t1 = [b"apple", b"peach", b"grape"]
# 2 维数字会被认为是 2 维张量
t2 = [[True, False, False],
      [False, False, True],
      [False, True, False]]

with tf.Session() as sess:
    print('t0')
    print(sess.run(tf.zeros_like(t0)))
    print('t1')
    print(sess.run(tf.zeros_like(t1)))
    print('t2')
    zeros_like = tf.zeros_like(t2)
    print(type(zeros_like))
    result = sess.run(zeros_like)
    print(result)
    print(type(result)) # 注意观察类型
    print(sess.run(tf.ones_like(t2)))


# 类型判断
if tf.int32 == np.int32:
    print('tf.int32 is np.int32')