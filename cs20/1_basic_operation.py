import tensorflow as tf
a = tf.constant(2, name='a') # 指定名称
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

# Setup 1 执行时会出现如下的警告
# Your CPU supports instructions that this TensorFlow binary
# was not compiled to use: AVX2 FMA
# 可以通过下面的代码屏蔽
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Setup 2 在 Graph 定义完成后实际执行前，创建 summary 用来记录
writer = tf.summary.FileWriter('data/1-graphs', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(x))

# 注意要关闭 writer
writer.close()
