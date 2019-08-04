import tensorflow as tf

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='mul') # 会以 Numpy 广播的形式进行计算

zeros = tf.zeros([2, 3], tf.int32)
ones = tf.ones([3, 2], dtype=tf.int32)
zeros_like = tf.zeros_like(zeros)
ones_like = tf.ones_like(ones)

fill_six = tf.fill([2, 3], 6)

lin_space = tf.lin_space(1.0, 8.0, 4)
six_range = tf.range(6)
limit_range = tf.range(3, 18, 3)

# 用正态分布产生随机数，默认是标准正态分布
random_normal = tf.random_normal([3, 3])
# 产生正态分布的值如果与均值的差值大于两倍的标准差，那就重新生成
truncated_normal = tf.truncated_normal([3, 3])
# 用均匀分布产生随机值，默认浮点数范围[0, 1)
random_uniform = tf.random_uniform([3, 3])
# 每一次都把其中的一些行换位置或者不换
random_shuffle = tf.random_shuffle([3, 3])
# random_crop 主要用于裁剪图片，这里不展示
# 从多项式分布中抽取样本，就是根据概率分布的大小，返回对应维度的下标序号
multinomial = tf.multinomial(random_normal, 5)
# 根据gamma分布个数，每个分布给出shape参数对应个数数据
random_gamma = tf.random_gamma([3, 3], 1, 2)

# 设定种子
tf.set_random_seed(314)

with tf.Session() as sess:
    print('x')
    print(sess.run(x))
    print('zeros')
    print(sess.run(zeros))
    print('ones')
    print(sess.run(ones))
    print('zeros_like')
    print(sess.run(zeros_like))
    print('ones_like')
    print(sess.run(ones_like))
    print('fill_six')
    print(sess.run(fill_six))
    print('lin_space')
    print(sess.run(lin_space))
    print('six_range')
    print(sess.run(six_range))
    print('limit_range')
    print(sess.run(limit_range))
    print('random_normal')
    print(sess.run(random_normal))
    print('truncated_normal')
    print(sess.run(truncated_normal))
    print('random_uniform')
    print(sess.run(random_uniform))
    print('random_shuffle')
    print(sess.run(random_shuffle))
    print('multinomial')
    print(sess.run(multinomial))
    print('random_gamma')
    print(sess.run(random_gamma))
    print('--------------')
    print('a')
    print(sess.run(a))
    print('b')
    print(sess.run(b))
    print('div(b,a)')
    print(sess.run(tf.div(b, a)))
    print('divide(b, a)')
    print(sess.run(tf.divide(b, a)))
    print('truediv(b, a)')
    print(sess.run(tf.truediv(b, a)))
    print('floordiv(b, a)')
    print(sess.run(tf.floordiv(b, a)))
    print('realdiv(b, a)') # 需要是实数
    #print(sess.run(tf.realdiv(b, a)))
    print('truncatediv(b, a)')
    print(sess.run(tf.truncatediv(b, a)))
    print('floor_div(b, a)')
    print(sess.run(tf.floor_div(b, a)))

