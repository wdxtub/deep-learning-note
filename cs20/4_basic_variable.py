import tensorflow as tf

# 尽量使用 tf.get_variable 而非 tf.Variable 来创建变量
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

with tf.Session() as sess:
    # 初始化单个
    sess.run(W.initializer)
    # 初始化部分
    sess.run(tf.variables_initializer([s, m]))
    # 初始化全部
    sess.run(tf.global_variables_initializer())
    # 可以用 eval 函数来获取值
    print(W.eval())

W = tf.Variable(10)
W.assign(100)
assign_op = W.assign(100)

my_var = tf.Variable(2, name="my_var")
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval()) # 并没有生效
    sess.run(assign_op)
    print(W.eval()) # 才会生效
    sess.run(my_var.initializer)
    print('my_var', my_var.eval())
    sess.run(my_var_times_two)
    print('my_var', my_var.eval())
    sess.run(my_var_times_two)
    print('my_var', my_var.eval())
    sess.run(my_var.assign_add(10))
    print('my_var', my_var.eval())
    sess.run(my_var.assign_sub(2))
    print('my_var', my_var.eval())

print('session 变量独立')
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))
print(sess2.run(W.assign_sub(2)))
print(sess1.run(W.assign_add(100)))
print(sess2.run(W.assign_sub(50)))

sess1.close()
sess2.close()
