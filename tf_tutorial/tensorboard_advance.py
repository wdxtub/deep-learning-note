import numpy as np
import tensorflow as tf
import random

x = tf.placeholder(tf.float32, shape=[2, 3])
y = tf.placeholder(tf.float32, shape=[2, 2])

weight = tf.get_variable("weight", [3, 2], tf.float32, initializer=tf.random_normal_initializer())
bias = tf.get_variable("bias", [2, 2], tf.float32, initializer=tf.random_normal_initializer())
mul_op = tf.matmul(x, weight, name="mul_op")
pred = tf.add(mul_op, bias, name="add_op")

loss = tf.square(y - pred, name="loss")

optimizer = tf.train.GradientDescentOptimizer(0.0003)

grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

# 收集值
tf.summary.histogram("weight", weight)
tf.summary.histogram("bias", bias)

# 合并 summary
merged_summary = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter("./calc_graph_advance")
summary_writer.add_graph(tf.get_default_graph())
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(7000):
        train_x = [
            [random.uniform(1, 10), random.uniform(1, 10), random.uniform(1, 10)],
            [random.uniform(1, 10), random.uniform(1, 10), random.uniform(1, 10)]
        ]
        y1 = train_x[0][0] * 1 + train_x[0][1] * 2 + train_x[0][2] * 3 + 10
        y2 = train_x[0][0] * 4 + train_x[0][1] * 5 + train_x[0][2] * 3 + 20
        y3 = train_x[1][0] * 1 + train_x[1][1] * 2 + train_x[1][2] * 3 + 30
        y4 = train_x[1][0] * 4 + train_x[1][1] * 5 + train_x[1][2] * 3 + 40

        train_y = [[y1, y2], [y3, y4]]
        _, summary, weight_value = sess.run([train_op, merged_summary, weight], feed_dict={x:train_x, y:train_y})
        summary_writer.add_summary(summary, step)
        if step % 1000 == 0:
            for w in weight_value:
                print(w)
            print("")

# tensorboard
# tensorboard --logdir=./calc_graph_advance