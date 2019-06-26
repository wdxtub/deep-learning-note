import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weight = tf.get_variable("weight", [], tf.float32, initializer=tf.random_normal_initializer())
bias = tf.get_variable("bias", [], tf.float32, initializer=tf.random_normal_initializer())
pred = tf.add(tf.multiply(x, weight, name="mul_op"), bias, name="add_op")

loss = tf.square(y - pred, name="loss")

optimizer = tf.train.GradientDescentOptimizer(0.01)

grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

# 收集值
tf.summary.scalar("weight", weight)
tf.summary.scalar("bias", bias)
tf.summary.scalar("loss", loss[0])

# 合并 summary
merged_summary = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter("./calc_graph_event")
summary_writer.add_graph(tf.get_default_graph())
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(500):
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1)*0.01 + 10
        _, summary = sess.run([train_op, merged_summary], feed_dict={x:train_x, y:train_y})
        summary_writer.add_summary(summary, step)

