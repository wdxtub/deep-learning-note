import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

# 每次训练 1000 张
batch_size = 1000

# 设定输入和输出格式
inputs = tf.placeholder(tf.float32, [batch_size, 784])
targets = tf.placeholder(tf.float32, [batch_size, 10])

# 第一层 500 个神经元的全连接
with tf.variable_scope("layer_1"):
    fc_1_out = fc(inputs, num_outputs=500, activation_fn=tf.nn.sigmoid)
# 第二层 784 个神经元的全连接
with tf.variable_scope("layer_2"):
    fc_2_out = fc(fc_1_out, num_outputs=784, activation_fn=tf.nn.sigmoid)
# 第三层 10 个神经元的输出层
with tf.variable_scope("layer_3"):
    logits = fc(fc_2_out, num_outputs=10)

# 设定 loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

if __name__ == '__main__':
    mnist_save_dir = 'data'
    mnist = input_data.read_data_sets(mnist_save_dir, one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # 只训练 3 个 batch，主要看性能
        for i in range(3):
            print("epoch %d start" % (i+1))
            batch_input, batch_target = mnist.train.next_batch(batch_size)
            feed_dict = {inputs: batch_input,
                         targets: batch_target}

            sess.run(train_op,
                     feed_dict=feed_dict,
                     options=options,
                     run_metadata=run_metadata)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('trace/timeline_2_step_%d.json' % (i+1), 'w') as f:
                f.write(chrome_trace)

print("all done")
