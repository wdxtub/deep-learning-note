import tensorflow as tf

# 先进先出队列
queue = tf.FIFOQueue(100, "float")
# 入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 启动五个线程，每个线程运行 enqueue_op 操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 加入 TF 计算图
tf.train.add_queue_runner(qr)
# 定义出队列操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    coord.request_stop()
    coord.join(threads)
