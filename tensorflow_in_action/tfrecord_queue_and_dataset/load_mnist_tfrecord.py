import tensorflow as tf

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["data/output.tfrecords"])

# 读取一个样例
_, serialized_example = reader.read(filename_queue)
# 解析读入的样例
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    }
)

# 解析成像素数组
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    print(sess.run([image, label, pixels]))

