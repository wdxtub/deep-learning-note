import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 代码参考
# https://github.com/sbavon/Tensorflow-TFRecord/blob/f873fc7d69729eec68635ea5ee437d2b6bf02c7f/tf_record.py
# https://github.com/fanfanfeng/deeplearning/blob/af229b07e10749f1c0fe7a91a6032d0137639ea0/github_fork_partise/tf-stanford-tutorials/09_tfrecord_example.py
# https://github.com/adamleo/ai_courses_learning/blob/76fe801ac77d175236d7f897370eda87f016b5c5/tensorflow_stanford_tutorials/examples/09_tfrecord_example.py

def get_image_binary(filename):
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized, features={
        'label': tf.FixedLenFeature([],tf.int64),
        'shape': tf.FixedLenFeature([],tf.string),
        'image': tf.FixedLenFeature([],tf.string),
    }, name='features')

    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)

    image = tf.reshape(image, shape)
    label = tfrecord_features['label']
    return label, shape, image


def read_tfrecord(tfrecod_file):
    label, shape, image = read_from_tfrecord([tfrecod_file])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        label, image, shape = sess.run([label, image, shape])
        coord.request_stop()
        coord.join(threads)

    print(label)
    print(shape)
    plt.imshow(image)
    plt.show()


print('创建 TFRecord Writer')
writer = tf.io.TFRecordWriter('data/tfrecord/naruto.tfr')

print('获取 shape 和图像的二进制')
shape, binary_image = get_image_binary('data/images/naruto.jpeg')

label = 1

print('创建 tf.train.Features 对象')
features = tf.train.Features(feature={
    'label': _int64_feature(label),
    'shape': _bytes_feature(shape),
    'image': _bytes_feature(binary_image)
})

print('创建 Sample')
sample = tf.train.Example(features=features)

print('写入到 tfreord 中')
writer.write(sample.SerializeToString())
writer.close()
