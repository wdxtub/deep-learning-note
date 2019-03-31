import glob
import os.path
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.platform import gfile

INPUT_DATA = 'data/flower_photos'
OUTPUT_FILE = 'data/flower_processed_data.npy'

# 测试数据和验证数据比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 读取数据并将数据分割成训练、验证和测试数据
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有子目录
    for sub_dir in sub_dirs:
        print("sub_dir ", sub_dir)
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

        if not file_list: continue

        # 处理图片数据
        for file_name in file_list:
            print("file:", file_name)
            # 读取并解析图片，转化为 299x299 给 inception-v3 模型来处理
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # 随机划分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    # 将数据随机打乱以获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    start = datetime.datetime.now()
    print("start", start)
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 通过 numpy 格式保存处理后的数据
        np.save(OUTPUT_FILE, processed_data)
    end = datetime.datetime.now()
    print("done", end)


if __name__ == '__main__':
    main()

