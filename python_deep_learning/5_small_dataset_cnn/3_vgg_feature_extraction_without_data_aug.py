from keras.applications import VGG16
from keras import layers, models, optimizers

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

print("将 VGG 16 卷积基实例化")
# include_top 表示是否包含 Dense 分类器，这里我们不需要
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print("输出模型结构")
print(conv_base.summary())

base_dir = 'data/small' # 保存小数据集
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        start = i * batch_size
        end = start + batch_size
        features[start: end] = features_batch
        labels[start: end] = labels_batch
        i += 1
        if start >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

print("平整形状")
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

print("定义并训练 Dense 分类器")
