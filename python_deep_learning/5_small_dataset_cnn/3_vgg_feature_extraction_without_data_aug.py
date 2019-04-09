from keras.applications import VGG16
from keras import layers, models, optimizers

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
            loss='binary_crossentropy',
            metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
print("保存模型")
model.save('model/cats_and_dogs_small_3.h5')

print("绘制结果")
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
print("模型几乎从一开始就过拟合，因为没有用数据增强，而数据增强对防止小型图像数据集的过拟合非常重要")
