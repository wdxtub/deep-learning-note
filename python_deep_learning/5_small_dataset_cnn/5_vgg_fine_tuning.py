from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers

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
print("这种方法计算代价更高，采用了数据增强，没有 GPU 则不要运行")

print("在卷积层上添加一个 Dense 分类器")
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print("现在的模型架构")
print(model.summary())
print("参数非常多！")

print("冻结卷积基，不要重新训练，不然就没有利用预训练的效果了")
conv_base.trainable = False

print("数据预处理")
print("图像除以 255 缩放，并应用数据增强")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
print("不能增强测试数据")
test_datagen = ImageDataGenerator(rescale=1./255)
print("调整图片大小为 150x150")
train_generator = train_datagen.flow_from_directory(train_dir, 
            target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = train_datagen.flow_from_directory(validation_dir, 
            target_size=(150, 150), batch_size=20, class_mode='binary')  

print("编译模型")
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)



# 从这往上跟第四步一样
print("冻结直到某一层的所有层")
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print("微调模型")
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

print("保存模型")
model.save('model/cats_and_dogs_small_5.h5')

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

print("重新绘制以看清规律。删除前十个点；将每个数据点替换为前面数据点的指数移动平均值")
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.clf()
plt.plot(epochs, smooth_curve(acc) , 'bo', label='Smoothed Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print("在测试数据上最终评估这个模型")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('Test Acc:', test_acc)