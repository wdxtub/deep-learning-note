from keras import layers, models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# 依赖 pillow，安装 pip install pillow

import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

base_dir = 'data/small' # 保存小数据集
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

print("构造模型")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
print("添加 Dropout")
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print("模型结构")
print(model.summary())
print("编译模型")
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

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

print("利用批量生成器进行模型拟合（训练）")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50)
print("保存模型")
model.save('model/cats_and_dogs_small_2.h5')
print("绘制训练过程中损失和精度曲线")
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