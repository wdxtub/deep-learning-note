import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

num_classes = 10
img_rows, img_cols = 28, 28

f = np.load("/Users/dawang/Documents/GitHub/deep-learning-note/tensorflow_in_action/data/mnist/mnist.npz")
trainX, trainY = f['x_train'], f['y_train']
testX, testY = f['x_test'], f['y_test']
f.close()

# 根据不同的底层来设置输入的格式
if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    # 黑白，所以第一维是 1
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 转换图像像素
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 2550.0

# 答案 one-hot 编码
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

model = Sequential()
# 深度 32， 过滤器 5x5
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
# 过滤器 2x2 的池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# 深度 64， 过滤器 5x5
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
# 卷积层的输出拉直，作为全连接输入
model.add(Flatten())
# 全连接层，500 个节点
model.add(Dense(500, activation='relu'))
# 全连接得到最后输出
model.add(Dense(num_classes, activation='softmax'))

# 定义损失函数
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

# 具体训练过程
model.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          validation_data=(testX, testY))

# 测试数据集上计算准确率
score = model.evaluate(testX, testY)
print('Test loss:', score[0])
print('Test accuracy:', score[1])