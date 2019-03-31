from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# 最多使用单词数
max_features = 20000
# RNN 截断长度
maxlen = 80
batch_size = 32

# 加载数据并将单词转化成 ID，max_features 是最多使用的单词数
(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
print(len(trainX), 'train sequences')
print(len(testX), 'test sequences')

# 要把每句话的长度加工到一样
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)
# 输出维度
print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

# 构建模型
model = Sequential()
# 构建 Embedding 层
model.add(Embedding(max_features, 128))
# 构建 LSTM
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# 构建全连接层，构建 LSTM 层之后只会得到最后一个节点的输出
model.add(Dense(1, activation='sigmoid'))

# 指定损失函数
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练轮数
model.fit(trainX, trainY, batch_size=batch_size, epochs=15, validation_data=(testX, testY))

# 在测试数据上评测模型
score = model.evaluate(testX, testY, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])