import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import plot_model

max_features = 2000
max_len = 500

print("载入数据")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print("构造模型")
model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
print("模型结构")
print(model.summary())
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print("添加回调函数记录信息")
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='log', # 日志写入位置
        histogram_freq=1, # 每一轮都记录激活直方图
        embeddings_freq=1, # 每一轮之后记录嵌入数据
        embeddings_data=x_test[:100] # 这里需要添加这一行，原文没有
    )
]
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)
print("使用 tensorboard --logdir=log 启动，访问 localhost:6006")
print("绘制模型结构图片")
plot_model(model, to_file='log/model.png', show_shapes=True)