import tensorflow as tf
import keras

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import subprocess

# Reference https://www.tensorflow.org/tfx/tutorials/serving/rest_simple

tf.logging.set_verbosity(tf.logging.ERROR)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

# 这里我们启动 tfserving 服务器

'''
以下在命令行执行，确保已经完成 docker 环境的安装以及 tfserving 镜像准备
docker run -t --rm -p 8501:8501 \
    -v "/Users/wdxtub/Documents/GitHub/deep-learning-note/tf_serving/model:/models/1" \
    -e MODEL_NAME=1\
    tensorflow/serving &
!!!注意，这里 填写 1 的内容一定要对应文件夹名称，不然无法启动
'''


print("[Step 1]Show Random Image from Test Set")
def show(idx, title):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()

import random
rando = random.randint(0,len(test_images)-1)
show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))


print("[Step 2]Prepare Data")
import json
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

# Reference https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
print("[Step 3]Make a request to your model in TensorFlow Serving")

import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/1:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

for i in range(3):
    show(i, 'Model Result {} (class {}), True Result a {} (class {})'.format(
        class_names[np.argmax(predictions[i])], test_labels[i], 
        class_names[np.argmax(predictions[i])], test_labels[i]))

print("ALL DONE")