from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

print("将 VGG 16 卷积基实例化")
# include_top 表示是否包含 Dense 分类器，这里我们不需要
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print("输出模型结构")
print(conv_base.summary())

base_dir = 'data/small'
