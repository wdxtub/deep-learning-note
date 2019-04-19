import scipy
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
import numpy as np

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
            float(size[0]) / img.shape[1],
            float(size[1]) / img.shape[2],
            1)
    return scipy.ndimage.zoom(img, factors, order=1)

def deprocess_image(x):
    # 通用函数，将一个张量转换为有效图像
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    
    x /= 2.
    x += 0.3
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

# 用于打开图像、改变图像大小以及将图像格式转化为 Inception V3 模型能够处理的张量
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img