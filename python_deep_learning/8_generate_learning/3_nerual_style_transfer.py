from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19

target_image_path = 'data/transfer-target.png' # 想要变化的图像路径
style_reference_image_path = 'data/transfer-reference.png' # 风格图像的路径

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height) # 生成图像的尺寸

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # 加上 ImageNet 的平均像素值
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x