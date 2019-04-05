import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# 日志文件名和地址
LOG_DIR = 'log/projector'
SPRITE_FILE = 'mnist_sprite.pdf'
META_FILE = 'mnist_meta.tsv'

# 使用给出的 MNIST 图片列表生成 sprite 图像
def create_sprite_image(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # sprite 可以理解为所有小图片拼成的大正方形矩阵
    m = int(np.ceil(np.sqrt(images.shape[0])))

    # 使用全 1 来初始化最终的大图片
    sprite_image = np.ones((img_h*m, img_w*m))

    for i in range(m):
        for j in range(m):
            # 计算当前图片编号
            cur = i * m + j
            if cur < images.shape[0]:
                # 将小图片的内容复制到最终的 sprite 图像
                sprite_image[i*img_h:(i+1)*img_h,
                             j*img_w:(j+1)*img_w] = images[cur]
    return sprite_image

# 加载 mnist 数据，制定 one_hot=False，得到的 labels 就是一个数字，而不是一个向量
mnist = input_data.read_data_sets("../data/mnist", one_hot=False)

# 生成 sprite 图像
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
sprite_image = create_sprite_image(to_visualise)

# 将生成的 sprite 放到相应的日志目录下
path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')

# 生成每张图片对应的标签文件并写道相应的日志目录下
path_for_mnist_sprites = os.path.join(LOG_DIR, META_FILE)
with open(path_for_mnist_sprites, 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index, label))
