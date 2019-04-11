from keras.applications import VGG16
from keras import backend as K
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("引入 VGG16 模型")
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

print("为过滤器的可视化定义损失张量")
lyaer_output = model.get_layer(layer_name).output
loss = K.mean(lyaer_output[:, :, :, filter_index])
print("获取损失相对于输入的梯度")
grads = K.gradients(loss, model.input)[0]
print("梯度标准化技巧")
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

print("获取 loss 和 grads")
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
print("通过随机梯度让损失最大化")
input_img_data = np.random.random((1, 150, 150, 3))*20 + 128. # 一张带有噪声的灰度图像
step = 1
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

def deprocess_image(x):
    # 将张量转换为有效图像的函数
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    # 过滤器可视化函数
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step = 1
    for _ in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)

print("可视化 block3_conv1 的最大响应")
plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

print("生成某一层的所有过滤器响应模式")
layer_name = 'block1_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img

print("转换下格式，不然无法正常显示")
results = np.clip(results, 0, 255).astype('uint8')
print("显示 result 网络")
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()