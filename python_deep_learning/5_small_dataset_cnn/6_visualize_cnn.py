from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
# 忽略除 0 错误
np.seterr(divide='ignore',invalid='ignore')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("载入第二步生成的模型")
model = load_model('model/cats_and_dogs_small_2.h5')
print("模型结构")
print(model.summary())

print("选定一张图像作为测试，不能是训练图像")
img_path = 'data/small/test/cats/cat.1765.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print("图像形状", img_tensor.shape)
print("显示测试图像")

plt.imshow(img_tensor[0])
plt.show()

print("用一个输入张量和一个输出张量列表将模型实例化")
print("提取前 8 层的输出")
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input, outputs=layer_outputs)

print("以预测模型运行模型")
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print("第一个卷积层的激活", first_layer_activation.shape, "有 32 个通道")
print("可视化第 4 个通道")
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
print("可视化第 7 个通道")
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()

print("将每个中间层可视化")
layer_names = []
print("获取每层的名称")
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] # 特征图中的特征个数
    size = layer_activation.shape[1] # 特征图形状为 (1, size, size, n_features)
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # 处理特征使显示起来更好看
            channel_image -= channel_image.mean()
            # 避免除 0
            channel_image /= (channel_image.std()+1e-5)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col+1)*size,
                        row * size : (row+1)*size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()