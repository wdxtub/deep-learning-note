from keras.applications import inception_v3
from keras import backend as K
import numpy as np
from utils import *

print("不需要训练模型，禁用所有与训练有关的操作")
K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
print("计算损失，即梯度上升过程中需要最大化的量")
print("这次我们要把多隔层所有过滤器的激活同时最大化")
print("设置额 DeepDream 配置")
layer_contributions = {
    'mixed2': 0.2, 
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5, # 层的名字可以在 model.summary 中看到
}
print("定义需要最大化的损失")
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.) # 定义损失时将层的贡献添加到这个标量变量中
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output # 获取层的输出
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # 将 L2 加入到 loss 中避免出现边界伪影
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling 

print("设置梯度上升过程")
dream = model.input # 用于保存生成的图像，即梦境图像
grads = K.gradients(loss, dream)[0] # 计算损失相对于梦境图像的梯度
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) # 将梯度标准化，重要技巧
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations): # 运行 iterations 次梯度上升
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

print("在多个连续尺度上运行梯度上升，改变下面超参数会有新的效果")
step = 0.01 # 梯度上升步长
num_octave = 3 # 运行梯度上升的尺度个数
octave_scale = 1.4 # 两个尺度大小比例
iterations = 20 # 在每个尺度上运行梯度上升的步数

max_loss = 10 # 如果损失增大到大于 10 就终止，不然结果会很丑

base_image_path = 'data/deepdream-origin.jpeg'

img = preprocess_image(base_image_path)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                        iterations=iterations,
                        step=step,
                        max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='data/dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')