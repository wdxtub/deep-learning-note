from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras import backend as K

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#  pip install opencv-python
import cv2

print("载入模型，带 Dense 分类器")
model = VGG16(weights='imagenet')

print("预处理一张输入图片")
name = 'cat.664'
img_path = 'data/small/train/cats/%s.jpg' % name
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_tensor = x / 255.
x = preprocess_input(x)


print("显示图像")
plt.imshow(img_tensor[0])
plt.show()


print("进行预测")
preds = model.predict(x)
print('预测结果:', decode_predictions(preds, top=3)[0])
print("索引编号", np.argmax(preds[0]))

print("应用 Grad-CAM 算法")
cat_output = model.output[:, np.argmax(preds[0])]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(cat_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
print("标准化热力图")
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

print("加载原始图像")
img = cv2.imread(img_path)
print("调整热力图大小")
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
print("热力图转换为 RGB 格式")
heatmap = np.uint8(255*heatmap)
print("应用热力图到原始图像")
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('data/%s_cam.jpg' % name, superimposed_img)
