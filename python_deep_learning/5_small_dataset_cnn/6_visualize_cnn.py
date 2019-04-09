from keras.models import load_model

print("载入第二步生成的模型")
model = load_model('model/cats_and_dogs_small_2.h5')
print("模型结构")
print(model.summary())

print("选定一张图像作为测试，不能是训练图像")