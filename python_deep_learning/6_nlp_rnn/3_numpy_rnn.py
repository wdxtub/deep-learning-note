import numpy as np

timesteps = 100
print("输入序列的时间步数", timesteps)
input_features = 32
print("输入特征空间的维度", input_features)
output_features = 64
print("输出特征空间的维度", output_features)

print("输入数据是随机噪声")
inputs = np.random.random((timesteps, input_features))
print("初始状态：全零向量")
state_t = np.zeros((output_features,))
print("创建随机的权重矩阵")
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t # 更新网络状态

final_output_sequence = np.stack(successive_outputs, axis=0)
