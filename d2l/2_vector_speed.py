from time import time

import torch

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] =  a[i] + b[i]
print(time() - start)

print('矢量加法')
start = time()
d = a + b
print(time() - start)
