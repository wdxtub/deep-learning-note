import numpy as np

print("Show an Array")
a = np.array([1,2,3,4])
print(a)

import time

print("Create two Arrays with 1000000 dimension")
a = np.random.rand(1000000)
b = np.random.rand(1000000)
print("Vectorication Version")
tic = time.time()
c = np.dot(a, b)
toc = time.time()
print("Result " + str(c) + " Time: " + str(1000*(toc-tic)) + "ms")
print("For Version") 
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()
print("Result " + str(c) + " Time: " + str(1000*(toc-tic)) + "ms")

