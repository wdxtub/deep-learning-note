import numpy as np

a = np.arange(20,0,-1).reshape((5,4))
print('a', a)
print('每个元素排序', np.sort(a))
print('转置（两种方法）', np.transpose(a), a.T)
print('限定最大最小值', np.clip(a, 5, 15))
print('选取第二行', a[1])
print('选取第二行第二个元素（两种方式)', a[1][1], a[1, 1])
print('选取第三行前两个元素',a[2, 0:2])
print('打印行')
for row in a:
    print(row)
print('打印列')
for col in a.T:
    print(col)
print('多维转一维', a.flatten())

print('数组合并')
a = np.array([1,1,1])
b = np.array([2,2,2])
print('a', a)
print('b', b)
c = np.vstack((a,b))
print('c = a 和 b 上下合并', c)
print('a b c 的 shape', a.shape, b.shape, c.shape)
d = np.hstack((a,b))
print('d = a 和 b 左右合并', d)
print('a b d 的 shape', a.shape, b.shape, d.shape)
new_a = a[np.newaxis, :]
print('数组 a 转为矩阵', new_a, new_a.shape)
new_b = b[:, np.newaxis]
print('数组 b 转为矩阵', new_b, new_b.shape)

print('多个矩阵合并')
a = a[:, np.newaxis]
b = b[:, np.newaxis]
print('a', a)
print('b', b)
c = np.concatenate([a, b, b, a], axis=0)
print('纵向合并', c)
d = np.concatenate([b, a, a, b], axis=1)
print('横向合并', d)

print('切分矩阵')
a = np.arange(24).reshape((4,6))
print('a', a)
b = np.split(a, 2, axis=1)
new_b = np.hsplit(a, 2)
print('纵向切分（两种写法）', b, new_b)
c = np.split(a, 2, axis=0)
new_c = np.vsplit(a, 2)
print('横向切分（两种写法）', c, new_c)