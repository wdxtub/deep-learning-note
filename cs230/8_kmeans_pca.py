import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ui
import scipy.io as sio
import scipy.optimize as opt
from scipy.io import loadmat
import seaborn as sb

print('KMeans 聚类简单例子')
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx
data = loadmat('data/ex7data2.mat')
X = data['X']
initial_centroids = initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

print('测试函数正常工作')
idx = find_closest_centroids(X, initial_centroids)
print(idx[0:3])
print('查看数据坐标')
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print(data2.head())
print('绘制图像')
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids
print('计算中心点')
print(compute_centroids(X, idx, 3))
print('定义 Kmeans 函数')
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    
    return idx, centroids
print('执行 Kmeans 并绘制图像')
cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()
print('定义初始化聚类中心函数')
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids
print('测试函数')
print(init_centroids(X, 3))

ui.split_line1()
print('利用 kmeans 进行图像压缩，找到最具代表性的少数颜色，然后映射到低维的颜色空间')
print('载入图片')
image_data = loadmat('data/bird_small.mat')
print(image_data)
print('查看单张图片尺寸')
A = image_data['A']
print(A.shape)
plt.imshow(A)
plt.show()
print('预处理图片')
# normalize value ranges
A = A / 255.

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print('预处理后 X 的大小')
print(X.shape)
print('提取颜色中心点并复原图片')
# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]
print(X_recovered.shape)
# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print(X_recovered.shape)
print('展示复原后的图片')
plt.imshow(X_recovered)
plt.show()

ui.split_line1()
print('利用 Scikit-Learn 实现 Kmeans')
from skimage import io

# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('data/bird_small.png') / 255.
io.imshow(pic)
plt.show()
print('图片尺寸')
print(pic.shape)
# serialize data
data = pic.reshape(128*128, 3)
print(data.shape)
from sklearn.cluster import KMeans#导入kmeans库

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)
print('获取中心点')
centroids = model.cluster_centers_
print(centroids.shape)

C = model.predict(data)
print(C.shape)
print('压缩图片')
compressed_pic = centroids[C].reshape((128,128,3))
io.imshow(compressed_pic)
plt.show()

ui.split_line1()
print('PCA 部分')
print('载入数据')
data = loadmat('data/ex7data1.mat')
print(data)
print('查看数据分布')
X = data['X']

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()
print('定义 PCA 函数')
def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()
    
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    
    # perform SVD
    U, S, V = np.linalg.svd(cov)
    
    return U, S, V

U, S, V = pca(X)
print(U, S, V)
print('定义映射函数')
def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

print('映射')
Z = project_data(X, U, 1)
print(Z)

print('定义恢复函数')
def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)
print('恢复')
X_recovered = recover_data(Z, U, 1)
print(X_recovered)
print('查看恢复后的数据')
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()

ui.split_line2()
print('PCA 压缩人脸')
faces = loadmat('data/ex7faces.mat')
X = faces['X']
print(X.shape)

def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

face = np.reshape(X[3,:], (32, 32))
print('查看某张人脸图片')
plt.imshow(face)
plt.show()
print('应用 PCA')
U, S, V = pca(X)
Z = project_data(X, U, 100)
print('恢复并再次展示出来')
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)
plt.show()