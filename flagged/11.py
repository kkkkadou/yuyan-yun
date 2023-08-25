import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 模拟虚拟的高维数据
np.random.seed(42)
n_samples = 100
n_features = 50  # 假设有50个特征
n_clusters = 3
X = np.random.randn(n_samples, n_features)

# 使用PCA降维到二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用K-Means进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 绘制PCA降维后的散点图，并根据聚类结果进行聚集
plt.figure(figsize=(6, 6))
for i in range(n_clusters):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i}', alpha=0.7)
plt.title('PCA降维后的散点图（聚类结果）')
plt.legend()
plt.show()

