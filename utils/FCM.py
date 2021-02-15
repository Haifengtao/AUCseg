import numpy as np
from sklearn.cluster import KMeans


class FCM:
    def __init__(self, image, n_clusters, centriod="kmeans", m=2, epsilon=0.01,
                 max_iter=150):
        """
        Fuzzy C means clustering.
        :param image: image array.
        :param n_clusters: number of clustering subclass.
        :param centriod: how to initial the centriods.
        :param m:m
        :param epsilon: bounding value.
        :param max_iter: max iteration.
        """
        self.image = image
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.shape = image.shape
        self.X = image.flatten().astype('float16')
        self.numPixels = image.size
        self.labels_ = None

        # -----------------Initial membership matrix-----------------
        self.U = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            self.U[idxii, ii] = 1

        # -----------------Initial centers-----------------
        if centriod == "kmeans":
            X_km = np.reshape(self.X, (-1, 1))
            model = KMeans(n_clusters=self.n_clusters, max_iter=20).fit(X_km)
            self.cluster_centers_ = model.cluster_centers_.T
        else:
            self.cluster_centers_ = np.linspace(np.min(image), np.max(image), n_clusters)
            self.cluster_centers_ = self.cluster_centers_.reshape(self.n_clusters, 1)  #

    def update_U(self):
        """Compute weights"""
        c_mesh, x_mesh = np.meshgrid(self.cluster_centers_, self.X)  # 形成两个等大小的隶属度矩阵，其中C被中心填充，X被像素填充
        power = 2. / (self.m - 1)  # 指数
        p1 = abs(x_mesh - c_mesh) ** power  # 公式的分子
        p2 = np.sum((1.0 / abs(x_mesh - c_mesh)) ** power, axis=1)  # 公式的分母
        return 1.0 / (p1 * p2[:, None])  # result

    def update_C(self):
        """Compute centroid of clusters"""
        num = np.dot(self.X, self.U ** self.m)
        den = np.sum(self.U ** self.m, axis=0)

        return num / den

    def fit(self):
        i = 0
        while True:
            old_u = np.copy(self.U)
            self.U = self.update_U()
            self.cluster_centers_ = self.update_C()
            d = np.sum(abs(self.U - old_u))
            if d < self.epsilon or i > self.max_iter:
                break
            i += 1
        self.segmentImage()

    def deFuzzify(self):
        return np.argmax(self.U, axis=1)  # 返回每一行最大值的下标

    def segmentImage(self):
        result = self.deFuzzify()
        self.labels_ = result.reshape((-1)).astype('uint8')
