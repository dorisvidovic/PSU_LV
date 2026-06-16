from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

x = generate_data(500, 1)

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(x)
centers = kmeans.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis') # c=labels boji točke po grupama
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='*') 
plt.show()


z = linkage(x, method='complete')
plt.figure(figsize=(10, 7))
plt.title("Dendogram")
dendrogram(z)
plt.show()


inertias = []
K = range(1, 21)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(x)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertias, marker='o')
plt.xlabel("Broj klastera (k)")
plt.ylabel("Vrijednost kriterijske funkcije (inertia)")
plt.title("Elbow metoda")
plt.show()