import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster


try:
    img = mpimg.imread('example_grayscale.png')
except FileNotFoundError:
    print("Greška: Datoteka nije pronađena. Provjerite putanju!")
    from scipy import misc
    img = misc.face(gray=True)

X = img.reshape((-1, 1)) 

n_colors = 10
k_means = cluster.KMeans(n_clusters=n_colors, n_init=1)
k_means.fit(X) 


values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_


img_compressed = np.choose(labels, values)


img_compressed.shape = img.shape


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Originalna slika (256 nijansi)")
plt.axis('off') # Isključujemo koordinate radi ljepšeg prikaza

# Prikaz rezultata
plt.subplot(1, 2, 2)
plt.imshow(img_compressed, cmap='gray')
plt.title(f"Kvantizirana slika ({n_colors} nijansi)")
plt.axis('off')


plt.show()

