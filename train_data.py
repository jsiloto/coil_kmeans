## Initialisation

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

image = Image.open("coil-100/obj1__0.png")
imgvec = np.asarray(image)
df = imgvec.reshape(imgvec.shape[0]*imgvec.shape[1], imgvec.shape[2])

kmeans = KMeans(n_clusters=10)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
print(centroids)


