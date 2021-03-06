## Initialisation
from Tkinter import Image

import numpy as np
import os
from PIL import Image
import tables
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from random import randint
from numpy.linalg import norm
import matplotlib.pyplot as plt



def get_features(kmeans, imagevec):
    a  = kmeans.predict(imagevec)
    unique, counts = np.unique(a.tolist(), return_counts=True)
    counts = normalize(counts.reshape(1, -1))
    vals = dict(zip(unique.tolist(), counts[0].tolist()))
    b = np.zeros(centroids.shape[0])
    for index, intensity in vals.iteritems():
        b[index] = intensity
    return b


# Build the training dataset
# 1 - Load all images

list_of_filenames = [filename for filename in os.listdir('./coil-100') if filename.endswith(".png")]
list_of_filenames.sort()

# 2 - Choose a subsample of images
# Choosing all images

# 3 - Build a giant numpy array with all pixels from all images
list_of_imgvecs = []
for filename in list_of_filenames:
    img = Image.open('./coil-100/' + filename)
    imgvec = np.asarray(img)
    imgvec = imgvec.reshape(imgvec.shape[0] * imgvec.shape[1], imgvec.shape[2])
    img.close()
    list_of_imgvecs += [imgvec]

database = np.concatenate(list_of_imgvecs)
print "Database Loaded"
print "Database size = {}".format(database.shape)

# 4 - Subsample this array in some way resulting in our training dataset
percent = 0.001
size = database.shape[0]
subsample = np.random.choice(size, int(size*percent))
database = database[subsample, :]
print "Database Subsampled"
print "Database size = {}".format(database.shape)

# Train on our dataset
# 5 - Apply K-Means on the dataset giving our feature space
kmeans = KMeans(n_clusters=10)
kmeans.fit(database)
centroids = kmeans.cluster_centers_
print(centroids)


# 6/7 - Load all images separately and  Convert each image to the feature space
list_of_features = []
for filename in list_of_filenames:
    img = Image.open('./coil-100/' + filename)
    imgvec = np.asarray(img)
    imgvec = imgvec.reshape(imgvec.shape[0] * imgvec.shape[1], imgvec.shape[2])
    img.close()
    feat = get_features(kmeans, imgvec)
    list_of_features += [feat]


# Rank images
# Chose a single image in feature space
img_index = randint(0, len(list_of_filenames))
original = Image.open('./coil-100/' + list_of_filenames[img_index])

# Calculate the euclidean distance from every image in feature space from A

img_feature = list_of_features[img_index]
list_of_distances = [norm(img_feature - feat) for feat in list_of_features]
print list_of_distances

# Sort all distances
sort_indexes = sorted(range(len(list_of_distances)), key=lambda k: list_of_distances[k])



# Choose 10 smaller distances
# Plot images

original.show()

plt.figure(1)
for i in range(0, 8):
    img = Image.open('./coil-100/' + list_of_filenames[sort_indexes[i]])
    plt.subplot(241 + i)
    plt.imshow(img)

plt.show()

