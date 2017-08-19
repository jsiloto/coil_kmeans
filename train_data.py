## Initialisation

import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans


# Instructions:
# 1- Download coil-100.zip from
#    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
# 2- Extract it under ./coil-100

# image = Image.open("coil-100/obj1__0.png")
# imgvec = np.asarray(image)
# df = imgvec.reshape(imgvec.shape[0]*imgvec.shape[1], imgvec.shape[2])
#
# kmeans = KMeans(n_clusters=10)
# kmeans.fit(df)
#
# labels = kmeans.predict(df)
# centroids = kmeans.cluster_centers_
# print(centroids)


# Work Script

# Build the training dataset
# 1 - Load all images

list_of_filenames = [filename for filename in os.listdir('./coil-100') if filename.endswith(".png")]
list_of_filenames.sort()
print list_of_filenames

# 2 - Choose a subsample of images
# Choosing all images
list_of_imgs = [Image.open('./coil-100/' + filename) for filename in list_of_filenames]


# 3 - Build a giant numpy array with all pixels from all images
list_of_imgvecs = [np.asarray(img) for img in list_of_imgs]
list_of_imgvecs = [imgvec.reshape(imgvec.shape[0]*imgvec.shape[1], imgvec.shape[2]) for imgvec in list_of_imgvecs]

database = np.concatenate(list_of_imgvecs)
print "OK"
print database.shape


# 3 - Build a giant numpy array with all pixels from all images
# 4 - Subsample this array in some way resulting in our training dataset

# Train on our dataset
# 5 - Apply K-Means on the dataset giving our feature space
# 6 - Load all images separately
# 7 - Convert each image to the feature space

# Rank images
# Chose a single image in feature space(A)
# Calculate the euclidean distance from every image in feature space from A
# Sort all distances
# Choose 10 smaller distances
# Plot images


