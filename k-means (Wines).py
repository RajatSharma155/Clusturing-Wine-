# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:04:26 2020

@author: Rajat sharma
"""

# Importing the libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the data set
dataset = pd.read_csv('Wine_quality')
X = dataset.iloc[:, :-1].values

# Using elbow method to find the best no of Cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 5)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss )
plt.xlabel("Cluster")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
    
# Running the K-Means Cluster
kmeans = KMeans(n_clusters = 3, init = "k-means++", random_state = 5)
y_kmeans = kmeans.fit_predict(X)

new_dataset = dataset.copy()

new_dataset.insert(11, "Group", y_kmeans, True)

   
