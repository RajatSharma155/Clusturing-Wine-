# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:41:04 2020

@author: Rajat sharma
"""

# Importing the libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the data set
dataset = pd.read_csv('Wine_quality')
X = dataset.iloc[:, :-1].values

# Implimenting Dendogram

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Wines')
plt.ylabel("Eulcidians")
plt.show()
   
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_cluster = 2, affinity="euclidean", linkage = 'ward')
y_hc = hc.fit_predict(X)