# HC

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('C://Users/Prani/Desktop/python/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
                
#using the Dendrogram method to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method ='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidena Distances')
plt.show()

#fitting HC to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c = 'red', label = 'cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c = 'blue', label = 'cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c = 'green', label = 'target Customers')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c = 'cyan', label = 'cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c = 'magenta', label = 'cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()