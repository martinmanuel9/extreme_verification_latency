#%%
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


iris = datasets.load_iris()
x = iris.data
y = iris.target


estimator = KMeans(n_clusters=3)
y_kmeans = estimator.fit_predict(x)

#empty dictionaries

clusters_centroids=dict()
clusters_radii= dict()

'''looping over clusters and calculate Euclidian distance of 
each point within that cluster from its centroid and 
pick the maximum which is the radius of that cluster'''
print(np.shape(estimator.cluster_centers_))
for cluster in list(set(y)):
    clusters_centroids[cluster]=list(zip(estimator.cluster_centers_[:, 0],estimator.cluster_centers_[:,1]))[cluster]
    clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(x[y_kmeans == cluster, 0],x[y_kmeans == cluster, 1])])
    

#Visualising the clusters and cluster circles

fig, ax = plt.subplots(1,figsize=(7,5))

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
art = mpatches.Circle(clusters_centroids[0],clusters_radii[0], edgecolor='r',fill=False)
ax.add_patch(art)

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
art = mpatches.Circle(clusters_centroids[1],clusters_radii[1], edgecolor='b',fill=False)
ax.add_patch(art)

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
art = mpatches.Circle(clusters_centroids[2],clusters_radii[2], edgecolor='g',fill=False)
ax.add_patch(art)

#Plotting the centroids of the clusters
plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('kmeans.jpg',dpi=300)

# %%
