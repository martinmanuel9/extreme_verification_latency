#%%
#!/usr/bin/env python 
"""
Application:        Micro-Cluster Classification
File name:          mclassification.py
Author:             Martin Manuel Lopez
Creation:           11/17/2022

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""

# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import benchmark_datagen as bdg
import compose_data_gen as cbdg
from sklearn.cluster import Birch, KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

class MClassification(): 
    def __init__(self, 
                classifier,
                dataset,
                method,
                NClusters:int = 10): 
        """
        """
        self.classifier = classifier
        self.dataset = dataset
        self.NClusters = NClusters
        self.method = method
        self.class_cluster = {}
        self.centroid = {}
        self.centroid_radii = {}
        self.clusters = {}
        self.mcluster = {}
        self.X = {}
        self.Y = {}
        self.T = {}
        self._initialize()
        
    
    def _setData(self):
        
        data_gen = cbdg.COMPOSE_Datagen()
        # get data, labels, and first labels synthetically for timestep 0
        # data is composed of just the features 
        # labels are the labels 
        # core supports are the first batch with added labels 
        data, labels, first_labels, dataset = data_gen.gen_dataset(self.dataset)
        ts = 0 
        # set dataset (all the data features and labels)
        for i in range(0, len(data[0])):
            self.X[ts] = data[0][i]
            ts += 1
        # set all the labels 
        ts = 0
        for k in range(0, len(labels[0])):
            self.Y[ts] = labels[0][k]
            ts += 1
        # gets first core supports from synthetic
        self.T = np.squeeze(first_labels)


    def _initialize(self):
        """
        Get initial labeled data T 
        Begin MC for the labeled data
        """
        self._setData()
        # initial set of cluster based on inital labeled data T; using next time step from datastream for preds
        if self.method == 'kmeans':
            clusters = np.max(np.unique(self.T[:,-1])).astype(int) + 1 # adding an additional cluster
            kmeans_model = KMeans(n_clusters= clusters)
            self.centroid[0], self.centroid_radii[0] = self._create_centroid(inCluster = kmeans_model, fitCluster = kmeans_model.fit_predict(self.T), x= self.X[0] , y=self.T )
            kmeans_model.fit(self.T)
            self.preds = kmeans_model.predict(self.X[1])
            self.clusters[0] = kmeans_model.cluster_centers_
        elif self.method == 'gmm':
            gmm_model = GMM(n_components=self.NClusters)
            gmm_model.fit(self.T) 
            self.preds = gmm_model.predict(self.Y[1])
            self.clusters[0] = self.preds
        elif self.method == 'birch':
            birch_model = Birch(branching_factor=50, n_clusters= self.NClusters)
            birch_model.fit(self.T)
            self.preds = birch_model.predict(self.Y[1])
            self.clusters[0] = self.preds

        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class.

        for i in range(self.NClusters):
                xhat = self.X[i][self.preds]
                mode_val,_ = stats.mode(xhat)
                self.class_cluster[i] = mode_val

    def _create_mclusters(self, inCluster, yhat):
        """
        Clustering options:
        1. k-means
        2. GMM 
        3. Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)
        
        MC is defined as a 4 tuple (N, LS, SS, y) where:
        N = number of data points in a cluster
        LS = linear sum of N data points 
        SS = square sum of data points 
        y = label for a set of data points
        """
        # develop microcluster based on inputs 
        mcluster = {}
        N = len(inCluster)
        LS = np.sum(inCluster)
        SS = 0
        for s in range(len(inCluster)):
            SS += s ** 2
        mcluster['N'] = N
        mcluster['LS'] = LS
        mcluster['SS'] = SS
        mcluster['yhat'] = yhat

        return mcluster
        
    def _classify(self, trainData, trainLabel, testData):
        """
        Inputs include training data, training label, test data
        Two classifiers 
        1. K Nearest Neighbor
        2. Support Vector Machine
        """
        
        if len(trainData) >= len(trainLabel):
            indx = np.unique(np.argwhere(~np.isnan(trainLabel))[:,0])
            trainData = trainData[indx]
        elif len(trainLabel) >= len(trainData):
            indx = np.unique(np.argwhere(~np.isnan(trainData))[:,0])
            trainLabel = trainLabel[indx]
        
        if self.classifier == 'knn':
            knn = KNeighborsClassifier(n_neighbors=10).fit(trainData, trainLabel)   # KNN.fit(train_data, train label)
            predicted_label = knn.predict(testData)
        elif self.classifier == 'svm':
            svm_mdl = SVC(gamma='auto').fit(trainData, trainLabel)                  # fit(Xtrain, X_label_train)
            predicted_label = svm_mdl.predict(testData)

        return predicted_label

    def _create_centroid(self, inCluster, fitCluster, x, y):
        """
        inCluster = cluster model 
        fitCluster = fitted model
        x = datastream
        y = label
        """
        cluster_centroids = {}
        cluster_radii = {}
        if self.method == 'kmeans':
            for cluster in list(set(y[:,-1].astype(int))):
                cluster_centroids[cluster] = list(zip(inCluster.cluster_centers_[:,0], inCluster.cluster_centers_[:,1]))[cluster]
                cluster_radii[cluster] = max([np.linalg.norm(np.subtract(i, cluster_centroids[cluster])) for i in zip(x[fitCluster == cluster, 0], x[fitCluster == cluster, 1])])

            fig, ax = plt.subplots(1,figsize=(7,5))
            plot_iter = list(set(y[:,-1].astype(int)))
            for i in plot_iter:

                plt.scatter(x[fitCluster == i, 0], x[fitCluster == i, 1], s = 100, c = np.random.rand(3,), label = i)
                art = mpatches.Circle(cluster_centroids[i],cluster_radii[1], edgecolor='b', fill=False)
                ax.add_patch(art)

            #Plotting the centroids of the clusters
            plt.scatter(inCluster.cluster_centers_[:, 0], inCluster.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

            plt.legend()
            plt.tight_layout()
            plt.show()

            return cluster_centroids, cluster_radii
        elif self.method == 'gmm':
            pass # need to determine how to calc radii for gmm 
        elif self.method == 'birch':
            pass # need to determine how calc radii for birch

    def _append_mcluster(self, yhat, inCentroid, inRadii):
        """
        Method determines if yhat (preds) can be appended to the mcluster based on the radii that was determined by the original mcluster 
        We determine the yhat based on euclidean distance 
        """
    def run(self):
        """
        Micro-Cluster Classification
        """
        """
        1. Get first set of labeled data T  # done
        2. Create MCs of the labeled data of first data T # done 
        3. classify  # done 
            a. predicted label yhat_t for the x_t from stream is given by the nearest MC by euclidean distance 
            b. determine if the added x_t to the MC exceeds the maximum radius of the MC
                1.) if the added x_t to the MC does not exceed the MC radius -> update the (N, LS, SS )
                2.) if the added x_t to the MC exceeds radius -> new MC carrying yhat_t is created to carry x_t
            c. The 
        """

        past_centroid = self.clusters[0]
        yhat = self._classify(trainData=self.T, trainLabel=self.preds, testData=self.X[1])
        self.mcluster[0] = self._create_mclusters(inCluster= past_centroid, yhat= yhat)
        
        
        # temp_current_cluster = KMeans(n_clusters=self.NClusters, init=past_centroid, n_init=1).fit(self.X[0]).cluster_centers_
        

# test mclass
run_mclass = MClassification(classifier='knn', method = 'kmeans', dataset='UG_2C_2D').run()

# %%
