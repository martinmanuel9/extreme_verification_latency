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
import pandas as pd
import datagen_synthetic as cbdg
from sklearn.cluster import Birch, KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import classifier_performance as cp
from scipy import stats
import time 
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
        self.cluster_centers ={}
        self.preds = {}
        self.class_cluster = {}
        self.clusters = {}
        self.total_time = []
        self.performance_metric = {}
        self.avg_perf_metric = {}
        self.microCluster = {}
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

    def _cluster(self, X, y, ts):
        if self.method == 'kmeans':
            clusters = np.max(np.unique(y[:,-1])).astype(int) # adding an additional cluster
            kmeans_model = KMeans(n_clusters= clusters)
            # computes cluster centers and radii of cluster for initial ts
            self.microCluster[ts] = self._create_centroid(inCluster = kmeans_model, fitCluster = kmeans_model.fit_predict(y), x= X[ts] , y= y)
            # predict closes cluster center for each self.T belongs to
            self.clusters[ts] = kmeans_model.fit_predict(y)
            self.cluster_centers[ts] = kmeans_model.cluster_centers_
        elif self.method == 'gmm':
            gmm_model = GMM(n_components=self.NClusters)
            gmm_model.fit(y) 
            self.clusters[ts] = gmm_model.predict(self.Y[ts+1])
            self.cluster_centers[ts] = self.clusters[ts]
        elif self.method == 'birch':
            birch_model = Birch(branching_factor=50, n_clusters= self.NClusters)
            birch_model.fit(self.T)
            self.clusters[ts] = birch_model.predict(self.Y[ts+1])
            self.cluster_centers[ts] = self.clusters[ts]

        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class.
        for i in range(self.NClusters):
                xhat = self.X[i][self.clusters[ts]]
                mode_val,_ = stats.mode(xhat)
                self.class_cluster[i] = mode_val

    def _create_mclusters(self, inClusterpoints) :
        """
        Clustering options:
        1. k-means
        
        MC is defined as a 4 tuple (N, LS, SS, y) where:
        N = number of data points in a cluster
        LS = linear sum of N data points 
        SS = square sum of data points 
        y = label for a set of data points
        """
        # develop microcluster based on inputs
        mcluster = {}
        N = len(inClusterpoints)
        LS = sum(inClusterpoints)
        SS = np.zeros(np.shape(inClusterpoints)[1])
        for c in range(np.shape(inClusterpoints)[1]):
            for r in range(np.shape(inClusterpoints)[0]):
                SS[c] += inClusterpoints[:,c][r] ** 2
        mcluster['ClusterPoints'] = inClusterpoints
        mcluster['N'] = N
        mcluster['LS'] = LS
        mcluster['SS'] = SS
        mcluster['Centroid'] = LS / N
        mcluster['Radii'] = np.sqrt((SS/N) - ((LS/N)**2))
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
        length = len(np.unique(list(set(y[:,-1].astype(int)))))
        if self.method == 'kmeans':
            for cluster in range(length):
                cluster_centroids[cluster] = list(zip(inCluster.cluster_centers_[:,0], inCluster.cluster_centers_[:,1]))[cluster]
                cluster_radii[cluster] = max([np.linalg.norm(np.subtract(i, cluster_centroids[cluster])) for i in zip(x[fitCluster == cluster, 0], x[fitCluster == cluster, 1])])
            # print(cluster_radii)
            fig, ax = plt.subplots(1,figsize=(7,5))
            clust_iter = len(np.unique(list(set(y[:,-1].astype(int)))))
            # creates cluster data
            mcluster = {}
            for c in range(clust_iter):
                mcluster[c] = self._create_mclusters(inClusterpoints=x[fitCluster==c])
            # plot clusters
            for i in range(clust_iter):
                plt.scatter(x[fitCluster == i, 0], x[fitCluster == i, 1], s = 100, c = np.random.rand(3,), label ='Class '+ str(i))
                art = mpatches.Circle(cluster_centroids[i],cluster_radii[i], edgecolor='b', fill=False)
                ax.add_patch(art)
            #Plotting the centroids of the clusters
            plt.scatter(inCluster.cluster_centers_[:, 0], inCluster.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')
            plt.legend()
            plt.tight_layout()
            plt.show()
            # package for return
            microCluster = {}
            microCluster['MC'] = mcluster
            microCluster['Threshold'] = cluster_radii
            return microCluster
        elif self.method == 'gmm':
            pass # need to determine how to calc radii for gmm 
        elif self.method == 'birch':
            pass # need to determine how calc radii for birch

    def _append_mcluster(self, yhat,inData, inMicroCluster, ts):
        """
        Method determines if yhat (preds) can be appended to the mcluster based on the radii that was determined by the original mcluster 
        We determine the yhat based on euclidean distance 
        """
        preds = yhat
        data = inData
        step = ts
        currentClusterPoints = {}
        centroids = {}
        for mcKey in inMicroCluster['MC'].keys():
            centroids[mcKey] = inMicroCluster['MC'][mcKey]['Centroid'][0:2:1]
        
        for mcKey in inMicroCluster['MC'].keys():
            currentClusterPoints[mcKey] = inMicroCluster['MC'][mcKey]['ClusterPoints']
        thresholds = inMicroCluster['Threshold']
        # determine cluster for data for yhat 
        yhatClusters = np.max(np.unique(data[:,-1]).astype(int))
        
        yhatKmeans = KMeans(n_clusters=yhatClusters)
        # need to add this for create centroid method to take preds
        addToPreds = np.zeros((np.shape(data)[0], (np.shape(data)[1]-1)))
        preds = np.column_stack((addToPreds, preds))
        yhatCluster = self._create_centroid(inCluster = yhatKmeans, fitCluster = yhatKmeans.fit_predict(data), x= data , y=preds)
        self.microCluster[step] = yhatCluster


        yhatClusterPoints ={}
        for yhkey in yhatCluster['MC'].keys():
            yhatClusterPoints[yhkey] = yhatCluster['MC'][yhkey]['ClusterPoints']

        ## TODO: Is below really needed?? as we do a kmeans and we will classifiy p
        # calculate euclidean distance 
        to_newMC = []
        addToMC = {}
        indx_toAppend = {}
        indx_toNewMC = {}
        for t in thresholds:
            addToMC[t] = []
            indx_toAppend[t] = []
            indx_toNewMC[t] = []
        # original centroid centers 
        for c in centroids:
        # go through cluster of the clustered yhat (preds) created
            to_append = []
            indx_Append = []
            indx_NewMC = []
            for yhatClust in yhatClusterPoints: 
                # go through each point of each cluster 
                points = yhatClusterPoints[yhatClust][:,:-1]
                for l in range(0, len(points)):
                    point = points[l]
                    dist = np.linalg.norm(centroids[c] - point)
                    # compare dist and point for each radii of each origial centroid center
                    for t in thresholds:
                        if dist > thresholds[t]:
                            indx_NewMC.append(l)
                            to_newMC.append(point)
                        elif dist <= thresholds[t]:
                            indx_Append.append(l)
                            to_append.append(point)
            to_append = np.array(to_append)   
            indx_NewMC = np.array(indx_NewMC)
            indx_Append = np.array(indx_Append)
            indx_toAppend[c] = np.unique(indx_Append)
            indx_toNewMC[c] = np.unique(indx_NewMC)
            addToMC[c] = to_append
        to_newMC = np.array(to_newMC) 
        # add new cluster points to associated MCs
        for mc in currentClusterPoints:
            yhatIndx = indx_toAppend[mc]
            currentClusterPoints[mc] = np.vstack((currentClusterPoints[mc], yhatClusterPoints[mc][yhatIndx])) 

        newMC = []
        for p in yhatClusterPoints:
            newMC.append(yhatClusterPoints[p][indx_toNewMC[p]])
        newMC = np.array(newMC)
        newMC = newMC.reshape(-1,np.shape(currentClusterPoints[0])[1])
        # update centroid 
        mcluster = self._updateCentroid(inCurrentClusters= currentClusterPoints, inNewClusters= newMC, x = data, y= preds)

    def _updateCentroid(self, inCurrentClusters, inNewClusters, x, y):
        cluster_centroids = {}
        cluster_radii = {}
        length = max(inCurrentClusters.keys()) + 1 
        inCurrentClusters[length] = inCurrentClusters
        #TODO: create centroid for new mc ???
        


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
        """
        timesteps = self.X.keys()

        for ts in range(0, len(timesteps) - 1):
            total_start = time.time()
            if ts == 0:
                self.microCluster[ts] = self._cluster(X= self.X, y= self.T, ts=ts )
                t_start = time.time()
                # classify based on the clustered predictions (self.preds) done in the init step
                self.preds[ts] = self._classify(trainData=self.T, trainLabel= self.clusters[ts], testData=self.X[ts])
                t_end = time.time()
                perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.preds[ts], test= self.X[ts][:,-1], \
                                                dataset= self.dataset , method= self.method , \
                                                classifier= self.classifier, tstart=t_start, tend=t_end)
                self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.preds[ts], test= self.X[ts+1][:,-1])
            # determine if added x_t to MC exceeds radii of MC
            else:
                self.microCluster[ts] = self._cluster(X= self.X, y= self.X[ts-1], ts=ts )
                t_start = time.time()
                # classify based on the clustered predictions (self.preds) done in the init step
                self.preds[ts] = self._classify(trainData=self.X[ts], trainLabel=self.clusters[ts], testData=self.X[ts+1])
                t_end = time.time()
                perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.preds[ts], test= self.X[ts+1][:,-1], \
                                                dataset= self.dataset , method= self.method , \
                                                classifier= self.classifier, tstart=t_start, tend=t_end)
                self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.preds[ts], test= self.X[ts+1][:,-1])
                #TODO: what to do with this:
                # self._append_mcluster(yhat = self.preds[ts], inData= self.X[ts], inMicroCluster= self.microCluster[ts], ts= ts)

            total_end = time.time()
        self.total_time = total_start - total_end
        avg_metrics = cp.PerformanceMetrics(tstart= total_start, tend= total_end)
        self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
        print(self.avg_perf_metric)
# test mclass
run_mclass = MClassification(classifier='svm', method = 'kmeans', dataset='UG_2C_2D').run()

#%%
