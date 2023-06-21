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
import unsw_nb15_datagen as unsw
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import classifier_performance as cp
from scipy import stats
from sklearn.metrics import silhouette_score
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
                datasource): 
        """
        """
        self.classifier = classifier
        self.dataset = dataset
        self.datasource = datasource
        self.NClusters = 0
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
        self.setData()
        
    def setData(self):
        data_gen = cbdg.Synthetic_Datagen()
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

    def findClosestMC(self, x, MC_Centers):
        """
        x = datastream point 
        MC = microcluster
        """
        inData = x[:,:-1]
        # Find the MC that the point is closest to
        distances = np.linalg.norm(inData[:, np.newaxis, :] - MC_Centers[:,:-1], axis=2)
        points_with_distances = np.column_stack((inData, distances))
        pointMCDistance = {}
        for i, dp in enumerate(inData):
            pointMCDistance[tuple(dp)] = distances[i].tolist()
        
        minDistIndex = []
        for i, dp in enumerate(inData):
            minDistIndex.append(pointMCDistance[tuple(dp)].index(min(pointMCDistance[tuple(dp)]))) 

        minDistMC = {}
        minDistMC['Xt'] = x
        minDistMC['Points'] = inData
        minDistMC['Distances'] = distances
        minDistMC['MinDistIndex'] = minDistIndex   
        return minDistMC
    
    def find_silhoette_score(self, X, y, ts):
        """
        Find Silhoette Scores allows us to get the optimal number of clusters for the data
        """
        if self.method == 'kmeans':
            sil_score = {}
            for c in range(2, 11):
                kmeans_model = KMeans(n_clusters=c).fit(X)
                score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
                sil_score[c] = score
            optimal_cluster = max(sil_score, key=sil_score.get)
            self.NClusters = optimal_cluster

    def cluster(self, X, y, ts):
        if self.method == 'kmeans':
            if ts == 0:
                self.find_silhoette_score(X=X[ts], y=y, ts=ts)
                kmeans_model = KMeans(n_clusters=self.NClusters).fit(X[ts])
            else:
                kmeans_model = KMeans(n_clusters=self.NClusters).fit(X[ts]) #may not need to do this as we need to create a new cluster for the new data
            # computes cluster centers and radii of cluster for initial ts
            self.microCluster[ts] = self.create_centroid(inCluster = kmeans_model, fitCluster = kmeans_model.fit_predict(X[ts]), x= X[ts] , y= y)
            self.clusters[ts] = kmeans_model.predict(X[ts]) # gets the cluster labels for the data
            self.cluster_centers[ts] = kmeans_model.cluster_centers_

        elif self.method == 'gmm':
            gmm_model = GMM(n_components=self.NClusters)
            gmm_model.fit(y) 
            self.clusters[ts] = gmm_model.predict(self.Y[ts+1])
            self.cluster_centers[ts] = self.clusters[ts] 
        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class.
        for i in range(self.NClusters):
            xhat = self.X[i][self.clusters[ts]]
            mode_val,_ = stats.mode(xhat)
            self.class_cluster[i] = mode_val

    def create_mclusters(self, inClusterpoints, threshold) :
        """
        Clustering options:
        1. k-means
        MC is defined as a 4 tuple (N, LS, SS, y) where:
        N = number of data points in a cluster
        LS = linear sum of N data points 
        SS = square sum of data points 
        y = label for a set of data points
        """
        mcluster = {}
        N = len(inClusterpoints)
        LS = sum(inClusterpoints)
        SS = 0
        for point in inClusterpoints:
            SS += sum([element**2 for element in point])
        
        mcluster['ClusterPoints'] = inClusterpoints
        mcluster['N'] = N
        mcluster['LS'] = LS
        mcluster['SS'] = SS
        mcluster['Centroid'] = LS / N
        # mcluster['Radii'] = np.sqrt((SS / N) - ((LS / N)**2 ))
        mcluster['Radii'] = threshold
        mcluster['Threshold'] = threshold
        return mcluster
        
    def classify(self, trainData, trainLabel, testData):
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

    def create_centroid(self, inCluster, fitCluster, x, y):
        """
        inCluster = cluster model 
        fitCluster = fitted model
        x = datastream
        y = label
        """
        cluster_centroids = {}
        cluster_radii = {}
        ## calculates the cluster centroid and the radii of each cluster
        if self.method == 'kmeans':
            for cluster in range(self.NClusters):
                cluster_centroids[cluster] = list(zip(inCluster.cluster_centers_[:,0], inCluster.cluster_centers_[:,1]))[cluster]
                cluster_radii[cluster] = max([np.linalg.norm(np.subtract(i, cluster_centroids[cluster])) for i in zip(x[fitCluster == cluster, 0], x[fitCluster == cluster, 1])])
            ## plot clusters
            # fig, ax = plt.subplots(1,figsize=(7,5))

            ## gets the indices of each cluster 
            cluster_indices = {}
            for i in range(self.NClusters):
                cluster_indices[i] = np.array([ j for j , x in enumerate(inCluster.labels_) if x == i])
            
            ## creates cluster data
            mcluster = {}

            ## calculates the microcluster for each cluster based on the number of classes 
            for c in range(self.NClusters):
                mcluster[c] = self.create_mclusters(inClusterpoints= x[cluster_indices[c]][:,: np.shape(x)[1]-1], threshold=cluster_radii[c]) 
            
            ## plot clusters
            # for i in range(self.NClusters):
            #     plt.scatter(x[fitCluster == i, 0], x[fitCluster == i, 1], s = 100, c = np.random.rand(3,), label ='Class '+ str(i))
            #     art = mpatches.Circle(cluster_centroids[i],cluster_radii[i], edgecolor='b', fill=False)
            #     ax.add_patch(art)

            ## Plotting the centroids of the clusters
            # plt.scatter(inCluster.cluster_centers_[:, 0], inCluster.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

            return mcluster
        
        elif self.method == 'gmm':
            pass # need to determine how to calc radii for gmm 

    def preGroupMC(self, inDict, ts):
        option_arrays = {}
        # Iterate over each point and option in parallel using zip
        for point, option in zip(inDict["Points"], inDict["MinDistIndex"]):
            # If the option doesn't exist as a key in the option_arrays dictionary,
            # create a new key with an empty list as its value
            if option not in option_arrays:
                option_arrays[option] = []
            # Append the current point to the list associated with the corresponding option key
            option_arrays[option].append(point)

        assert(len(option_arrays.keys()) == np.shape(self.cluster_centers[ts-1])[0])
        sortedArray = sorted(option_arrays.keys())
        groupedMC = {}
        for key in sortedArray:
            groupedMC[key] = option_arrays[key]

        return option_arrays
    
    def pointInCluster(self, target_point, cluster_data, radius):
        for point in cluster_data:
            # euclidean distance if the point in stream should belong to prevoius cluster
            distance = np.sqrt((target_point[0] - point[0]) ** 2 + (target_point[1] - point[1]) ** 2)
            if distance <= radius:
                return True
        return False
    
    def evaluateXt(self, inPreGroupedXt, prevMCluster):
        """
        inPreGroupedXt = pre grouped xt 
        inCluster = cluster 
        """
        tempAddMC = {}
        tempNewMC = {}
        for i in inPreGroupedXt:
            dataPoints = inPreGroupedXt[i]
            inCluster = False
            radius = prevMCluster[i]['Radii']
            clusterPoints = prevMCluster[i]['ClusterPoints']
            for point in dataPoints:
                inCluster = False
                # Resets if in Cluster for each point
                if self.pointInCluster(point, clusterPoints, radius):
                    inCluster = True
                    if i in tempAddMC:
                        tempAddMC[i].append(point)
                    else:
                        tempAddMC[i] = [point]
                else:
                    inCluster = False
                    if i in tempNewMC:
                        tempNewMC[i].append(point)
                    else:
                        tempNewMC[i] = [point]
        
        return tempAddMC, tempNewMC
    
    def updateMCluster(self, inMCluster, addToMC, ts):
        assert(len(inMCluster) == len(addToMC))
        mcluster = {}
        for mc in range(0, len(inMCluster)):
            pastClusterPoints = inMCluster[mc]['ClusterPoints']
            toAddClusterPoints = np.squeeze(addToMC[mc])
            newClusterPonts = np.concatenate((pastClusterPoints, toAddClusterPoints))
            mcluster[mc] = self.create_mclusters(inClusterpoints= newClusterPonts, threshold=inMCluster[mc]['Threshold'])
        # the new MC at ts will be the incremented statistic where the new points are added to the existing MCs
        self.microCluster[ts] = mcluster 


    def createNewMC(self, inData,  ts):
        ## This assuming that we have the previous model 
        if self.method == 'kmeans':
            updatedModel = KMeans(n_clusters=self.NClusters).fit(inData)
            predictedLabels = updatedModel.fit_predict(inData)
        elif self.method == 'gmm':
            updatedModel = GMM(n_components=self.NClusters).fit(inData)
            predictedLabels = updatedModel.fit_predict(inData)
        
        self.microCluster[ts] = self.create_centroid(inCluster= updatedModel, fitCluster= updatedModel.fit_predict(inData), x=inData, y = predictedLabels) 
        self.clusters[ts] = updatedModel.predict(inData)
        self.cluster_centers[ts] = updatedModel.cluster_centers_

    def initial_labeled_data(self, ts, inData, inLabels):
        self.cluster(X= inData, y= inLabels, ts=ts )
        t_start = time.time()
        # classify based on the clustered predictions (self.preds) done in the init step
        self.preds[ts] = self.classify(trainData=inData[ts] , trainLabel= inLabels[:,-1], testData=self.X[ts+1])
        t_end = time.time()
        perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.preds[ts], test= self.X[ts][:,-1], \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end)
        self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.preds[ts], test= self.X[ts+1][:,-1])

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
        ---------------
        1. The Algo takes an initial set of labeled data T and builds a set of labeled MCs (this is the first labeled data) -- complete 
        2. The classification phase we predict yhat for each example xt from the stream 
        3. The classification is based on the nearest MC according to the Euclidean Distance 
        4. We determine if xt from the stream corresponds to the nearest MC using the incrementality property and then we would 
        need to update the statistic of that MC if it does NOT exceed the radius (that is predefined) 
        5. If the radius exceeds the threshold, a new MC carrying the predicted label is created to allocate the new example 
        6. The algo must search the two farthest MCs from the predicted class to merge them by using the additivity property. 
        The two farthest MCs from xt are merged into one MC that will be placed closest to the emerging new concept. 
        """
        timesteps = self.X.keys()
        for ts in range(0, len(timesteps) - 1):
            total_start = time.time()
            # This takes the fist labeled data set T and creates the initial MCs
            if ts == 0:
                # Classify first labeled dataset T
                self.initial_labeled_data(inData= self.X, inLabels= self.T, ts=ts)
            # determine if added x_t to MC exceeds radii of MC
            else:
                # Step 2 begin classification of next stream to determine yhat 
                t_start = time.time()
                # classify based on the clustered predictions (self.preds) done in the init step 
                # self.clusters is the previous preds
                closestMC = self.findClosestMC(x= self.X[ts], MC_Centers= self.cluster_centers[ts-1])
                preGroupedXt = self.preGroupMC(inDict= closestMC, ts= ts)
                addToMC, newMC = self.evaluateXt(inPreGroupedXt= preGroupedXt, prevMCluster= self.microCluster[ts-1])
            
                ## TODO: Need to find the find the indexes of the addToMC 
                # 1. find the indexes of the addToMC
                # 2. need to create a set that gets 
                # what if I check if there are any new mcs and then do that then we add to microcluster
                
                
                
                ## fake newMC data
                newMC[0] = [[10.0, 10.0, 3],
                            [10.0, -10.0, 3],
                            [-10.0, 10.0, 3],
                            [-10.0, -10.0, 3],
                            [5.0, 0.0, 3],
                            [-5.0, 0.0, 3],
                            [0.0, 5.0, 3],
                            [0.0, -5.0, 3],
                            [10.0, 2.0, 3],
                            [10.0, -2.0, 3],
                            [-10.0, 2.0, 3],
                            [-10.0, -2.0, 3],
                            [2.0, 10.0, 3],
                            [2.0, -10.0, 3],
                            [8.0, 10.0, 3],
                            [8.0, -10.0, 3],
                            [5.0, 5.0, 3],
                            [-5.0, 5.0, 3],
                            [5.0, -5.0, 3],
                            [-5.0, -5.0, 3]]
                
                # we first check if we first need to create a new cluster based on streaming data
                if len(newMC) > 0:
                    # need to add previous time step stream data and current stream data to update model
                    self.NClusters = self.NClusters + len(newMC) 
                    ## test data added to create a fake cluster for testing
                    testData = np.concatenate((self.X[ts], newMC[0]))
                    self.createNewMC(inData= testData , ts= ts)
                
                # update microclusters based on streaming data to add to new MCs
                self.updateMCluster(self.microCluster[ts-1], addToMC=addToMC, ts= ts)

                ##TODO: Need to extract graphing capability as seperate function

                self.preds[ts] = self.classify(trainData=self.X[ts], trainLabel=self.clusters[ts-1], testData=self.X[ts+1])
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
        return self.avg_perf_metric

# test mclass
run_mclass = MClassification(classifier='knn', method = 'kmeans', dataset='UG_2C_2D', datasource='Synthetic').run()
# print(run_mclass)
#%%
