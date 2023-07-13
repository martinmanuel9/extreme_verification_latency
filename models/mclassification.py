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
from tqdm import tqdm
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
                datasource,
                graph = True): 
        """
        """
        self.classifier = classifier
        self.dataset = dataset
        self.datasource = datasource
        self.NClusters = 0
        self.method = method
        self.cluster_centers ={}
        self.graph = graph
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
        inPoints = x[:,:-1] 
        # Find the MC that the point is closest to
        distances = np.linalg.norm(inPoints[:, np.newaxis, :] - MC_Centers[:,:-1], axis=2)
        points_with_distances = np.column_stack((inPoints, distances))
        pointMCDistance = {}
        for i, dp in enumerate(inPoints):
            pointMCDistance[tuple(dp)] = distances[i].tolist()
        
        minDistIndex = []
        for i, dp in enumerate(inPoints):
            minDistIndex.append(pointMCDistance[tuple(dp)].index(min(pointMCDistance[tuple(dp)]))) 

        minDistMC = {}
        minDistMC['Xt'] = x
        minDistMC['Points'] = inPoints
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
                kmeans_model = KMeans(n_clusters=c,  n_init='auto').fit(X)
                score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
                sil_score[c] = score
            optimal_cluster = max(sil_score, key=sil_score.get)
            self.NClusters = optimal_cluster

    def cluster(self, X, y, ts):
        if self.method == 'kmeans':
            if ts == 0:
                self.find_silhoette_score(X=X[ts], y=y, ts=ts)
                kmeans_model = KMeans(n_clusters=self.NClusters, n_init='auto').fit(X[ts])
            else:
                kmeans_model = KMeans(n_clusters=self.NClusters, n_init='auto').fit(X[ts]) #may not need to do this as we need to create a new cluster for the new data
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
        
        # TODO: Not sure what this stat is used for???
        # for i in range(self.NClusters):
        #     xhat = self.X[i][self.clusters[ts]]
        #     mode_val,_ = stats.mode(xhat)
        #     self.class_cluster[i] = mode_val

    def create_mclusters(self, inClusterpoints, inData,  threshold) :
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
        Xt = inData
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
        radii = np.sqrt((SS / N) - ((LS / N)**2 ))
        # mcluster['Radii'] = threshold
        distance = np.sqrt((radii[0] - mcluster['Centroid'][0])**2 + (radii[1] - mcluster['Centroid'][1])**2)
        mcluster['Radii'] = distance  
        mcluster['Threshold'] = distance # threshold was previous
        mcluster['Xt'] = Xt
        mcluster['Disjoint'] = False

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

            ## gets the indices of each cluster 
            cluster_indices = {}
            for i in range(self.NClusters):
                cluster_indices[i] = np.array([ j for j , x in enumerate(inCluster.labels_) if x == i])
            
            ## creates cluster data
            mcluster = {}
            ## calculates the microcluster for each cluster based on the number of classes 
            for c in range(self.NClusters):
                mcluster[c] = self.create_mclusters(inClusterpoints= x[cluster_indices[c]][:,: np.shape(x)[1]-1], inData= x[cluster_indices[c]], threshold=cluster_radii[c]) 
            if self.graph: 
                self.graphMClusters(inCluster= inCluster, fitCluster= fitCluster, x= x)

            return mcluster
        
        elif self.method == 'gmm':
            pass # need to determine how to calc radii for gmm 

    def graphMClusters(self, inCluster, fitCluster, x):
        cluster_centroids = {}
        cluster_radii = {}
        ## calculates the cluster centroid and the radii of each cluster
        if self.method == 'kmeans':
            for cluster in range(self.NClusters):
                cluster_centroids[cluster] = list(zip(inCluster.cluster_centers_[:,0], inCluster.cluster_centers_[:,1]))[cluster]
                cluster_radii[cluster] = max([np.linalg.norm(np.subtract(i, cluster_centroids[cluster])) for i in zip(x[fitCluster == cluster, 0], x[fitCluster == cluster, 1])])
        ## plot clusters
        fig, ax = plt.subplots(1,figsize=(7,5))
        ## plot clusters
        for i in range(self.NClusters):
            plt.scatter(x[fitCluster == i, 0], x[fitCluster == i, 1], s = 100, color = np.random.rand(3,), label ='Cluster '+ str(i))
            art = mpatches.Circle(cluster_centroids[i],cluster_radii[i], edgecolor='b', fill=False)
            ax.add_patch(art)

        # Plotting the centroids of the clusters
        plt.scatter(inCluster.cluster_centers_[:, 0], inCluster.cluster_centers_[:,1], s = 100, color = 'yellow', label = 'Centroids')
        plt.legend()
        plt.tight_layout()
        plt.show()


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
        
        # gets all the data points and groups them by the option in parallel using zip
        group_data = {}
        for data, option in zip(inDict["Xt"], inDict["MinDistIndex"]):
            if option not in group_data:
                group_data[option] = []
            group_data[option].append(data)
        
        # assert len(option_arrays.keys()) == np.shape(self.cluster_centers[ts-1])[0], 'Past cluster centers should be the same as grouped arrays.'
        sortedArray = sorted(option_arrays.keys())
        groupedPointsMC = {}
        for key in sortedArray:
            groupedPointsMC[key] = option_arrays[key]

        sortedData = sorted(group_data.keys())
        groupedDataMC = {}
        for key in sortedData:
            groupedDataMC[key] = group_data[key]

        return groupedPointsMC, groupedDataMC
    
    def pointInCluster(self, target_point, cluster_data, radius):
        for point in cluster_data:
            # euclidean distance if the point in stream should belong to prevoius cluster
            distance = np.sqrt((target_point[0] - point[0]) ** 2 + (target_point[1] - point[1]) ** 2)
            if distance <= radius:
                return True
        return False
    
    def evaluateXt(self, inPreGroupedPoints, prevMC, inPreGroupedXt):
        """
        inPreGroupedXt = pre grouped xt 
        inCluster = cluster 
        """
        tempPointsAddMC = {}
        tempPointsNewMC = {}
        tempXtAddMC = {}
        tempXtNewMC = {}
        
        for i in inPreGroupedPoints:
            dataPoints = inPreGroupedPoints[i]
            xtPoints = inPreGroupedXt[i]
            assert len(dataPoints) == len(xtPoints), 'Length of points and xt needs to be the same'
            inCluster = False
            radius = prevMC[i]['Radii']
            clusterPoints = prevMC[i]['ClusterPoints']
            for point in range(0, len(dataPoints)):
                inCluster = False
                # Resets if in Cluster for each point
                if self.pointInCluster(dataPoints[point], clusterPoints, radius):
                    inCluster = True
                    if i in tempPointsAddMC:
                        tempPointsAddMC[i].append(dataPoints[point])
                        tempXtAddMC[i].append(xtPoints[point])
                    else:
                        tempPointsAddMC[i] = [dataPoints[point]]
                        tempXtAddMC[i] = [xtPoints[point]]
                else:
                    inCluster = False
                    if i in tempPointsNewMC:
                        tempPointsNewMC[i].append(dataPoints[point])
                        tempXtNewMC[i].append(xtPoints[point])
                    else:
                        tempPointsNewMC[i] = [dataPoints[point]]
                        tempXtNewMC[i] = [xtPoints[point]]
        
        return tempPointsAddMC, tempPointsNewMC, tempXtAddMC, tempXtNewMC
    
    def updateMCluster(self, inMCluster, addToMC, inXtAddMC, ts): 
        mcluster = {}
        for mc in addToMC:
            pastClusterPoints = inMCluster[mc]['ClusterPoints']
            toAddClusterPoints = np.squeeze(addToMC[mc])
            newClusterPoints = np.vstack((pastClusterPoints, toAddClusterPoints))
            newXt = np.vstack((inMCluster[mc]['Xt'], inXtAddMC[mc]))
            mcluster[mc] = self.create_mclusters(inClusterpoints= newClusterPoints, inData= newXt,  threshold=inMCluster[mc]['Threshold'])
   
        # the new MC at ts will be the incremented statistic where the new points are added to the existing MC 
        self.microCluster[ts] = mcluster
 
    def updateModelMC(self, inData,  ts):
        ## This assuming that we have the previous model 
        if self.method == 'kmeans':
            updatedModel = KMeans(n_clusters=self.NClusters, n_init='auto').fit(inData)
            predictedLabels = updatedModel.fit_predict(inData)
        elif self.method == 'gmm':
            updatedModel = GMM(n_components=self.NClusters).fit(inData)
            predictedLabels = updatedModel.fit_predict(inData)
        
        
        updatedMicroCluster = self.create_centroid(inCluster= updatedModel, fitCluster= updatedModel.fit_predict(inData), x=inData, y = predictedLabels) 
        if self.graph:
            self.graphMClusters(inCluster= updatedModel, fitCluster= predictedLabels, x= inData)

        updatedClusters = updatedModel.predict(inData)
        updatedClusterCenters = updatedModel.cluster_centers_
        return updatedMicroCluster, updatedClusters, updatedClusterCenters
      

    def initLabelData(self, ts, inData, inLabels):
        self.cluster(X= inData, y= inLabels, ts=ts )
        t_start = time.time()
        # classify based on the clustered predictions (self.preds) done in the init step
        self.preds[ts] = self.classify(trainData= inData[ts] , trainLabel= inLabels[:,-1], testData=self.X[ts+1])
        t_end = time.time()
        perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.preds[ts], test= self.X[ts][:,-1], \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end)
        self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.preds[ts], test= self.X[ts+1][:,-1])

    def findDisjointMCs(self, inMCluster):
        disjoint_sets = []
        keys = list(inMCluster.keys())
        num_keys = len(keys)

        for i in range(num_keys):
            for j in range(i + 1, num_keys):
                set_i = set(tuple(element) for element in inMCluster[keys[i]]['Xt'])
                set_j = set(tuple(element) for element in inMCluster[keys[j]]['Xt'])
                
                # Check for common elements
                if set_i.intersection(set_j):
                    continue  # Not disjoint
                else:
                    disjoint_sets.append((keys[i], keys[j]))

        return np.array(disjoint_sets)
    
    def additivityMC(self, disjointMC, inMCluster, ts): 
        for key in disjointMC:
            newPoints = np.vstack((inMCluster[key[0]]['ClusterPoints'], inMCluster[key[1]]['ClusterPoints']))
            newXt = np.vstack((inMCluster[key[0]]['Xt'], inMCluster[key[1]]['Xt']))
            newN = inMCluster[key[0]]['N'] + inMCluster[key[1]]['N']
            newLS = inMCluster[key[0]]['LS'] + inMCluster[key[1]]['LS']
            newSS = inMCluster[key[0]]['SS'] + inMCluster[key[1]]['SS']
            newRaddiPoints = np.sqrt((newSS / newN) - ((newLS / newN)**2 ))
            newCentroid = newLS / newN
            distance = np.sqrt((newRaddiPoints[0] - newCentroid[0])**2 + (newRaddiPoints[1] - newCentroid[1])**2)

            self.microCluster[ts][key[0]]['ClusterPoints'] = newPoints
            self.microCluster[ts][key[0]]['N'] = newN
            self.microCluster[ts][key[0]]['LS'] = newLS
            self.microCluster[ts][key[0]]['SS'] = newSS
            self.microCluster[ts][key[0]]['Centroid'] = newCentroid
            self.microCluster[ts][key[0]]['Radii'] = distance 
            self.microCluster[ts][key[0]]['Threshold'] = distance
            self.microCluster[ts][key[0]]['Xt'] = newXt
            self.microCluster[ts][key[0]]['Disjoint'] = True
            self.microCluster[ts][key[0]]['DisjointPair'] = key
    
        toDeleteMC = [key for key, value in self.microCluster[ts].items() if value['Disjoint'] == False] 

        for mc in toDeleteMC:
            del self.microCluster[ts][mc]
        # To get only the joined MCs 
        joinedMC = [key for key, value in self.microCluster[ts].items() if value['Disjoint'] == True]
        filteredXt = np.vstack([self.microCluster[ts][key]['Xt'] for key in joinedMC])
        
        # # remove non-unique values from the clusters
        uniqueFilteredXt =  np.unique(filteredXt, axis=0)
        # update based on joined clusters
       
        self.microCluster[ts], self.clusters[ts], self.cluster_centers[ts] = self.updateModelMC(inData= uniqueFilteredXt, ts= ts)
        
        for mc in self.microCluster[ts]:
            self.microCluster[ts][mc]['Step'] = 'Additivity' 
            self.microCluster[ts][mc]['DisjointPairs'] = disjointMC
    
    def drop_non_unique(self, *dictionaries):
        unique_values = set()
        duplicates = set()

        # Iterate over each dictionary
        for dictionary in dictionaries:
            for value in dictionary.values():
                if isinstance(value, np.ndarray):
                    # Convert numpy array to tuple of tuples
                    value = tuple(map(tuple, value))
                if value in unique_values:
                    duplicates.add(value)
                else:
                    unique_values.add(value)

        # Create a new dictionary with unique values
        unique_dict = {}
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    value = tuple(map(tuple, value))
                if value not in duplicates:
                    unique_dict[key] = value

        
         # Convert tuples back into numpy arrays
        for key, value in unique_dict.items():
            if isinstance(value, tuple):
                unique_dict[key] = np.array(value)

        return unique_dict

    def run(self):
        """
        Micro-Cluster Classification
        1. The Algo takes an initial set of labeled data T and builds a set of labeled MCs (this is the first labeled data) -- complete 
        2. The classification phase we predict yhat for each example xt from the stream 
        3. The classification is based on the nearest MC according to the Euclidean Distance 
        4. We determine if xt from the stream corresponds to the nearest MC using the incrementality property and then we would 
            need to update the statistic of that MC if it does NOT exceed the radius (that is predefined) 
        5. If the radius exceeds the threshold, a new MC carrying the predicted label is created to allocate the new example 
        6. The algo must search the two farthest MCs from the predicted class to merge them by using the additivity property. 
        The two farthest MCs from xt are merged into one MC that will be placed closest to the emerging new concept. 
        """
        total_start = time.time()
        timesteps = self.X.keys()
        for ts in tqdm(range(len(timesteps) - 1), position=0, leave=True):
            # This takes the fist labeled data set T and creates the initial MCs
            if ts == 0:
                # Classify first labeled dataset T
                self.initLabelData(inData= self.X, inLabels= self.T, ts=ts)
            # determine if added x_t to MC exceeds radii of MC
            else:
                # Step 2 begin classification of next stream to determine yhat 
                t_start = time.time()
                # classify based on the clustered predictions (self.preds) done in the init step 
                # self.clusters is the previous preds
                closestMC = self.findClosestMC(x= self.X[ts], MC_Centers= self.cluster_centers[ts-1])
                preGroupedPoints, preGroupedXt = self.preGroupMC(inDict= closestMC, ts= ts)
                pointsAddToMC, pointsNewMC, xtAddToMC, xtNewMC = self.evaluateXt(inPreGroupedPoints= preGroupedPoints, 
                                                                                 prevMC= self.microCluster[ts-1], inPreGroupedXt= preGroupedXt)
                
                
                # we first check if we first need to create a new cluster based on streaming data
                if len(xtNewMC) > 0:
                    self.NClusters = self.NClusters + len(xtNewMC) 
                    newMCData = np.vstack([xtNewMC[mc] for mc in xtNewMC.keys()])
                    # inData = np.vstack((self.X[ts-1], newMCData))
                    
                    if len(newMCData) >= self.NClusters:
                        self.microCluster[ts], self.clusters[ts], self.cluster_centers[ts] = self.updateModelMC(inData= newMCData , ts= ts)
                    else:
                        self.updateMCluster(inMCluster=self.microCluster[ts-1], inXtAddMC= xtAddToMC, addToMC=pointsAddToMC, ts= ts)

                if not xtNewMC: 
                    self.updateMCluster(inMCluster=self.microCluster[ts-1], inXtAddMC= xtAddToMC, addToMC=pointsAddToMC, ts= ts)
    
                # remove non-unique values from the clusters
                uniqueDict = {}
                for mc in self.microCluster[ts]:
                    uniqueDict[mc] = self.microCluster[ts][mc]['Xt']
                
                uniqueData = self.drop_non_unique(uniqueDict)
                inData = np.vstack([value for value in uniqueData.values()])
                
                # update model after incrementing MCs
                self.microCluster[ts], self.clusters[ts], self.cluster_centers[ts] = self.updateModelMC(inData= inData, ts= ts)

                for mc in self.microCluster[ts]:
                    self.microCluster[ts][mc]['Step'] = 'Incrementality' 
                
                ## The additivity property 
                '''
                The additivity considers that if we have two disjoint Micro-Clusters MCA and MCB,  
                the  union  of  these  two  groups  is equal to the sum of its parts
                '''
                # find the disjoint sets of the microclusters 
                disjointMC = self.findDisjointMCs(self.microCluster[ts])
                self.additivityMC(disjointMC= disjointMC, inMCluster= self.microCluster[ts], ts= ts)
                # inData = np.vstack([self.microCluster[ts][mc]['Xt'] for mc in self.microCluster[ts].keys()])

                ## remove non-unique values from the clusters
                uniqueDict = {}
                for mc in self.microCluster[ts]:
                    uniqueDict[mc] = self.microCluster[ts][mc]['Xt']
                
                uniqueData = self.drop_non_unique(uniqueDict)
                inData = np.vstack([value for value in uniqueData.values()])

                self.preds[ts] = self.classify(trainData= inData, trainLabel=inData[:,-1], testData=self.X[ts+1])
                t_end = time.time()
                perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.preds[ts], test= self.X[ts+1][:,-1], \
                                                dataset= self.dataset , method= self.method , \
                                                classifier= self.classifier, tstart=t_start, tend=t_end)
                self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.preds[ts], test= self.X[ts+1][:,-1])
              
        total_end = time.time()
        self.total_time = total_end - total_start
        avg_metrics = cp.PerformanceMetrics(tstart= total_start, tend= total_end)
        self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
        return self.avg_perf_metric

# test mclass
run_mclass = MClassification(classifier='knn', method = 'kmeans', dataset='UG_2C_2D', datasource='Synthetic', graph=False).run()
print(run_mclass)
#%%
