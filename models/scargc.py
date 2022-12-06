#!/usr/bin/env python 
"""
Application:        SCARGC
File name:          scargc.py
Author:             Martin Manuel Lopez
Creation:           05/30/2021

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

from cProfile import label
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


from multiprocessing import pool
import statistics
from turtle import position
import numpy as np 
from scipy import stats
from sklearn import preprocessing
from sklearn.svm import SVC, SVR
from tqdm import tqdm
import math
import benchmark_datagen as bdg
import unsw_nb15_datagen as unsw
import classifier_performance as cp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import time
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from knn import knn as Bknn


class SetData:
    def __init__(self, dataset):
        self.dataset = dataset
        self._initialize()

    def _initialize(self):
        set_data = bdg.Datagen()
        data_gen = set_data.gen_dataset(self.dataset)
        data ={}
        labeled = {}
        unlabeled = {}
        ts = 0

        # set a self.data dictionary for each time step 
        # self.dataset[0][i] loop the arrays and append them to dictionary
        # data is the datastream 
        for i in range(0, len(data_gen[0])):
            data[ts] = data_gen[0][i]
            ts += 1

        # filter out labeled and unlabeled from of each timestep
        for i in data:
            len_of_batch = len(data[i])
            label_batch = []
            unlabeled_batch = []            
            for j in range(0, len_of_batch - 1):
                if data[i][j][2] == 1:              # will want to say that label == 1
                    label_batch.append(data[i][j])
                    labeled[i] = label_batch
                else:
                    unlabeled_batch.append(data[i][j])
                    unlabeled[i] = unlabeled_batch

        # convert labeled data to match self.data data structure
        labeled_keys = labeled.keys()
        for key in labeled_keys:        
            if len(labeled[key]) > 1:
                len_of_components = len(labeled[key])
                array_tuple = []
                for j in range(0, len_of_components):
                    array = np.array(labeled[key][j])
                    arr_to_list = array.tolist()
                    array_tuple.append(arr_to_list)
                    array = []
                    arr_to_list = []
                concat_tuple = np.vstack(array_tuple)
                labeled[key] = concat_tuple
        
        self.X = labeled    # set of all labels as a dict per timestep ; we only need X[0] for initial labels
        self.Y = data       # data stream

class SCARGC: 
    def __init__(self, 
                Xinit,
                Yinit,
                Kclusters:int=10,
                maxpool:int=25, 
                resample:bool=True, 
                T:int=100,
                classifier:str='',
                dataset = []): 
        """
        """
        # set the classifier that is used [eg 1nn or svm]
        self.classifier = classifier 
        # set the number of clusters for kmeans
        self.Kclusters = Kclusters
        # get the number of classes in the dataset 
        self.nclasses = len(np.unique(Yinit))
        # this will associate a cluster to a class in the data 
        # self.class_cluster = np.zeros((self.Kclusters,))
        self.class_cluster = {}
        # set the data 
        self.X = Xinit
        self.Y = Yinit
        # set resample 
        self.resample = resample
        # set max pool size 
        self.maxpool = maxpool
        # initialize the cluster model
        self._initialize()
        self.T = 0
        self.dataset = dataset
        self.performance_metric = {}
        self.avg_perf_metric = {}
        self.preds = {}

    def _initialize(self): 
        """
        """
        # run the clustering algorithm on the training data then find the cluster 
        # assignment for each of the samples in the training data         
        if self.classifier == '1nn':
            self.cluster = KMeans(n_clusters=self.Kclusters).fit(self.X[0])
            labels = self.cluster.predict(self.X[1])
            
            # for each of the clusters, find the labels of the data samples in the clusters
            # then look at the labels from the initially labeled data that are in the same
            # cluster. assign the cluster the label of the most frequent class. 
            for i in range(self.Kclusters):
                yhat = self.Y[i][labels]
                mode_val,_ = stats.mode(yhat)
                self.class_cluster[i] = mode_val
        elif self.classifier == 'svm':
            self.cluster = KMeans(n_clusters=self.Kclusters).fit(self.X)
            labels = self.cluster.predict(self.X)
            
            # for each of the clusters, find the labels of the data samples in the clusters
            # then look at the labels from the initially labeled data that are in the same
            # cluster. assign the cluster the label of the most frequent class. 
            for i in range(self.Kclusters):
                yhat = self.Y[i][labels]
                mode_val,_ = stats.mode(yhat)
                self.class_cluster[i] = mode_val

    def run(self, Xts, Yts): 
        '''
        Xts = Initial Training data
        Yts = Data stream
        '''
        total_time_start = time.time()
        # Build Classifier 
        if self.classifier == '1nn':
            if len(Yts[0]) < len(Xts[0]):
                dif = int(len(Xts[0]) - len(Yts[0]))
                xts_array = list(Xts[0])
                for i in range(dif):
                    xts_array.pop()
                Xts = np.array(xts_array)
            elif len(Yts[0]) > len(Xts[0]):
                dif = int(len(Yts[0]) - len(Xts[0]))
                xts_array = list(Yts[0])
                for i in range(dif):
                    xts_array.pop()
                Yts[0] = np.array(xts_array)
            knn = KNeighborsRegressor(n_neighbors=1).fit(Yts[0], Xts[0])           # KNN.fit(train_data, train label)
            predicted_label = knn.predict(Yts[1])

            # brute knn
            # bknn = Bknn(k=1, problem=1, metric=0)
            # bknn.fit(Yts[0], Xts[0])
            # predicted_label = bknn.predict(Yts[1])

        elif self.classifier == 'svm':
            if len(Yts[0]) < len(Xts):
                dif = int(len(Xts) - len(Yts[0]))
                xts_array = list(Xts)
                for i in range(dif):
                    xts_array.pop()
                Xts = np.array(xts_array)
            elif len(Yts[0]) > len(Xts):
                dif = int(len(Yts[0]) - len(Xts[0]))
                xts_array = list(Yts[0])
                for i in range(dif):
                    xts_array.pop()
                Yts[0] = np.array(xts_array)
            
            svn_clf = SVC(gamma='auto').fit(Xts[0][:,:-1], Yts[0][:,-1])
            predicted_label = svn_clf.predict(Yts[1][:,:-1])
            self.preds[0] = predicted_label
            
        self.T = len(Yts)      

        # empty sets for pool and labels
        pool_data = []
        pool_label = []
        pool_index = 0
        past_centroid = self.cluster.cluster_centers_


        labeled_data_labels = Xts
        labeled_data = Yts
        
        # run the experiment 
        for t in tqdm(range(self.T-1), position=0, leave=True): 
            # get the data from time T and resample if required
            # it seems that the algo takes in the labeled data labels and the labeled data as inputs 
            if self.classifier == '1nn':
                if t == 0: 
                    Xt, Yt = np.array(labeled_data_labels[t]), np.array(labeled_data[t])       # Xt = train labels ; Yt = train data
                    Xe, Ye = np.array(labeled_data_labels[t+1]), np.array(Yts[t+1])            # Xe = test labels ; Ye = test data
                else: 
                    Xt, Yt = np.array(labeled_data_labels), np.array(labeled_data)             # Xt = train labels ; Yt = train data
                    Xe, Ye = np.array(labeled_data_labels), np.array(Yts[t+1])                 # Xe = test labels ; Ye = test data
            elif self.classifier == 'svm': 
                if t == 0:
                    Xt, Yt = np.array(labeled_data_labels[t]), np.array(Yts[t])                   # Xt = train labels ; Yt = train data
                    Xe, Ye = np.array(Xts), np.array(Yts[t+1])                                 # Xe = test labels ; Ye = test data
                else:
                    Xt, Yt = np.array(labeled_data_labels), np.array(labeled_data)             # Xt = train labels ; Yt = train data
                    Xe, Ye = np.array(labeled_data_labels), np.array(Yts[t+1])                 # Xe = test labels ; Ye = test data
    
            
            t_start = time.time()
        
            if self.resample:
                N = len(Yt)
                V = len(Xt)
                ii = np.random.randint(0, N, N)
                jj = np.random.randint(0, V,V)
                Xt = Xt[jj] 
                Yt =  Yt[ii]

            if t == 0:
                pool_data = Ye
                pool_label = np.array(predicted_label)
                pool_index += 1
            else:
                if self.classifier == '1nn':
                    if len(Yt) > len(Xt):
                        dif = len(Yt) - len(Xt)
                        yt_reduced = list(Yt)
                        for q in range(dif):
                            yt_reduced.pop()
                        Yt = np.array(yt_reduced)

                    knn_mdl = KNeighborsRegressor(n_neighbors=1).fit(Yt, Xt)    # fit(train_data, train_label)
                    predicted_label = knn_mdl.predict(Ye)

                    # bknn_mdl = Bknn(k=0, problem=1, metric=0)
                    # bknn_mdl.fit(Yt, Xt)
                    # predicted_label = bknn_mdl.predict(Ye)

                elif self.classifier == 'svm':
                    svm_mdl = SVC(gamma='auto').fit(Yt[:,:-1], Yt[:,-1])        # fit(Xtrain, X_label_train)
                    predicted_label = svm_mdl.predict(Ye[:,:-1])
                
                pool_data = np.vstack((pool_data, Ye))

                # remove extra dimensions from pool label
                pool_label = np.squeeze(pool_label)
                predicted_label = np.squeeze(predicted_label)
                self.preds[t] = predicted_label
                
                if self.classifier == '1nn':
                    pool_label = np.concatenate((pool_label, predicted_label))
                elif self.classifier == 'svm':
                    pool_label = np.concatenate((pool_label, predicted_label))
                
                if t > 0:
                    sbrt_pool_lbl = list(pool_label)
                    sbrt_pool_lbl.pop(0)
                pool_label = np.array(pool_label)
                
                pool_index += 1
            concordant_label_count = 0

            # if |pool| == maxpoolsize
            if len(pool_label) > self.maxpool:
                # C <- Clustering(pool, k)
                temp_current_centroids = KMeans(n_clusters=self.Kclusters, init=past_centroid, n_init=1).fit(pool_data).cluster_centers_
                # find the label for the current centroids               
                # new labeled data
                new_label_data = np.zeros(np.shape(temp_current_centroids)[1])
                for k in range(self.Kclusters):
                    if self.classifier == '1nn':
                        nearestData = KNeighborsRegressor(n_neighbors=1).fit(past_centroid, temp_current_centroids)
                        centroid_label = nearestData.predict([temp_current_centroids[k]])[0]
                        
                        new_label_data = np.vstack((new_label_data, centroid_label))
                        
                        # _,new_label_data = nearestData.kneighbors([temp_current_centroids[k]])
                        # new_label_data = np.vstack(new_label_data[0][0])
                        # nearestData = Bknn(k=0, problem=1, metric=0)
                        # nearestData.fit(past_centroid, temp_current_centroids)
                        # centroid_label = nearestData.predict(temp_current_centroids[k])[0]
                        # new_label_data = np.vstack((new_label_data[0], centroid_label))
                        
                        
                    elif self.classifier == 'svm':
                        nearestData = SVR(gamma='auto').fit(past_centroid[:,:-1], temp_current_centroids[:,-1])
                        centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                        new_label_data = np.vstack(centroid_label)
                
                new_label_data = list(new_label_data)
                new_label_data.pop(0)
                new_label_data = np.array(new_label_data)
                
                # concordant data 
                for l in range(0, len(pool_data)):
                    if pool_data[l][-1] == 1:
                        concordant_label_count += 1
                
                if concordant_label_count != 1:
                    labeled_data = pool_data
                    labeled_data_labels = new_label_data
                    past_centroid = temp_current_centroids
                
                    
                # get prediction score 
                if self.classifier == '1nn': 
                    Ye = np.array(Ye[:,-1])
                if self.classifier == 'svm': 
                    Ye = np.array(Ye[:,-1])
                
                # if self.classifier == '1nn':
                #     if t>0:
                #         predicted_label = predicted_label[:,-1]
                
                # reset 
                pool_data = np.zeros(np.shape(pool_data)[1])
                pool_label = np.zeros(np.shape(pool_data))
                pool_index = 0    
            
            t_end = time.time() 
            perf_metric = cp.PerformanceMetrics(timestep= t, preds= self.preds[t], test= Ye, \
                                                dataset= self.dataset , method= '' , \
                                                classifier= self.classifier, tstart=t_start, tend=t_end)
            self.performance_metric[t] = perf_metric.findClassifierMetrics(preds= self.preds[t], test= Ye)

        total_time_end = time.time()

        self.total_time = total_time_end - total_time_start
        avg_metrics = cp.PerformanceMetrics(tstart= total_time_start, tend= total_time_end)
        self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
        return self.avg_perf_metric

# scargc_svm_data = SetData(dataset= 'UG_2C_2D')
# run_scargc_svm = SCARGC(Xinit= scargc_svm_data.X[0], Yinit= scargc_svm_data.Y , classifier = 'svm', dataset= 'UG_2C_2D')
# results = run_scargc_svm.run(Xts = scargc_svm_data.X, Yts = scargc_svm_data.Y)
# print(results)
