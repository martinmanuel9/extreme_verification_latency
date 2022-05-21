#!/usr/bin/env python 
"""
Application:        SCARGC
File name:          scargc.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021
SCARGC Origin:      Stream Classification Guided by Clustering (SCARGC) by
                    Vinicius M. A. Souza, Diego Silva, Joao Gama, Gustavo Batista

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""
# MIT License
#
# Copyright (c) 2021
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
import statistics
from turtle import position
import numpy as np 
from scipy import stats
from sklearn import preprocessing
from tqdm import tqdm
import math
import benchmark_datagen as bdg
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class SetData:
    def __init__(self, dataset = 'UG_2C_2D'):
        self.dataset = dataset
        self._initialize()

    def _initialize(self):
        set_data = bdg.Datagen()
        data_gen = set_data.gen_dataset(self.dataset)
        data ={}
        labeled = {}
        unlabeled = {}
        ts = 0

        ## set a self.data dictionary for each time step 
        ## self.dataset[0][i] loop the arrays and append them to dictionary
        for i in range(0, len(data_gen[0])):
            data[ts] = data_gen[0][i]
            ts += 1

        # filter out labeled and unlabeled from of each timestep
        for i in data:
            len_of_batch = len(data[i])
            label_batch = []
            unlabeled_batch = []            
            for j in range(0, len_of_batch - 1):
                if data[i][j][2] == 1:
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
        
        self.X = labeled    # set of all labels as a dict per timestep
        
        # X_temp = list(self.X.values())
        # self.X = np.array(X_temp) 
        # tempX = np.zeros(np.shape(self.X[0])[1])
        # for k in range(len(self.X)):
        #     tempX = np.vstack((tempX,self.X[k]))
        # tempX = list(tempX)
        # tempX.pop(0)
        # self.X = np.array(tempX)
        
        self.Y = data       # data stream
        
        # Y_temp = list(self.Y.values())
        # self.Y = np.array(Y_temp) 
        # tempY = np.zeros(np.shape(Y_temp[0])[1])
        # for k in range(len(self.Y)):
        #     tempY = np.vstack((tempY,self.Y[k]))
        # tempY = list(tempY)
        # tempY.pop(0)
        # self.Y = np.array(tempY)

class SCARGC: 
    def __init__(self, 
                Xinit,
                Yinit,
                Kclusters:int=10,
                maxpool:int=25, 
                resample:bool=True, 
                T:int=100,
                classifier:str='1nn'): 
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
        self.class_error = {}
        self.accuracy = {}

    def _initialize(self): 
        """
        """
        # run the clustering algorithm on the training data then find the cluster 
        # assignment for each of the samples in the training data         
        self.cluster = KMeans(n_clusters=self.Kclusters).fit(self.X)
        labels = self.cluster.predict(self.X)
        
        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class. 
        for i in range(self.Kclusters):
            yhat = self.Y[i][labels]
            mode_val,_ = stats.mode(yhat)
            self.class_cluster[i] = mode_val
        
    def classification_error(self, preds, L_test):  
        return np.sum(preds != L_test)/len(preds)

    def run(self, Xts, Yts): 
        '''
        Xts = Initial Training data
        Yts = Data stream
        '''
        # Build Classifier 
        if self.classifier == '1nn':
            if len(Xts[0]) < len(Xts[1]):
                dif = int(len(Xts[1]) - len(Xts[0]))
                xts_array = list(Xts[1])
                for i in range(dif):
                    xts_array.pop()
                Xts[1] = np.array(xts_array)
            elif len(Xts[0]) > len(Xts[1]):
                dif = int(len(Xts[0]) - len(Xts[1]))
                xts_array = list(Xts[0])
                for i in range(dif):
                    xts_array.pop()
                Xts[0] = np.array(xts_array)
            knn = KNeighborsRegressor(n_neighbors=1).fit(Xts[0], Xts[1]) # KNN.fit(train_data, train label)
            predicted_label = knn.predict(Yts[0])
            
        self.T = len(Yts)      

        # empty sets for pool and labels
        pool_data = []
        pool_label = []
        pool_index = 0
        past_centroid = self.cluster.cluster_centers_
        # run the experiment 
        for t in tqdm(range(self.T-1), position=0, leave=True): 
            # get the data from time T and resample if required
            Xt, Yt = np.array(Xts[t]), np.array(Yts[t])             # Xt = train labels ; Yt = train data
            Xe, Ye = Xts[t+1], Yts[t+1]                             # Xe = test labels ; Ye = test data
            
            if self.resample: 
                N = len(Xt)
                ii = np.random.randint(0, N, N)
                Xt, Yt = Xt[ii], Yt[ii]
            
            if t == 0:
                pool_data = Ye
                pool_label = np.array([predicted_label])
                pool_index += 1
            else:
                if self.classifier == '1nn':
                    knn_mdl = KNeighborsRegressor(n_neighbors=1).fit(Yt, Xt) # .fit(train_data, train_label)
                    predicted_label = knn_mdl.predict(Ye)
                pool_data = np.vstack([pool_data, Ye])
                pool_label = np.concatenate((pool_label, np.array([predicted_label])))
                pool_index += 1
            concordant_label_count = 0
            # if |pool| == maxpoolsize
            if len(pool_label) > self.maxpool:
                # C <- Clustering(pool, k)
                temp_current_centroids = KMeans(n_clusters=self.Kclusters, init=past_centroid, n_init=1).fit(pool_data).cluster_centers_
                # find the label for the current centroids                
                # new labeled data
                for k in range(self.Kclusters):
                    if self.classifier == '1nn':
                        nearestData = KNeighborsRegressor(n_neighbors=1).fit(past_centroid, temp_current_centroids)
                        centroid_label = nearestData.predict([temp_current_centroids[k]])[0]
                        _,new_label_data = nearestData.kneighbors([temp_current_centroids[k]])
                        new_label_data = new_label_data[0][0]

                # concordant data 
                for l in range(0, len(pool_data)):
                    if pool_data[l][-1] == 1:
                        concordant_label_count += 1
                
                if concordant_label_count/self.maxpool < 1:
                    labeled_data = pool_data
                    labeled_data_labels = new_label_data
                    past_centroid = temp_current_centroids
                
                # reset 
                pool_data = np.zeros(np.shape(pool_data)[1])
                pool_label = np.zeros(np.shape(pool_label))
                pool_index = 0    
            # get prediction score 
            self.class_error[t] = self.classification_error(preds=pool_label, L_test= Ye)
            self.accuracy[t] = 1 - self.class_error[t]            

if __name__ == '__main__':
    dataset = SetData()
    run_scargc = SCARGC(Xinit = dataset.X[0], Yinit = dataset.Y)
    # DS = Yts ; T = Xts
    run_scargc.run(Xts = dataset.X, Yts = dataset.Y )
    print('error:\n' , run_scargc.class_error, '\n')
    print('accuracy:\n', run_scargc.accuracy)

