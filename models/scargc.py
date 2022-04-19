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
import numpy as np 
from scipy import stats
from sklearn import preprocessing
from tqdm import tqdm
import benchmark_datagen as bdg
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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
        # set the data [these will be the labeled data]
        self.X = Xinit
        self.Y = Yinit
        # set resample 
        self.resample = resample
        # set max pool size 
        self.maxpool = maxpool
        # initialize the cluster model 
        self._initialize()
        self.T = 0


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
            yhat = self.Y[labels==i]
            mode_val,_ = stats.mode(yhat)
            self.class_cluster[i] = mode_val[0]
        
    
    def run(self, Xts, Yts): 
        '''
        '''
        self.T = len(Xts)
        N = len(Xts[0])
        
        # run the experiment 
        for t in tqdm(range(self.T-1)):
            # get the data from time T and resample if required 
            Xt, Yt = Xts[t], Yts[t]
            Xe, Ye = Xts[t+1], Yts[t+1]
            if self.resample: 
                ii = np.random.randint(0, N, N)
                Xt, Yt = Xt[ii], Yt[ii]
            for n in range(len(Xt)): 
                # make the prediction
                if self.classifier == '1nn': 
                    # one nearest neighbor classifier
                    knn = KNeighborsRegressor(n_neighbors=1).fit(self.X, self.Y)
                    predicted_label = knn.predict([Xt])
                else: 
                    ValueError('The classifier %s is not implemented. ' % (self.classifier))
                
                if n == 0: 
                    pool_data = Xt
                    pool_label = np.array([predicted_label])
                else:
                    pool_data = np.vstack((pool_data, Xt))
                    pool_label = np.concatenate((pool_label, np.array([predicted_label])))

                if len(pool_label) == self.maxpool: 
                    # mex 
                    centroids_cur = KMeans(n_clusters=self.Kclusters, 
                                           init=self.cluster.cluster_centers_).fit(pool_data).cluster_centers_

                    for k in range(self.Kclusters):
                        if self.classifier == '1nn': 
                            mdl = KNeighborsClassifier(n_neighbors=1).fit(self.X, self.Y)
                            yhat = mdl.predict([centroids_cur[k]])[0]
                            _,nn = mdl.kneighbors([centroids_cur[k]])
                            nn = nn[0][0]


        self.classifier

if __name__ == '__main__':
    set_data = bdg.Datagen()
    data_gen = set_data.gen_dataset('UG_2C_2D')

    data ={}
    labeled = {}
    unlabeled = {}
    ts = 0

    ## set a self.data dictionary for each time step 
    ## self.dataset[0][i] loop the arrays and append them to dictionary
    for i in range(0, len(data_gen[0])):
        ts += 1
        data[ts] = data_gen[0][i]


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
    
    
    labeled = labeled[1]
    ds = data[1]

    # unlabeled = np.array(list(unlabeled.values()))
    
    # X_train = np.array(labeled[0])
    

    # X_test = np.array(unlabeled[0])

    # if len(X_train) < len(X_test):
    #     dif = int(len(X_test) - len(X_train))
    #     X_test = list(X_test)
    #     for i in range(0,dif):
    #         X_test.pop()
    #     X_test = np.array(X_test)
    # elif len(X_train) > len(X_test):
    #     dif = int(len(X_train) - len(X_test))
    #     X_train = list(X_train)
    #     for i in range(0,dif):
    #         X_train.pop()
    #     X_train = np.array(X_train)
    
    # dataset = np.array(list(data.values()))
    # dataset = np.array(dataset[0])
    
    init_ds = ds[:len(labeled)]

    run_scargc = SCARGC(labeled, init_ds)
    run_scargc.run(labeled, ds)

