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
from sklearn.cluster import Birch, KMeans
from sklearn.mixture import GaussianMixture as GMM
from scipy import stats
from matplotlib import pyplot as plt

class MClassification(): 
    def __init__(self, 
                classifier,
                dataset,
                NClusters:int = 10): 
        """
        """
        self.classifier = classifier
        self.dataset = dataset
        self.NClusters = NClusters
        self.class_cluster = {}
        self.mcluster = {}
        self._initialize()
    
    def _setData(self):
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
        
        self.Y = labeled        # set of all labels as a dict per timestep ; we only need X[0] for initial labels
        self.X = data           # data stream
        self.T = labeled[0]     # initial labeled set    

    def _initialize(self):
        """
        Get initial labeled data T 
        Begin MC for the labeled data
        """
        self._setData()
        # initial set of cluster based on inital labeled data T; using next time step from datastream for preds
        if self.classifier == 'kmeans':
            kmeans_model = KMeans(n_clusters=self.NClusters)
            kmeans_model.fit(self.T)
            preds = kmeans_model.predict(self.Y[1])
        elif self.classifier == 'gmm':
            gmm_model = GMM(n_components=self.NClusters)
            gmm_model.fit(self.T)
            preds = gmm_model.predict(self.Y[1])
        elif self.classifier == 'birch':
            birch_model = Birch(branching_factor=50, n_clusters= self.NClusters)
            birch_model.fit(self.T)
            preds = birch_model.predict(self.Y[1])

        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class.
        for i in range(self.NClusters):
                xhat = self.X[i][preds]
                mode_val,_ = stats.mode(xhat)
                self.class_cluster[i] = mode_val

    def _create_mclusters(self):
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
        
    def _classify(self):
        pass

    def _euclidean_distance(self):
        pass


    def run(self, Xt, Yt, Ut):
        """
        """
        self.classifier
        """
        1. Get first set of labeled data T 
        2. Create MCs of the labeled data of first data T
        3. classify 
            a. predicted label yhat_t for the x_t from stream is given by the nearest MC by euclidean distance 
            b. determine if the added x_t to the MC exceeds the maximum radius of the MC
                1.) if the added x_t to the MC does not exceed the MC radius -> update the (N, LS, SS )
                2.) if the added x_t to the MC exceeds radius -> new MC carrying yhat_t is created to carry x_t
            c. The 

        """


run_mclass = MClassification(classifier='gmm', dataset='UG_2C_2D')

# %%
