#!/usr/bin/env python 
"""
Application:        Online Learning in Extreme Verification Latency
File name:          run_experiments.py
Author:             Martin Manuel Lopez
Creation:           03/26/2022

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

import numpy as np
import benchmark_datagen as bdg
from sklearn.cluster import Birch

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

class MClassification(): 
    def __init__(self, 
                classifier, 
                method): 
        """
        """
        self.classifier = classifier
        self.mcluster = {}

    def _initialize(self):
        """
        Get initial labeled data T 
        Begin MC for the labeled data
        """

    def create_mclusters(self):
        """
        Clustering options:
        1. k-means
        2. GMM 
        3. Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)
        """
        if self.classifier == 'kmeans':
            pass
        elif self.classifier == 'gmm':
            pass
        elif self.classifier == 'birch':
            pass

    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier
        """
        1. Get first set of labeled data T 
        2. Craete MCs of the labeled data 
        3. classify 
            a. predicted label yhat_t for the x_t from stream is given by the nearees MC by euclidean distance 
            b. determine if the added x_t to the MC exceeds the maximum radius of the MC
                1.) if the added x_t to the MC does not exceed the MC radius -> update the (N, LS, SS )
                2.) if the added x_t to the MC exceeds radius -> new MC carrying yhat_t is created to carry x_t
            c. The 

        """