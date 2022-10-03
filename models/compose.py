#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          compose.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""

# MIT License
#
# Copyright (c) 2021 Martin M Lopez
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

from cProfile import run
from socketserver import ThreadingUnixDatagramServer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cse 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import qns3vm as ssl
import benchmark_datagen as bmdg
import random
import time 
import label_propagation as lbl_prop
import util as ut
import matplotlib.animation as animation
import math
import sklearn.metrics as metric
import classifier_performance as cp
import jax

class COMPOSE: 
    def __init__(self, 
                classifier = 'QN_S3VM', 
                method= 'fast_compose',
                num_cores = 0.8, 
                selected_dataset = 'UG_2C_2D'): 
        """
        Initialization of Fast COMPOSE
        """
        self.timestep = 0                   # The current timestep of the datase
        self.n_cores =  num_cores                   # Level of feedback displayed during run {default}
        self.data = {}                      #  array of timesteps each containing a matrix N instances x D features
        self.labeled = {}                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = {}
        self.core_supports = {}             #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.total_time = []
        self.selected_dataset = selected_dataset
        self.classifier = classifier
        self.method = method                
        self.dataset = selected_dataset
        self.predictions = {}                   # predictions from base classifier 
        self.user_data_input = {}
        self.hypothesis = {}                # hypothesis to copy available labels & core supports 
        self.performance_metric = {}
        self.avg_perf_metric = {}
        self.compact_time = {}
        self.num_cs = {}
        
        if self.classifier is None:
            avail_classifier = ['knn', 's3vm']
            print('The following classifiers are available:\n' , avail_classifier)
            classifier_input = input('Enter classifier:')
            self.classifier = classifier_input
        

    def compose(self):
        """
        """
        # set labels and unlabeles and dataset to process
        self.set_data()

        # set drift window
        self.set_drift_window()
    
    def set_drift_window(self):
        """
        Finds the lower and higher limits to determine drift
        Initial assumption is based on dataset min/max
        """
        self.figure_xlim = np.amin(self.dataset)
        self.figure_ylim = np.amax(self.dataset)


    def set_cores(self):
        """
        Establishes number of cores to conduct parallel processing
        """
        num_cores = multiprocessing.cpu_count()         # determines number of cores
        percent_cores = math.ceil(self.n_cores * num_cores)
        if percent_cores > num_cores:
            print("You do not have enough cores on this machine. Cores have to be set to ", num_cores)
            self.n_cores = int(num_cores)                   # sets number of cores to available 
        else:
            self.n_cores = int(percent_cores)                   # original number of cores to 1


    def get_core_supports(self, unlabeled, timestep):
        """
        Method provides core supports based on desired core support extraction.
        Available Core Support Extraction includes: 
        GMM, Parzen Window, and Alpha Shape Core Supports

        Prior to doing the core support extraction: 
        This method preprocesses the data before extracting the core supports from the stream
        The intent of this method is to complete the following:
            1. Remove duplicate data instances 
            2. Sort the classes prior to extracting core supports 
            3. Extract the core supports
        """
        ts = timestep
        # make sure hypothesis are the labels based on label propagation preds
        self.hypothesis[ts] = self.hypothesis[ts][:,-1]
        # 1. Remove duplicate data instances 
        # check the data bu rows and remove the duplicate instances keeping track what was removed
        self.data[ts], sortID = np.unique(self.data[ts], axis=0, return_index=True)
        # remove labels of removed instances
        sorter = []
        for id in sortID:
            if id > len(self.labeled[ts]):
                break
            else:
                sorter.append(id) 
        self.labeled[ts] = self.labeled[ts][sorter]
        # remove core supports dupes
        if ts > 1:
            sorter = []
            for id in sortID:
                if id > len(self.core_supports[ts]):
                    break
                else:
                    sorter.append(id) 
            #TODO out of bounds
            self.core_supports[ts] = self.core_supports[ts][sorter]
        # remove hypothesis dupes
        sorter = []
        for id in sortID:
            if id > len(self.hypothesis[ts]):
                break
            else:
                sorter.append(id)
        self.hypothesis[ts] = self.hypothesis[ts][sorter]
        # remove unlabeled data indices of removed instances
        sorter = []
        for id in sortID:
            if id > len(unlabeled):
                break
            else:
                sorter.append(id)
        unlabeled = unlabeled[sorter]

        # Sort the class
        uniq_class = np.unique(self.hypothesis[ts])     # determine number of classes
        # sort by the class
        self.hypothesis[ts], sortID = np.sort(self.hypothesis[ts], kind="heapsort", axis=0), np.argsort(self.hypothesis[ts], kind="heapsort", axis=0)
        # match data with sort 
        self.data[ts] = self.data[ts][sortID]
        # match labeles with sort 
        self.labeled[ts] = self.labeled[ts][sortID]
        
        # match core supports with sort 
        if ts > 0:
            sorter = []
            # if the sortID is larger than any index in the available core supports
            for i in range(len(sortID)):
                if len(self.core_supports[ts-1])-1 < sortID[i]:
                    pass
                else:
                    sorter.append(sortID[i])
            self.core_supports[ts-1] = self.core_supports[ts-1][sorter]
            # self.core_supports[ts-1] = self.core_supports[ts-1][sortID]
        # match unlabeled with sort
        unlabeled = unlabeled[sortID]
        t_start = time.time()
        # class_offset = 0 # class offset to keep track of how many instances have been analyzed so each class can be returned in correct spot after cse 
        # ------------------------------------------------------------
        # step 3 in this method to extract core supports 
        # step 7 -9 
        # for each class:  
            # call CSE for core supports, 
            # add core supports to labeled data
            # L^(t+1) = L^(t+1) U CSc
            # Y^(t+1) = Y^(t+1) U {y_u: u in [|CSc|], y = c}
        # ------------------------------------------------------------

        for c in uniq_class:
            class_ind = np.squeeze(np.argwhere(self.hypothesis[ts] == c))
            if class_ind is None:
                extract_cs = cse.CSE(data=self.data[ts], method=self.method)    # gets core support based on first timestep
            else:
                extract_cs = cse.CSE(data=self.data[ts][class_ind], method=self.method)    # gets core support based on first timestep
        t_end = time.time()
        self.compact_time[ts] = t_end - t_start
        self.core_supports[ts] = extract_cs.core_support_extract()
        self.num_cs[ts] = len(self.core_supports[ts])

        # set hypothesis dimension as start of method
        to_add = np.zeros((len(self.hypothesis[ts]), np.shape(self.data[ts])[1]-1))
        self.hypothesis[ts] = np.column_stack((to_add, self.hypothesis[ts]))

    def set_data(self):
        """
        Method sets the dataset in its repespective bins, data with timesteps, gets labaled data and unlabeled data from dataset
        """
        if not self.dataset:
            avail_data_opts = ['UG_2C_2D','MG_2C_2D','1CDT', '2CDT', 'UG_2C_3D','1CHT','2CHT','4CR','4CREV1','4CREV2','5CVT','1CSURR',
                '4CE1CF','FG_2C_2D','GEARS_2C_2D', 'keystroke', 'UG_2C_5D', 'UnitTest']
            print('The following datasets are available:\n' , avail_data_opts)
            self.dataset = input('Enter dataset:')
        self.user_data_input = self.dataset
        data_gen = bmdg.Datagen()
        dataset_gen = data_gen.gen_dataset(self.dataset)
        self.dataset = dataset_gen              
        
        ts = 0

        ## set a self.data dictionary for each time step 
        ## self.dataset[0][i] loop the arrays and append them to dictionary
        for i in range(0, len(self.dataset[0])):
            self.data[ts] = self.dataset[0][i]
            ts += 1
        
        # filter out labeled and unlabeled from of each timestep
        for i in self.data:
            len_of_batch = len(self.data[i])
            label_batch = []
            unlabeled_batch = []            
            for j in range(0, len_of_batch - 1):
                if self.data[i][j][2] == 1:
                    label_batch.append(self.data[i][j])
                    self.labeled[i] = label_batch
                else:
                    unlabeled_batch.append(self.data[i][j])
                    self.unlabeled[i] = unlabeled_batch

        # convert labeled data to match self.data data structure
        labeled_keys = self.labeled.keys()
        for key in labeled_keys:        
            if len(self.labeled[key]) > 1:
                len_of_components = len(self.labeled[key])
                array_tuple = []
                for j in range(0, len_of_components):
                    array = np.array(self.labeled[key][j])
                    arr_to_list = array.tolist()
                    array_tuple.append(arr_to_list)
                    array = []
                    arr_to_list = []
                concat_tuple = np.vstack(array_tuple)
                self.labeled[key] = concat_tuple
        
        # convert unlabeled data to match self.data data structure
        unlabeled_keys = self.unlabeled.keys()
        for key in unlabeled_keys:        
            if len(self.unlabeled[key]) > 1:
                len_of_components = len(self.unlabeled[key])
                array_tuple = []
                for j in range(0, len_of_components):
                    array = np.array(self.unlabeled[key][j])
                    arr_to_list = array.tolist()
                    array_tuple.append(arr_to_list)
                    array = []
                    arr_to_list = []
                concat_tuple = np.vstack(array_tuple)    
                self.unlabeled[key] = concat_tuple 

    def learn(self, X_train_l, L_train_l, X_train_u, X_test):
        """
        Available classifiers : 'label_propagation',  'QN_S3VM'

        For QN_S3VM:  
        Sets classifier by getting the classifier object from ssl module
        loads classifier based on user input
        The QN_S3VM options are the following:  
        X_l -- patterns of labeled part of the data
        L_l -- labels of labeled part of the data
        X_u -- patterns of unlabeled part of the data
        random_generator -- particular instance of a random_generator (default None)
        kw -- additional parameters for the optimizer
        """

        if self.classifier == 'QN_S3VM':
            random_gen = random.Random()
            random_gen.seed(0)

            X_L_train = []
            X_train_l = np.array(X_train_l)

            for i in range(0, len(X_train_l)):
                add = np.array(X_train_l[i])
                X_L_train.append(add)
            X_train_l = X_L_train
            
            L_l_train = []
            
            L_train_l = np.array(L_train_l)
            for i in range(0, len(L_train_l)):
                add = np.array(L_train_l[:,-1][i]) 
                L_l_train.append(add.astype(int))
            L_train_l = L_l_train
            
            L_train_l = np.array(L_train_l)
            
            model = ssl.QN_S3VM(X_train_l, L_train_l, X_train_u, random_gen)
            model.train()
            preds = model.getPredictions(X_test)
            return preds
            
        elif self.classifier == 'label_propagation':
            ssl_label_propagation = lbl_prop.Label_Propagation(X_train_l, L_train_l, X_train_u)
            preds = ssl_label_propagation.ssl()
            return preds
        # TODO: KNN 

    def set_stream(self, ts):
        """
        The intent of the method is so when ts!=1 we add the information of core supports from the previous timestep
        ts = current timestep. This method should not be invoked unless it is after timestep 1 
        This method will conduct the following:
        1. append the current data stream with the core supports 
        2. append the hypothesis to include the classes of the core supports
        3. append the labels to include the class of the core supports 
        4. append the core supports to accomodate the added core supports of the previous timestep 
        """
        # append the current data with the core supports
        # D_t = {(xl, yl): x in L where any l} Union {(xu, ht(xu_)): x in U where any u }
        self.data[ts] = np.concatenate((self.data[ts-1], self.core_supports[ts-1]))
        # append hypothesis to include classes of the core supports
        self.hypothesis[ts] = np.concatenate((self.hypothesis[ts-1], self.core_supports[ts-1])) 
        # append the labels to include the class of the core supports
        # receive labeled data from core supports and labels 
        self.labeled[ts] = np.concatenate((self.labeled[ts-1], self.core_supports[ts-1]))
        # append the core supports to accomodate the added core supports from previous ts
        # I dont think this is needed
        # if ts >= 1:
        #     self.core_supports[ts] = np.concatenate((self.core_supports[ts], self.core_supports[ts-1]))

    def classify(self, ts):
        """
        This method classifies the unlabeled data then goes through the Semi-Supervised Learning Algorithm to receive labels from classification. 
        In this method we complete the following: 
        1. sort the hypothesis sos that the unlabeled data is that the bottom
        2. sort the data to match hypothesis shift
        3. sort the labeled so that the it matches the hypothesis shift
        4. sort the core supports to match hypothesis shift
        5. classify the data via SSL algorithm
        ------------------------------------------
        This is step 4 of the COMPOSE Algorithm:
        Call SSL with L^t, Y^t, and U^t to obtain hypothesis, h^t: X->Y
        """
        if len(self.hypothesis[ts]) > len(self.data[ts]):
            diff = len(self.hypothesis[ts]) - len(self.data[ts])
            self.hypothesis[ts] = self.hypothesis[ts][:-diff]

        # 1. sort the hypothesis so unlabeled data is at the bottom; we do this by sorting in descending order
        self.hypothesis[ts], sortID = -np.sort(-self.hypothesis[ts], kind="heapsort", axis=0), np.argsort(-self.hypothesis[ts], kind="heapsort", axis=0)
        # 2. sort the data to match hypothesis shift
        self.data[ts] = np.take_along_axis(self.data[ts], sortID, axis=0)
        # 3. sort labeled to match hypothesis shift
        self.labeled[ts] = np.take_along_axis(self.labeled[ts], sortID, axis=0)
        # 4. sort the core supports to match hypothesis 
        # self.core_supports[ts] = np.take_along_axis(self.core_supports[ts], sortID, axis=0)

        # classify 
        # step 4 call SSL with L, Y , U
        t_start = time.time()
        self.predictions[ts] = self.learn(X_train_l= self.hypothesis[ts], L_train_l=self.labeled[ts], X_train_u = self.data[ts], X_test=self.data[ts+1])
        t_end = time.time() 

        # obtain hypothesis ht: X-> Y 
        self.hypothesis[ts] = self.predictions[ts]
        # get performance metrics of classification 
        perf_metric = cp.PerformanceMetrics(timestep= ts, preds= self.hypothesis[ts], test= self.labeled[ts], \
                                            dataset= self.selected_dataset , method= self.method , \
                                            classifier= self.classifier, tstart=t_start, tend=t_end) # test[:,-1]
        self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= self.hypothesis[ts], test= self.labeled[ts])
        return self.predictions[ts]

    def run(self):
        # set cores
        self.set_cores()
        with ProcessPoolExecutor(max_workers=self.n_cores):
            self.compose()
            start = self.timestep
            timesteps = self.data.keys()
            total_time_start = time.time() 
            ts = start
            for ts in range(0, len(timesteps)-1):     # iterate through all timesteps from the start to the end of the available data
                self.timestep = ts
                # determine if there are any labeled data & add core_suppports from previous timestep
                # steps 1 - 2
                if ts == 0: # there will be no core supports @ ts = 0 
                    self.hypothesis[ts] = self.labeled[ts]
                # if ts != 0 
                else:
                    # add previous core supports from previous time step
                    self.set_stream(ts)
                # step 3 receive unlabeled data U^t = { xu^t in X, u = 1,..., N}
                # step 4 call SSL with L^t, Y^t, and U^t
                unlabeled_data = self.classify(ts) 
                # get core supports 
                self.get_core_supports(timestep= ts , unlabeled= unlabeled_data)
                
                # remove core supports from previous time step
                # remove core supports from data
                if ts != 0:
                    df_data_ts = pd.DataFrame(self.data[ts])
                    df_data_prev = pd.DataFrame(self.data[ts-1])
                    merge_data_df = df_data_ts.merge(df_data_prev, how='left', indicator= True)
                    merge_data_df = merge_data_df[merge_data_df['_merge']=='left_only']
                    self.data[ts] = merge_data_df.to_numpy()
                    self.data[ts] = self.data[ts][:,:-1]
                    # remove core supports from hypothesis
                    df_hypoth_ts = pd.DataFrame(self.hypothesis[ts])
                    df_hypoth_prev = pd.DataFrame(self.hypothesis[ts-1])
                    merge_hypoth = df_hypoth_ts.merge(df_hypoth_prev, how='left', indicator=True)
                    merge_hypoth = merge_hypoth[merge_hypoth['_merge']=='left_only']
                    self.hypothesis[ts] = merge_hypoth.to_numpy()
                    self.hypothesis[ts] = self.hypothesis[ts][:,:-1]
                    # remove core supports from labels
                    df_label_ts = pd.DataFrame(self.labeled[ts])
                    df_label_prev = pd.DataFrame(self.labeled[ts-1])
                    merge_label_df = df_label_ts.merge(df_label_prev, how='left', indicator=True)
                    merge_label_df = merge_label_df[merge_label_df['_merge']=='left_only']
                    self.labeled[ts] = merge_label_df.to_numpy()
                    self.labeled[ts] = self.labeled[ts][:,:-1]
                    # remove core supports frpm previous core supports
                    df_cs_ts = pd.DataFrame(self.core_supports[ts])
                    df_cs_prev = pd.DataFrame(self.core_supports[ts-1])
                    merge_cs = df_cs_ts.merge(df_cs_prev, how='left', indicator=True)
                    merge_cs = merge_cs[merge_cs['_merge']=='left_only']
                    self.core_supports[ts] = merge_cs.to_numpy()
                    self.core_supports[ts] = self.core_supports[ts][:,:-1]
                
            total_time_end = time.time()
            self.total_time = total_time_end - total_time_start
            # figure out how to call out functions
            avg_metrics = cp.PerformanceMetrics(tstart= total_time_start, tend= total_time_end)
            self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
            return self.avg_perf_metric
