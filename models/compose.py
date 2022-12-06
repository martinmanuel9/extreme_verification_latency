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

import numpy as np
import pandas as pd
import cse 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import qns3vm as ssl
import datagen_synthetic as bmdg
import unsw_nb15_datagen as unsw
import random
import time 
import label_propagation as lbl_prop
import util as ut
import matplotlib.animation as animation
import math
import classifier_performance as cp
from sklearn.svm import SVC, SVR

class COMPOSE: 
    def __init__(self, 
                classifier = 'QN_S3VM', 
                method= 'fast_compose',
                mode = 'gmm', 
                num_cores = 0.8, 
                selected_dataset = 'UG_2C_2D',
                datasource = 'synthetic'): 
        """
        Initialization of Fast COMPOSE
        """
        self.timestep = 0                   # The current timestep of the datase
        self.n_cores =  num_cores           # Level of feedback displayed during run {default}
        self.data = {}                      #  array of timesteps each containing a matrix N instances x D features
        self.labeled = {}                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.core_supports = {}             #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.total_time = []
        self.selected_dataset = selected_dataset
        self.classifier = classifier
        self.method = method   
        self.mode = mode             
        self.dataset = selected_dataset
        self.datasource = datasource 
        self.predictions = {}               # predictions from base classifier 
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
        if self.method == 'compose':
            ts = timestep
            # make sure hypothesis are the labels based on label propagation preds
            # 1. Remove duplicate data instances 
            # check the data bu rows and remove the duplicate instances keeping track what was removed
            self.data[ts], sortID = np.unique(self.data[ts], axis=0, return_index=True)
            # remove labels of removed instances
            sorter = []
            # if index is out of range we skip to the next index
            for id in sortID:
                if id >= len(self.labeled[ts]):
                    break
                else:
                    sorter.append(id) 
            self.labeled[ts] = self.labeled[ts][sorter]
            # remove core supports dupes
            sorter = []
            for id in sortID:
                if id >= len(self.core_supports[ts]):
                    break
                else:
                    sorter.append(id) 
            self.core_supports[ts] = self.core_supports[ts][sorter] 
            # remove hypothesis dupes
            sorter = []
            for id in sortID:
                if id >= len(self.hypothesis[ts]):
                    break
                else:
                    sorter.append(id)
            self.hypothesis[ts] = self.hypothesis[ts][sorter]
            # remove unlabeled data indices of removed instances
            sorter = []
            for id in sortID:
                if id >= len(unlabeled):
                    break
                else:
                    sorter.append(id)
            unlabeled = unlabeled[sorter]

            # Sort by the classes
            uniq_class = np.unique(self.hypothesis[ts])     # determine number of classes
            # sort by the class
            self.hypothesis[ts], sortID = np.sort(self.hypothesis[ts], kind="heapsort", axis=0), np.argsort(self.hypothesis[ts], kind="heapsort", axis=0)
            # match data with sort 
            self.data[ts] = self.data[ts][sortID]
            # match labeles with sort 
            if self.labeled[ts].size == 0:
                self.labeled[ts] = self.labeled[ts-1]
                sorter = []
                for id in sortID:
                    if id >= len(self.labeled[ts]):
                        break
                    else:
                        sorter.append(id)
                self.labeled[ts] = self.labeled[ts][sorter]
            else:
                sorter = []
                for id in sortID:
                    if id >= len(self.labeled[ts]):
                        break
                    else:
                        sorter.append(id)
                self.labeled[ts] = self.labeled[ts][sorter]
            # match core supports with sort 
            sorter = []
            # if the sortID is larger than any index in the available core supports
            for i in range(len(sortID)):
                if len(self.core_supports[ts])-1 < sortID[i]:
                    pass
                else:
                    sorter.append(sortID[i])
            self.core_supports[ts] = self.core_supports[ts][sorter]
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
            # c_offset is used to keep track how many instances have been analyzed so each class can be returned in the correct spot after cse
            c_offset = 0       
            self.data[ts] = np.squeeze(self.data[ts])
            # For compose we do core support extraction
            core_supports = np.zeros((1, np.shape(self.data[ts])[1]))
            for c in uniq_class:
                class_ind = np.squeeze(np.argwhere(self.hypothesis[ts] == c))
                if class_ind is None:
                    extract_cs = cse.CSE(data=self.data[ts], mode=self.mode)    # gets core support based on first timestep
                else:
                    extract_cs = cse.CSE(data= np.squeeze(self.data[ts][class_ind]), mode=self.mode)    # gets core support based on first timestep
                self.core_supports[ts] = extract_cs.core_support_extract()
                inds = np.argwhere(self.core_supports[ts][:,0])
                inds = inds[:,0]
                inds = inds + c_offset
                sorter = []
                for ind in inds:
                    if ind >= len(self.core_supports[ts][:,0]):
                        break
                    else:
                        sorter.append(ind)
                new_cs = np.squeeze(self.core_supports[ts][sorter])
                if new_cs.ndim < 2:
                    new_cs[0] = 2
                elif new_cs.ndim > 1:
                    new_cs[:,0] = 2 
                core_supports = np.vstack((core_supports, new_cs))
                c_offset = c_offset + extract_cs.N_features
            core_supports = np.squeeze(core_supports)
            core_supports = np.delete(core_supports, 0, axis=0)
            self.core_supports[ts] = core_supports
            t_end = time.time()
            self.compact_time[ts] = t_end - t_start
            self.num_cs[ts] = len(self.core_supports[ts])

        elif self.method == 'fast_compose':
            ts = timestep
            # make sure hypothesis are the labels based on label propagation preds
            # 1. Remove duplicate data instances 
            # check the data by rows and remove the duplicate instances keeping track what was removed
            self.data[ts], sortID = np.unique(self.data[ts], axis=0, return_index=True)
            # remove labels of removed instances
            sorter = []
            # if index is out of range we skip to the next index
            for id in sortID:
                if id >= len(self.labeled[ts]):
                    break
                else:
                    sorter.append(id) 
            self.labeled[ts] = self.labeled[ts][sorter]
            # remove core supports dupes
            sorter = []
            for id in sortID:
                if id >= len(self.core_supports[ts]):
                    break
                else:
                    sorter.append(id) 
            self.core_supports[ts] = self.core_supports[ts][sorter] 
            # remove hypothesis dupes
            sorter = []
            for id in sortID:
                if id >= len(self.hypothesis[ts]):
                    break
                else:
                    sorter.append(id)
            self.hypothesis[ts] = self.hypothesis[ts][sorter]
            # remove unlabeled data indices of removed instances
            sorter = []
            for id in sortID:
                if id >= len(unlabeled):
                    break
                else:
                    sorter.append(id)
            unlabeled = unlabeled[sorter]

            # Sort by the classes
            uniq_class = np.unique(self.hypothesis[ts])     # determine number of classes
            # sort by the class
            self.hypothesis[ts], sortID = np.sort(self.hypothesis[ts], kind="heapsort", axis=0), np.argsort(self.hypothesis[ts], kind="heapsort", axis=0)
            # match data with sort 
            self.data[ts] = self.data[ts][sortID]
            # match labeles with sort 
            if self.labeled[ts].size == 0:
                self.labeled[ts] = self.labeled[ts-1]
                sorter = []
                for id in sortID:
                    if id >= len(self.labeled[ts]):
                        break
                    else:
                        sorter.append(id)
                self.labeled[ts] = self.labeled[ts][sorter]
            else:
                sorter = []
                for id in sortID:
                    if id >= len(self.labeled[ts]):
                        break
                    else:
                        sorter.append(id)
                self.labeled[ts] = self.labeled[ts][sorter]
            # match core supports with sort 
            sorter = []
            # if the sortID is larger than any index in the available core supports
            for i in range(len(sortID)):
                if len(self.core_supports[ts])-1 < sortID[i]:
                    pass
                else:
                    sorter.append(sortID[i])
            self.core_supports[ts] = self.core_supports[ts][sorter]
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
            # c_offset is used to keep track how many instances have been analyzed so each class can be returned in the correct spot after cse
            c_offset = 0       
            self.data[ts] = np.squeeze(self.data[ts])
            # D_t = {{(x_ut,ht(x_ut)) :x ∈ Ut∀u}}
            # step 7 for fast compose
            core_supports = np.zeros((1, np.shape(self.data[ts])[1]))
            for c in uniq_class:
                class_ind = np.squeeze(np.argwhere(self.hypothesis[ts] == c))
                new_cs = self.data[ts][class_ind]
                new_cs[:,0] = 2
                core_supports = np.vstack((core_supports, new_cs))
            core_supports = np.squeeze(core_supports)
            core_supports = np.delete(core_supports, 0, axis=0)
            self.core_supports[ts] = core_supports
            # add to labeled data for next time step
            self.labeled[ts] = np.concatenate((self.labeled[ts], self.core_supports[ts][:,-1]))
            t_end = time.time()
            self.compact_time[ts] = t_end - t_start
            self.num_cs[ts] = len(self.core_supports[ts])

    def set_data(self):
        """
        Method sets the dataset in its repespective bins, data with timesteps, gets labaled data and unlabeled data from dataset
        """
        # if not self.dataset:
        #     avail_data_opts = ['UG_2C_2D','MG_2C_2D','1CDT', '2CDT', 'UG_2C_3D','1CHT','2CHT','4CR','4CREV1','4CREV2','5CVT','1CSURR',
        #         '4CE1CF','FG_2C_2D','GEARS_2C_2D', 'keystroke', 'UG_2C_5D', 'UnitTest']
        #     print('The following datasets are available:\n' , avail_data_opts)
        #     self.dataset = input('Enter dataset:')
        # self.user_data_input = self.dataset
        if self.datasource == 'synthetic': 
            # get data, labels, and first core supports synthetically for timestep 0
            # data is composed of just the features 
            # labels are the labels 
            # core supports are the first batch with added labels 
            # synthetic data 
            data_gen = bmdg.Synthetic_Datagen()
            data, labels, core_supports, self.dataset = data_gen.gen_dataset(self.dataset) # returns self.data, self.labels, self.use, self.dataset
        elif self.datasource == 'unsw':
            # dataset = UNSW_NB15_Datagen()
            # gen_train_features = dataset.generateFeatTrain
            # gen_test_features =dataset.generateFeatTest 
            # X, y = dataset.create_dataset(train=gen_train_features, test=gen_test_features)
            # we have the following categoires : flow, basic, time, content, generated 
            unsw_gen = unsw.UNSW_NB15_Datagen()
            gen_train_features = unsw_gen.generateFeatTrain
            gen_test_features = unsw_gen.generateFeatTest 
            train , test = unsw_gen.create_dataset(train = gen_train_features, test = gen_test_features)
            data = train['Data']
            labels = train['Label']
            core_supports = train['Use']
            self.dataset = train['Dataset']


        ts = 0 
        # set data (all the features)
        for i in range(0, len(data[0])):
            self.data[ts] = data[0][i]
            ts += 1
        # set all the labels 
        ts = 0
        for k in range(0, len(labels[0])):
            self.labeled[ts] = labels[0][k]
            ts += 1
        # gets first core supports from synthetic
        self.core_supports[0] = np.squeeze(core_supports)

    def learn(self, X_train_l, L_train_l, X_train_u, X_test):
        """
        Available classifiers : 'label_propagation',  'QN_S3VM', 'svm'
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

            # X_L_train = []
            # X_train_l = np.array(X_train_l)
            # for i in range(0, len(X_train_l)):
            #     add = np.array(X_train_l[i])
            #     X_L_train.append(add)
            # X_train_l = X_L_train
            
            # L_l_train = []
            # L_train_l = np.array(L_train_l)
            # for i in range(0, len(L_train_l)):
            #     add = np.array(L_train_l[i]) 
            #     L_l_train.append(add)
            # L_train_l = L_l_train
            # L_train_l = np.array(L_train_l).astype(int)

            model = ssl.QN_S3VM(X_train_l, L_train_l, X_train_u[:,-1], random_gen)
            model.train()
            preds = model.getPredictions(X_test)
            return preds
            
        elif self.classifier == 'label_propagation':
            ssl_label_propagation = lbl_prop.Label_Propagation(X_train_l, L_train_l, X_train_u)
            preds = ssl_label_propagation.ssl()
            return preds
        elif self.classifier == 'svm':
            print(np.unique(X_train_l))
            ssl_svm = SVC(gamma='auto').fit(X_train_u[:,:-1], X_train_l)
            preds = ssl_svm.predict(X_train_u)
            return preds

    def set_stream_compose(self, ts):
        """
        The intent of the method for compose is so when ts!=1 we add the information of core supports from the previous timestep
        ts = current timestep. This method should not be invoked unless it is after timestep 1 
        This method will conduct the following:
        1. append the current data stream with the core supports 
        2. append the hypothesis to include the classes of the core supports
        3. append the labels to include the class of the core supports 
        4. append the core supports to accomodate the added core supports of the previous timestep 
        """
        # append the current data with the core supports
        # if compose :
        # D_t = {(xl, yl): x in L where any l} Union {(xu, ht(xu_)): x in U where any u }
        cs_indx = np.argwhere(self.core_supports[ts-1][:,0]==2)
        cs_indx = np.squeeze(cs_indx)
        prev_cs = self.core_supports[ts-1][cs_indx]
        self.data[ts] = np.concatenate((self.data[ts-1], prev_cs))
        # append hypothesis to include classes of the core supports
        self.hypothesis[ts] = np.concatenate((self.hypothesis[ts-1], prev_cs[:,-1])) 
        # append the labels to include the class of the core supports
        # receive labeled data from core supports and labels 
        self.labeled[ts] = np.concatenate((self.labeled[ts-1], prev_cs[:,-1]))
        # append the core supports to accomodate the added core supports from previous ts
        self.core_supports[ts] = np.concatenate((self.core_supports[ts-1], prev_cs))

    def set_stream_fast_compose(self, ts):
        """
        The intent of the method for fast compose 
        # D_t = {{(x_ut,ht(x_ut)) :x ∈ Ut∀u}}
        """
        # sets the same dim to add unlabeled and hypothesis
        to_add = np.zeros((len(self.hypothesis[ts]), (np.shape(self.data[ts])[1] - 1)))
        hypoth = np.column_stack((to_add, self.hypothesis[ts]))
        self.data[ts] = np.vstack((self.data[ts], hypoth))

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
        # 1. sort the hypothesis so unlabeled data is at the bottom; we do this by sorting in descending order
        self.hypothesis[ts], sortID = -np.sort(-self.hypothesis[ts], kind="heapsort", axis=0), np.argsort(-self.hypothesis[ts], kind="heapsort", axis=0)
        # 2. sort the data to match hypothesis shift
        sorter = []
        # if index is out of range we skip to the next index
        for id in sortID:
            if id >= len(self.data[ts]):
                break
            else:
                sorter.append(id)
        self.data[ts] = np.squeeze(self.data[ts][sorter])
        # 3. sort labeled to match hypothesis shift
        sorter = []
        # if index is out of range we skip to the next index
        for id in sortID:
            if id >= len(self.labeled[ts]):
                break
            else:
                sorter.append(id)
        self.labeled[ts] = np.squeeze(self.labeled[ts][sorter])
        # 4. sort the core supports to match hypothesis 
        sorter = []
        # if index is out of range we skip to the next index
        for id in sortID:
            if id >= len(self.core_supports[ts]):
                break
            else:
                sorter.append(id)
        self.core_supports[ts] = np.squeeze(self.core_supports[ts][sorter])
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
                                            classifier= self.classifier, tstart=t_start, tend=t_end)
        # make sure that preds and test have same dim
        if self.labeled[ts] is None:
            self.labeled[ts] = np.array(1)
        if len(self.hypothesis[ts]) > len(self.labeled[ts]):
            class_perf_hypoth = self.hypothesis[ts][0:len(self.labeled[ts])]
            self.performance_metric[ts] = perf_metric.findClassifierMetrics(preds= class_perf_hypoth, test= self.labeled[ts])
        else:
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
                if self.method == 'compose':
                    self.timestep = ts
                    # add available labels to hypothesis
                    if ts == 0:
                        if np.sum(self.core_supports[ts][:,-1] == 1) >= 1:
                            lbl_indx = np.argwhere(self.core_supports[ts][:,-1] == 1)
                            self.hypothesis[ts] = self.labeled[ts][lbl_indx]  
                    # steps 1 - 2
                    # if ts != 0 (not the first timestep)
                    if start != ts:
                        # add previous core supports from previous time step
                        self.set_stream_compose(ts)
                    # step 3 receive unlabeled data U^t = { xu^t in X, u = 1,..., N}
                    # step 4 call SSL with L^t, Y^t, and U^t
                    unlabeled_data = self.classify(ts) 
                    # get core supports 
                    self.get_core_supports(timestep= ts , unlabeled= unlabeled_data)
                
                elif self.method == 'fast_compose':
                    self.timestep = ts
                    # add available labels to hypothesis
                    if ts == 0:
                        if np.sum(self.core_supports[ts][:,-1] == 1) >= 1:
                            lbl_indx = np.argwhere(self.core_supports[ts][:,-1] == 1)
                            self.hypothesis[ts] = self.labeled[ts][lbl_indx]  
                    # steps 1 - 2 
                    if start != ts:
                        # add previous core supports from previous time step
                        self.set_stream_compose(ts)
                    # step 3 receive unlabeled data U^t = { xu^t in X, u = 1,..., N}
                    # step 4 call SSL with L^t, Y^t, and U^t
                    hypoth = self.classify(ts) 
                    # step 5 set D_t = {{(x_ut,ht(x_ut)) :x ∈ Ut∀u}}
                    self.set_stream_fast_compose(ts)
                    # get core supports and add to to labeled data for next time step 
                    self.get_core_supports(timestep= ts , unlabeled= hypoth)

            total_time_end = time.time()
            self.total_time = total_time_end - total_time_start
            # figure out how to call out functions
            avg_metrics = cp.PerformanceMetrics(tstart= total_time_start, tend= total_time_end)
            self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
            return self.avg_perf_metric