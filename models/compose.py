#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          compose.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
PhD Advisor: Dr. Gregory Ditzler and Dr. Salim Hariri
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

class COMPOSE: 
    def __init__(self, 
                classifier = 'QN_S3VM', 
                method= 'fast_compose',
                verbose = 1,
                num_cores = 0.8, 
                selected_dataset = 'UG_2C_2D'): 
        """
        Initialization of Fast COMPOSE
        """


        self.timestep = 0                   # The current timestep of the datase
        self.synthetic = 0                  # 1 Allows synthetic data during cse and {0} does not allow synthetic data
        self.n_cores =  num_cores                   # Level of feedback displayed during run {default}
        self.verbose = verbose              #    0  : No Information Displayed
                                            #    1  : Command line progress updates - {default}
                                            #    2  : Plots when possible and Command line progress updates

        self.data = {}                      #  array of timesteps each containing a matrix N instances x D features
        self.labeled = {}                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = {}
        self.core_supports = {}             #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.num_cs = {}                    #  number of core supports 
        self.total_time = 0
        self.cse_opts = []                  # options for the selected cse function in cse class
        self.selected_dataset = selected_dataset
        self.classifier = classifier
        self.method = method                
        self.dataset = selected_dataset
        self.figure_xlim = []
        self.figure_ylim = []
        self.predictions = {}                   # predictions from base classifier 
        self.classifier_accuracy = {}
        self.classifier_error = {}
        self.time_to_predict = {}
        self.user_data_input = {}
        self.avg_results = {}
        self.avg_results_dict = {}
        self.accuracy_sklearn = {}
        self.stream = {}                    # establishing stream
        self.hypothesis = {}                # hypothesis to copy available labels & core supports 
        
        if self.classifier is None:
            avail_classifier = ['knn', 's3vm']
            print('The following classifiers are available:\n' , avail_classifier)
            classifier_input = input('Enter classifier:')
            self.classifier = classifier_input
        
        if verbose is None:
            # set object displayed info setting 
            print("Only 3 options to display information for verbose: \n", 
                "0 - No Info ; \n", 
                "1 - Command Line Progress Updates; \n",
                "2 - Plots when possilbe and Command Line Progress \n")
            print("Set Verbose: ")
            verbose_input = input("Enter display information option:")
            self.verbose = verbose_input

        if self.verbose >= 0 and self.verbose <=2:
            if self.verbose == 1:
                print("Run method: ", self.verbose)
        else:
            print("Only 3 options to display information: \n", 
            "0 - No Info ;\n", 
            "1 - Command Line Progress Updates;\n",
            "2 - Plots when possilbe and Command Line Progress")

    def compose(self):
        """
        Sets COMPOSE dataset and information processing options
        Check if the input parameters are not empty for compose
        This checks if the dataset is empty and checks what option of feedback you want
        Gets dataset and verbose (the command to display options as COMPOSE processes)
        Verbose:    0 : no info is displayed
                    1 : Command Line progress updates
                    2 : Plots when possible and Command Line progress updates
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
        if self.verbose == 1:
            print("Drift window:" , [self.figure_xlim, self.figure_ylim])

    def set_cores(self):
        """
        Establishes number of cores to conduct parallel processing
        """
        num_cores = multiprocessing.cpu_count()         # determines number of cores
        if self.verbose == 1: 
            print("Available cores:", num_cores)
        percent_cores = math.ceil(self.n_cores * num_cores)
        if percent_cores > num_cores:
            print("You do not have enough cores on this machine. Cores have to be set to ", num_cores)
            self.n_cores = int(num_cores)                   # sets number of cores to available 
        else:
            self.n_cores = int(percent_cores)                   # original number of cores to 1
        if self.verbose == 1:
            print("Number of cores executing:", self.n_cores)

    def get_core_supports(self, input_data = None, next_data = None):
        """
        Method provides core supports based on desired core support extraction.
        Available Core Support Extraction includes: 
        GMM, Parzen Window, and Alpha Shape Core Supports
        """

        self.cse = cse.CSE(data=input_data, next_data= next_data)           # gets core support based on first timestep

        if self.method == 'fast_compose':
            self.cse.set_boundary('gmm')
            self.num_cs[self.timestep] = len(self.cse.gmm())
            self.core_supports[self.timestep] = self.cse.gmm()
        elif self.method == 'parzen':
            self.cse.set_boundary(self.method)
            self.num_cs[self.timestep] = len(self.cse.parzen())
            self.core_supports[self.timestep] = self.cse.parzen()
        elif self.method == 'a_shape':
            self.cse.set_boundary(self.method)
            self.core_supports[self.timestep] = self.cse.a_shape_compaction()

    def set_data(self):
        """
        Method sets the dataset in its repespective bins, data with timesteps, gets labaled data and unlabeled data from dataset
        """
        if not self.dataset:
            avail_data_opts = ['UG_2C_2D','MG_2C_2D','1CDT', '2CDT', 'UG_2C_3D','1CHT','2CHT','4CR','4CREV1','4CREV2','5CVT','1CSURR',
                '4CE1CF','FG_2C_2D','GEARS_2C_2D', 'keystroke', 'UG_2C_5D', 'UnitTest']
            print('The following datasets are available:\n' , avail_data_opts)
            self.dataset = input('Enter dataset:')
        if self.verbose == 1 :
            print("Dataset:", self.dataset)
            print("Method:", self.method)
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

    def classify(self, X_train_l, L_train_l, X_train_u, X_test):
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
        elif self.classifier == 'knn':
            self.cse = cse.CSE(data=self.data)
            self.cse.set_boundary('knn')
            self.cse.k_nn()

    
    def classification_error(self, preds, L_test):  
        return np.sum(preds != L_test)/len(preds)

    def results_logs(self):
        avg_error = np.array(sum(self.classifier_error.values()) / len(self.classifier_error))
        avg_accuracy = np.array(sum(self.classifier_accuracy.values()) / len(self.classifier_accuracy))
        avg_exec_time = np.array(sum(self.time_to_predict.values()) / len(self.time_to_predict))
        avg_results_df = pd.DataFrame( {'Dataset': [self.selected_dataset], 'Classifier': [self.classifier],'Method': [self.method],
                            'Avg_Error': [avg_error], 'Avg_Accuracy': [avg_accuracy], 'Avg_Exec_Time': [avg_exec_time], 'Total_Exec_Time' : [self.total_time] }, 
                            columns=['Dataset','Classifier','Method','Avg_Error', 'Avg_Accuracy', 'Avg_Exec_Time', 'Total_Exec_Time'] )
        self.avg_results_dict['Dataset'] = self.selected_dataset
        self.avg_results_dict['Classifier'] = self.classifier
        self.avg_results_dict['Method'] = self.method
        self.avg_results_dict['Average_Error'] = avg_error
        self.avg_results_dict['Average_Accuracy'] = avg_accuracy
        self.avg_results_dict['Avg_Exec_Time'] = avg_exec_time
        self.avg_results_dict['Total_Exec_Time'] = self.total_time
        run_method = self.selected_dataset + '_' + self.classifier + '_' + self.method
        self.avg_results[run_method] = avg_results_df
        
        
        if self.verbose == 1:
            print('Execition Time:', self.total_time[self.user_data_input], "seconds")
            print('Average error:', avg_error)
            print('Average Accuracy:', avg_accuracy)
            print('Average Execution Time per Timestep:', avg_exec_time, "seconds")

        df = pd.DataFrame.from_dict((self.classifier_accuracy.keys(), self.classifier_accuracy.values())).T
        accuracy_scores = pd.DataFrame(df.values, columns=['Timesteps', 'Accuracy'])
        x = accuracy_scores['Timesteps']
        y = accuracy_scores['Accuracy']
        

        if self.verbose == 1:
            plt.xlabel('Timesteps')
            plt.ylabel('Accuracy [%]')
            plt.title('Correct Classification [%]')
            plt.plot(x,y,'o', color='black')
            plt.show()

        return accuracy_scores
    
    def sort_classify(self, data_stream, hypothesis):
        """
        The intent of this method is the following:
            1. Sort the unlabeled data is at the bottom of the list
            2. Sort data to match hypothesis shifts [hypothesis = previous core supports from CSE]
            3. sort core supports to match hypothesis shifts
            4. keep track which instances were originally unlabeled so we know which to use for performance metrics
        This method should sort the data prior to using a SSL classifier
        """
        # sort the hypothesis in descending order 
        sortHypoth, sortID  = hypothesis[hypothesis[:,-1].argsort(kind='heapsort')], np.argsort(hypothesis[:,-1], kind='heapsort')
        # sort data to match hypothesis shift

        # sort labels to support hypothesis shifts

        # sort core supports to match hypothsis shifts

        # keep track of instances originally unlabeled so we know which instances to use for perf metrics
        
    def core_support_extract(self, data_stream):
        """
        This method preprocesses the data before extracting the core supports from the stream
        The intent of this method is to complete the following:
            1. Remove duplicate data instances 
            2. Sort the classes prior to extracting core supports 
            3. Extract the core supports
        """
        # remove duplicate data from stream, from labeles, previous core supports, hypothesis 
        uniq_stream, sortID = np.unique(data_stream), np.argsort(data_stream)
        print(uniq_stream, sortID)

    def run(self):
        # set cores
        self.set_cores()
        with ProcessPoolExecutor(max_workers=self.n_cores):
            self.compose()
            start = self.timestep
            timesteps = self.data.keys()
            if self.verbose == 1:
                print('SSL Classifier:', self.classifier)
            total_time_start = time.time() 
            ts = start
            
            for ts in range(0, len(timesteps)-1):     # iterate through all timesteps from the start to the end of the available data
                self.timestep = ts
                if self.verbose == 1:
                    print("Timestep:",ts)
                t_start = time.time()

                # TODO: copy labels and add to hypthothesis vector 
                if ts == 0: # there will be no core supports @ ts = 0 
                    self.hypothesis[ts] = self.labeled[ts]
                else:
                    self.hypothesis[ts] = np.column_stack((self.labeled[ts], self.core_supports[ts-1]))

                # Receive Unlabeled Data - step 1 - step 3 
                # We have received labeled data at initial time step and then we use the base classifier 
                if ts == 0:
                    if self.classifier == 'QN_S3VM':
                        self.predictions[ts] = self.classify(X_train_l=self.hypothesis[ts], L_train_l=self.labeled[ts+1], X_train_u = self.data[ts], X_test=self.data[ts+1]) 
                    elif self.classifier == 'label_propagation':
                        self.predictions[ts] = self.classify(X_train_l=self.hypothesis[ts], L_train_l=self.labeled[ts+1], X_train_u = self.data[ts], X_test=self.data[ts+1])       
                    
                    # Set D_t or data stream from concatenating the data stream with the predictions - step 3
                    # {xl, yl } = self.labeled[ts = 0]
                    # { xu, hu } = concatenate (self.data[ts][:,:-1], self.predictions[ts] ) 
                    if len(self.data[ts]) > len(self.predictions[ts]):
                        dif_xu_hu = len(self.data[ts]) - len(self.predictions[ts])
                        preds_to_add = []
                        for k in range(dif_xu_hu):
                            randm_list = np.unique(self.predictions[ts])
                            rdm_preds = random.choice(randm_list)
                            preds_to_add = np.append(preds_to_add, rdm_preds)
                        self.predictions[ts] = np.append(self.predictions[ts], preds_to_add)
                        
                    # { xu, hu }
                    xu_hu = np.column_stack((self.data[ts][:,:-1], self.predictions[ts]))
                    # Dt = { xl , yl } U { xu , hu }  
                    self.stream[ts] = np.concatenate((self.labeled[ts], xu_hu)) 
                    # set L^t+1 = 0, Y^t = 0 - step 4 
                    self.labeled[ts+1] = []
                    # steps 5 - 7 as it extracts core supports
                    self.get_core_supports(self.stream[ts])              # create core supports at timestep 0
                    # L^t+1 = L^t+1 
                    self.labeled[ts+1] = self.core_supports[ts]
                    t_end = time.time()         
                    elapsed_time = t_end - t_start
                    self.time_to_predict[ts] = elapsed_time
                    
                    if self.verbose == 1:
                        print("Time to predict: ", elapsed_time, " seconds")

                # after firststep
                if start != ts:
                    t_start = time.time()
                    # self.sort_classify(self.data[ts], self.core_supports[ts-1])
                    self.predictions[ts] = self.classify(X_train_l=self.core_supports[ts-1], L_train_l=self.data[ts], X_train_u=self.data[ts], X_test=self.data[ts+1])
                    # Set D_t or data stream from concatenating the data stream with the predictions - step 3
                    # {xl, yl } = self.labeled[ts = 0]
                    # { xu, hu } = concatenate (self.data[ts][:,:-1], self.predictions[ts] ) 
                    if len(self.data[ts]) > len(self.predictions[ts]):
                        dif_xu_hu = len(self.data[ts]) - len(self.predictions[ts])
                        preds_to_add = []
                        for k in range(dif_xu_hu):
                            randm_list = np.unique(self.predictions[ts])
                            rdm_preds = random.choice(randm_list)
                            preds_to_add = np.append(preds_to_add, rdm_preds)
                        self.predictions[ts] = np.append(self.predictions[ts], preds_to_add)
                    # { xu, hu }
                    if len(self.predictions[ts]) > len(self.data[ts]):
                        differ = len(self.predictions[ts]) - len(self.data[ts])
                        preds = list(self.predictions[ts])
                        for k in range(differ):
                            preds.pop()
                        self.predictions[ts] = np.array(preds)

                    xu_hu = np.column_stack((self.data[ts][:,:-1], self.predictions[ts]))
                    # Dt = { xl , yl } U { xu , hu } 
                    self.labeled[ts] = np.vstack((self.core_supports[ts-1], self.labeled[ts])) # to_cs , labeled
                    self.stream[ts] = np.vstack((self.labeled[ts], xu_hu))
                    # set L^t+1 = 0, Y^t = 0 - step 4 
                    self.labeled[ts+1] = []
                    # steps 5 - 7 as it extracts core supports
                    self.get_core_supports(self.stream[ts])              # create core supports at timestep
                    # L^t+1 = L^t+1 
                    self.labeled[ts+1] = self.core_supports[ts]
                    t_end = time.time()         
                    elapsed_time = t_end - t_start
                    self.time_to_predict[ts] = elapsed_time
                    if self.verbose == 1:
                        print("Time to predict: ", elapsed_time, " seconds")
                
                hypoth_label = np.shape(self.data[ts])[1]-1
                error = self.classification_error(list(self.predictions[ts]), list(self.data[ts+1][:,hypoth_label]))
                
                if len(self.data[ts+1][:,hypoth_label]) > len(self.predictions[ts]):
                    dif_hypoth_learner = len(self.data[ts+1][:,hypoth_label]) - len(self.predictions[ts])
                    ones_to_add = np.ones(dif_hypoth_learner)
                    self.predictions[ts] = np.append(self.predictions[ts], ones_to_add)
                
                self.classifier_accuracy[ts] = (1-error) 
                self.classifier_error[ts] = error

                if self.verbose == 1:
                    print("Classification error: ", error)
                    print("Accuracy: ", 1 - error)
                    # self.plotter()
                    
            total_time_end = time.time()
            self.total_time = total_time_end - total_time_start
            if self.verbose == 1:
                print('Total Time', self.total_time)
            ## Report out
            # classifier_perf = cp.ClassifierMetrics(preds = self.predictions, test= self.data, timestep= ts,\
            #                         method= self.method, classifier= self.classifier, dataset= self.selected_dataset, time_to_predict= self.time_to_predict[ts])
            return self.results_logs()
