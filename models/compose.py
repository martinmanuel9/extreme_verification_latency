#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          compose.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
PhD Advisor: Dr. Gregory Ditzler
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

class COMPOSE: 
    def __init__(self, 
                classifier = 'QN_S3VM', 
                method= 'gmm',
                verbose = 1,
                num_cores = 0.8, 
                selected_dataset = 'UG_2C_2D'): 
        """
        Initialization of Fast COMPOSE
        """


        self.timestep = 1                   # The current timestep of the datase
        self.synthetic = 0                  # 1 Allows synthetic data during cse and {0} does not allow synthetic data
        self.n_cores =  num_cores                   # Level of feedback displayed during run {default}
        self.verbose = verbose              #    0  : No Information Displayed
                                            #    1  : Command line progress updates - {default}
                                            #    2  : Plots when possible and Command line progress updates

        self.data = {}                      #  array of timesteps each containing a matrix N instances x D features
        self.labeled = {}                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = {}
        self.hypothesis = {}                #  array of timesteps each containing a N instances x 1 - Classifier hypothesis
        self.core_supports = {}             #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.num_cs = {}                    #  number of core supports 
        self.total_time = {}
        self.cse_opts = []                  # options for the selected cse function in cse class
        self.selected_dataset = selected_dataset
        self.classifier = classifier
        self.method = method                # not sure what to use for method
        self.dataset = selected_dataset
        self.figure_xlim = []
        self.figure_ylim = []
        self.step = 0 
        self.learner = {}
        self.classifier_accuracy = {}
        self.classifier_error = {}
        self.time_to_predict = {}
        self.user_data_input = {}
        self.avg_results = {}
        self.avg_results_dict = {}
        


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

    def get_core_supports(self, input_data = None):
        """
        Method provides core supports based on desired core support extraction.
        Available Core Support Extraction includes: 
        GMM, Parzen Window, KNN, and Alpha Shape Core Supports
        """

        self.cse = cse.CSE(data=input_data)           # gets core support based on first timestep

        if self.method == 'gmm':
            self.cse.set_boundary(self.method)
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
            ts += 1
            self.data[ts] = self.dataset[0][i]
        
        
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

    def classify(self, X_train_l, L_train_l, X_train_u, X_test, L_test):
        """
        Available classifiers : 'knn',  'QN_S3VM'

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
                L_l_train.append(add)
            L_train_l = L_l_train

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
        avg_results_df = pd.DataFrame({'Dataset': [self.selected_dataset], 'Classifier': [self.classifier],'Method': [self.method], 'Avg_Error': [avg_error], 'Avg_Accuracy': [avg_accuracy], 'Avg_Exec_time': [avg_exec_time]}, 
                            columns=['Dataset','Classifier','Method','Avg_Error', 'Avg_Accuracy', 'Avg_Exec_Time'])
        self.avg_results_dict['Dataset'] = self.selected_dataset
        self.avg_results_dict['Classifier'] = self.classifier
        self.avg_results_dict['Method'] = self.method
        self.avg_results_dict['Avg_Error'] = avg_error
        self.avg_results_dict['Avg_Accuracy'] = avg_accuracy
        self.avg_results_dict['Avg_Exec_Time'] = avg_exec_time
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
            for ts in range(1, len(timesteps)):                    # iterate through all timesteps from the start to the end of the available data
                self.timestep = ts
                # add core supports to hypothesis
                self.get_core_supports(self.data[ts])              # create core supports at timestep
                # if there is labeled data then copy labeles to hypothesis
                if ts in self.labeled:
                    self.hypothesis[ts] = self.labeled[ts]         # copy labels onto the hypthosis if they exist
                else:
                    self.hypothesis[ts] = self.labeled[ts-1]
                if self.verbose == 1:
                    print("Timestep:",ts)
                self.step = 1
                # first round with labeled data
                if ts == 1:
                    t_start = time.time()
                    if np.shape(self.data[ts]) > np.shape(self.labeled[ts]):
                        data_val = int(np.shape(self.data[ts])[0] - np.shape(self.labeled[ts])[0])
                        data_array = list(self.data[ts])
                        for i in range(0, data_val):
                            data_array.pop()
                        self.data[ts] = np.array(data_array)
                    if self.classifier == 'QN_S3VM':
                        self.learner[ts] = self.classify(X_train_l=self.labeled[ts], L_train_l=self.labeled[ts], X_train_u = self.unlabeled[ts], X_test=self.labeled[ts+1], L_test=self.hypothesis[ts]) 
                    elif self.classifier == 'label_propagation':
                        self.learner[ts] = self.classify(X_train_l=self.labeled[ts], L_train_l=self.labeled[ts], X_train_u = self.unlabeled[ts], X_test=self.labeled[ts+1], L_test=self.hypothesis[ts])        
                    t_end = time.time()
                    elapsed_time = t_end - t_start
                    self.time_to_predict[ts] = elapsed_time
                    if self.verbose == 1:
                        print("Time to predict: ", elapsed_time, " seconds")
                
                # after firststep
                if start != ts:
                    # append core supports to hypothesis 
                    # add labeled and hypothesis with core supports
                    if self.method == 'gmm': 
                        self.core_supports[ts-1] = np.reshape(self.core_supports[ts-1], (np.shape(self.core_supports[ts-1])[0], 1))
                        self.core_supports[ts] = np.reshape(self.core_supports[ts], (np.shape(self.core_supports[ts])[0], 1))
                    
                    if np.shape(self.core_supports[ts-1])[1] <= np.shape(self.core_supports[ts])[1]:
                        to_cs = np.zeros((len(self.core_supports[ts]),(np.shape(self.hypothesis[ts-1])[1] - 1)))
                        self.core_supports[ts-1] = np.column_stack((to_cs, self.core_supports[ts-1]))

                    if np.shape(self.hypothesis[ts-1])[1] < np.shape(self.core_supports[ts-1])[1]:
                        to_hs = np.zeros((len(self.hypothesis[ts-1]),(np.shape(self.core_supports[ts-1])[1] - np.shape(self.hypothesis[ts-1])[1])))
                        self.hypothesis[ts-1] = np.column_stack((to_hs, self.hypothesis[ts-1]))
                    self.hypothesis[ts] = np.append(self.hypothesis[ts-1], self.core_supports[ts-1], axis=0)

                    if np.shape(self.labeled[ts-1])[1] < np.shape(self.core_supports[ts-1])[1]:
                        to_ls = np.zeros((len(self.labeled[ts-1]), (np.shape(self.core_supports[ts-1])[1] - np.shape(self.labeled[ts-1])[1])))
                        self.labeled[ts-1] = np.column_stack((to_ls, self.labeled[ts-1]))
                    self.labeled[ts] = np.append(self.labeled[ts-1], self.core_supports[ts-1], axis=0)

                    if np.shape(self.labeled[ts+1])[1] < np.shape(self.core_supports[ts-1])[1]:
                        to_ls = np.zeros((len(self.labeled[ts+1]), (np.shape(self.core_supports[ts-1])[1] - np.shape(self.labeled[ts+1])[1])))
                        self.labeled[ts+1] = np.column_stack((to_ls, self.labeled[ts+1]))
                    
                    if np.shape(self.unlabeled[ts-1])[1] < np.shape(self.labeled[ts-1])[1]:
                        to_uls = np.zeros((len(self.unlabeled[ts-1]), (np.shape(self.labeled[ts-1])[1] - np.shape(self.unlabeled[ts-1])[1])))
                        self.unlabeled[ts-1] = np.column_stack((to_uls, self.unlabeled[ts-1]))
                    # self.unlabeled[ts] = np.append(self.labeled[ts-1], self.core_supports[ts-1], axis=0)

                    if np.shape(self.unlabeled[ts])[1] < np.shape(self.labeled[ts-1])[1]:
                        to_uls = np.zeros((len(self.unlabeled[ts]), (np.shape(self.labeled[ts-1])[1] - np.shape(self.unlabeled[ts])[1])))
                        self.unlabeled[ts] = np.column_stack((to_uls, self.unlabeled[ts]))
                    
                    t_start = time.time()
                    
                    if np.shape(self.labeled[ts]) > np.shape(self.labeled[ts-1]):
                        rows_to_add = int(np.shape(self.labeled[ts])[0] - np.shape(self.labeled[ts-1])[0])
                        self.labeled[ts-1] = np.append(self.labeled[ts-1], np.ones((rows_to_add,np.shape(self.labeled[ts-1])[1])), axis=0)
                    
                    self.learner[ts] = self.classify(X_train_l=self.labeled[ts-1], L_train_l=self.labeled[ts], X_train_u=self.unlabeled[ts], X_test=self.labeled[ts+1], L_test=self.hypothesis[ts])
                    t_end = time.time()
                    elapsed_time = t_end - t_start
                    self.time_to_predict[ts] = elapsed_time
                    if self.verbose == 1:
                        print("Time to predict: ", elapsed_time, " seconds")  
                hypoth_label = np.shape(self.hypothesis[ts])[1]-1
                error = self.classification_error(list(self.learner[ts]), list(self.hypothesis[ts][:,hypoth_label]))
                self.classifier_accuracy[ts] = 1-error * 100
                self.classifier_error[ts] = error
                
                if self.verbose == 1:
                    print("Classification error: ", error)
                    print("Accuracy: ", 1 - error)
                    # self.plotter()
                    
                self.step = 2 
            total_time_end = time.time()
            self.total_time[self.user_data_input] = total_time_end - total_time_start
            if self.verbose == 1:
                print('Performance', self.total_time[self.user_data_input])
            ## Report out
            return self.results_logs()
