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
from numpy.lib.function_base import append
import pandas as pd
import cse 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import qns3vm as ssl
import benchmark_datagen as bmdg

class FastCOMPOSE: 
    def __init__(self, 
                 classifier = 'QN_S3VM', 
                 method= 'gmm',
                 verbose = 1, 
                 dataset = None): 
        """
        Initialization of Fast COMPOSE
        """


        self.timestep = 0                   # The current timestep of the datase
        self.synthetic = 0                  # 1 Allows synthetic data during cse and {0} does not allow synthetic data
        self.n_cores =  1                   # Level of feedback displayed during run {default}
        self.verbose = verbose              #    0  : No Information Displayed
                                            #    1  : Command line progress updates - {default}
                                            #    2  : Plots when possible and Command line progress updates

        self.data = {}                      #  array of timesteps each containing a matrix N instances x D features
        self.labeled = {}                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = {}
        self.hypothesis = []                #  array of timesteps each containing a N instances x 1 - Classifier hypothesis
        self.core_supports = []            #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        # self.learner = {}                 #  Object from the ssl 

        # self.cse_func = []                  # corresponding to function in cse class -- no longer needed method will takes it cse method place
        self.cse_opts = []                  # options for the selected cse function in cse class

        # self.performance = []               # classifier performances at each timestep
        # self.comp_time = []                 # matrix of computation time for column 1 : ssl classification, column 2 : cse extraction

        self.classifier = classifier
        self.method = method                # not sure what to use for method
        self.dataset = []
        self.figure_xlim = []
        self.figure_ylim = []
        self.step = 0
        self.cse = cse.CSE()
        
        
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
        Verbose: 0 : no info is displayed
                 1 : Command Line progress updates
                 2 : Plots when possible and Command Line progress updates
        """

        # set cores
        self.set_cores()

        # set labels and unlabeles and dataset to process
        self.set_data()

        # set drift window
        self.set_drift_window()

        # set core support 
        # self.core_support = self.data     # load the dataset in the core support property this includes labeled and unlabeled data from set_data

        # set core supports
        # self.set_core_supports()
   
    
    def set_drift_window(self):
        """
        Finds the lower and higher limits to determine drift
        Initial assumption is based on dataset min/max
        """
        self.figure_xlim = np.amin(self.dataset)
        self.figure_ylim = np.amax(self.dataset)
        print("Drift window:" , [self.figure_xlim, self.figure_ylim])

    def set_cores(self):
        """
        Establishes number of cores to conduct parallel processing
        """
        num_cores = multiprocessing.cpu_count()         # determines number of cores

        if self.n_cores > num_cores:
            print("You do not have enough cores on this machine. Cores have to be set to ", num_cores)
            self.n_cores = num_cores                   # sets number of cores to available 
        else:
            self.n_cores = num_cores                   # original number of cores to 1
        
        # print("Available number of cores:", self.n_cores)
        # user_input = input("Enter the amount of cores you wish to begin processing: ")
        user_input = self.n_cores
        self.n_cores = user_input
        print("Num of cores:", self.n_cores)
        # print("User selected the following cores to process:", self.n_cores)

    def set_core_supports(self):
        """
        Method provides core supports based on desired core support extraction.
        Available Core Support Extraction includes: 
        GMM, Parzen Window, KNN, and Alpha Shape Core Supports
        """

        # construct cse if not done before
        if not self.cse:
            self.cse= cse.CSE(data=self.data)

        self.cse = cse.CSE(data=self.data)

        if self.method == 'gmm':
            self.cse.set_boundary(self.method)
            self.core_supports = self.cse.gmm()
        elif self.method == 'parzen':
            self.cse.set_boundary(self.method)
            self.core_supports = self.cse.parzen()
        elif self.method == 'knn':
            self.cse.set_boundary(self.method)
            self.core_supports = self.cse.k_nn()
        elif self.method == 'a_shape':
            self.cse.set_boundary(self.method)
            self.cse.alpha_shape()
            self.core_supports = self.cse.a_shape_compaction()

    def set_data(self):
        """
        Method sets the dataset in its repespective bins, data with timesteps, gets labaled data and unlabeled data from dataset
        """
        if not self.dataset:
            avail_data_opts = ['Unimodal','Multimodal','1CDT', '2CDT', 'Unimodal3D','1cht','2cht','4cr','4crev1','4crev2','5cvt','1csurr',
                '4ce1cf','fg2c2d','gears2c2d', 'keystroke', 'Unimodal5D', 'UnitTest']
            print('The following datasets are available:\n' , avail_data_opts)
            user_data_input = input('Enter dataset:')
        
        # user_data_input = 'UnitTest'     # comment out whenever you can run it or determine if I want to run all sets
        data_gen = bmdg.Datagen()
        dataset_gen = data_gen.gen_dataset(user_data_input)
        self.dataset = dataset_gen              
        print("Dataset:", user_data_input)
        

        ## set a self.data dictionary for each time step 
        ## self.dataset[0][i] loop the arrays and append them to dictionary
        for i in range(0, len(self.dataset[0])):
            self.timestep += 1
            self.data[self.timestep] = self.dataset[0][i]

        
        # filter out labeled and unlabeled from of each timestep
        for i in self.data:
            len_of_batch = len(self.data[i])
            label_batch = []
            unlabeled_batch = []            
            for j in range(0, len_of_batch - 1):
                if self.data[i][j][2] == 1:
                    # print(self.data[i][j])
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

                
        
    def classify(self):
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
        pass

    def run(self):
        pass


if __name__ == '__main__':
    fastcompose_test = FastCOMPOSE(classifier="qns3vm", method="gmm")
    fastcompose_test.compose()
    
    
    

    
    
# class ComposeV2(): 
#     """
#     """
#     def __init__(self, 
#                  classifier, 
#                  method): 
#         self.classifier = classifier 
    
#     def run(self, Xt, Yt, Ut): 
#         """
#         """
#         self.classifier


# class ComposeV1(): 
#     """
#     """
#     def __init__(self, 
#                  classifier, 
#                  method): 
#         self.classifier = classifier 

#     def run(self, Xt, Yt, Ut): 
#         """
#         """
#         self.classifier