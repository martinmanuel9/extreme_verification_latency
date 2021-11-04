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
import pandas as pd
import cse 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# from extreme_verification_latency.models.qns3vm import QN_S3VM
import qns3vm as ssl
import benchmark_datagen as bmdg

class ComposeV1(): 
    def __init__(self, 
                 classifier, 
                 method): 
        """
        Initialization of COMPOSEV1
        """
        self.timestep = 1                  # [INTEGER] The current timestep of the datase
        self.synthetic = 0                  # [INTEGER] 1 Allows synthetic data during cse and {0} does not allow synthetic data
        self.n_cores =  1                   # [INTEGER] Level of feedback displayed during run {default}
        self.verbose = 1                    #    0  : No Information Displayed
                                            #   {1} : Command line progress updates
                                            #    2  : Plots when possible and Command line progress updates
        self.batches = {}                    #  array of timesteps each containing a matrix N instances x D features
        self.labels = []                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = []
        self.hypothesis = {}                #  array of timesteps each containing a N instances x 1 - Classifier hypothesis
        self.core_support = []              #  array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.classifier_func = []
        self.classifier_opts = []           # [Tuple] Tuple of options for the selected classifer in ssl class
        self.learner = {}                   # Object from the ssl 

        self.cse_func = []                  # [STRING] string corresponding to function in cse class
        self.cse_opts = []                  # [Tuple] tuple of options for the selected cse function in cse class

        self.performance = []               # [list] column vector of classifier performances at each timestep
        self.comp_time = []                 # [MATRIX] matrix of computation time for column 1 : ssl classification, column 2 : cse extraction

        self.classifier = classifier
        self.method = method                # not sure what to use for method
        self.dataset = []
        self.figure_xlim = []
        self.figure_ylim = []
        self.data = {}                      # data to set up batches
        self.cse = cse.CSE(self.dataset)

    def compose(self, dataset, verbose):
        """
        Sets COMPOSE dataset and information processing options
        Check if the input parameters are not empty for compose
        This checks if the dataset is empty and checks what option of feedback you want
        Gets dataset and verbose (the command to display options as COMPOSE processes)
        Verbose: 0 : no info is displayed
                 1 : Command Line progress updates
                 2 : Plots when possible and Command Line progress updates
        """
        # sets dataset and verbose
        self.dataset = dataset
        self.verbose = verbose

        # set object displayed info setting
        if self.verbose >= 0 and self.verbose <=2:
           self.verbose = verbose 
        else:
            print("Only 3 options to display information: 0 - No Info ; 1 - Command Line Progress Updates; 2 - Plots when possilbe and Command Line Progress")

        if self.dataset.empty:
            print("Dataset is empty!")
        else:
            self.dataset = dataset

        # set labels and unlabeles and dataset to process
        self.set_data()

        # set batches to account for time steps with matrix of N instanced x  D features
        # self.batches = 
        self.set_batch()
        # set core support 
        self.core_support = self.batches

        # set hypthothesis
        self.hypothesis = [self.batches]

        # set performance
        self.performance = np.zeros(np.shape(self.dataset)[0])

        # set comp time 
        self.comp_time = np.zeros(2, np.shape(self.dataset)[0])

        # set cores
        self.set_cores()

        # set drift window
        self.set_drift_window()
   

    def set_drift_window(self):
        """
        Finds the lower and higher limits to determine drift
        """
        data = self.data
        dataset = self.dataset
        
        # find window where data will drift
        all_data = dataset
        all_data['data'] = pd.Series(data, index=all_data.index)
        
        min_values = all_data.min()
        max_values = all_data.max()
       
        self.figure_xlim = [min_values[0], max_values[0]]
        self.figure_ylim = [min_values[1], max_values[1]]

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
        
        print("Available number of cores:", self.n_cores)
        user_input = input("Enter the amount of cores you wish to begin processing: ")
        self.n_cores = user_input
        print("User selected the following cores to process:", self.n_cores)

    # TODO: need to understand how to set the classifier 
    def set_classifier(self, user_selction, user_options):
        """
        Sets classifier by getting the classifier object from ssl module
        loads classifier based on user input
        The QN_S3VM options are the following:  
        X_l -- patterns of labeled part of the data
        L_l -- labels of labeled part of the data
        X_u -- patterns of unlabeled part of the data
        random_generator -- particular instance of a random_generator (default None)
        kw -- additional parameters for the optimizer
        """
        if not self.learner: 
            # create the ssl
            self.learner = ssl.QN_S3VM()
            
        self.classifier_func = user_selction
        self.classifier_opts = user_options

    def set_cse(self, user_selection, user_options ):
        if not self.cse:
            self.cse= cse.CSE(self.dataset)
            
        self.cse.set_data(self.dataset)

        self.cse_func = user_selection
        self.cse_opts = user_options

        self.cse.set_boundary(self.cse_func)
        self.cse.set_user_opts(self.cse_opts)

        if self.cse_func == 'gmm':
            self.cse.gmm()
        elif self.cse_func == 'parzen':
            self.cse.parzen()
        elif self.cse_func == 'knn':
            self.cse.k_nn()
        elif self.cse_func == 'a_shape':
            self.cse.alpha_shape()
            self.cse.a_shape_compaction()

    def set_batch(self):
        """
        """
        self.batches = {"timestep": self.timestep, "data": self.dataset}
        print(self.batches)

    def set_data(self):
        """
        """
        # import from dataset generation
        avail_data_opts = ['Unimodal','Multimodal','1CDT', '2CDT', 'Unimodal3D','1cht','2cht','4cr','4crev1','4crev2','5cvt','1csurr',
            '4ce1cf','fg2c2d','gears2c2d', 'keystroke', 'Unimodal5D', 'UnitTest']
        print('The following datasets are available:\n' , avail_data_opts)
        user_data_input = input('Enter dataset:')
        data_gen = bmdg.Datagen()
        dataset_gen = data_gen.gen_dataset(user_data_input)

        self.dataset = dataset_gen
        data_id = 0
        labels = []
        unlabeled = []
        # get labeled data/unlabeled data
        for i in range(len(self.dataset)):
            if self.dataset['label'][i] == 1:
                data_id += 1
                self.data = [[data_id], [self.dataset.iloc[i]]]
                labels.append(self.data)
            else:
                data_id += 1
                self.data = [[data_id], [self.dataset.iloc[i]]]
                unlabeled.append(self.data)
        
        self.labels = pd.DataFrame(labels, columns=['data_id', 'data'])
        self.unlabeled = pd.DataFrame(unlabeled, columns=['data_id', 'data'])

    def run(self):
        ts = self.timestep
        for ts in range(len(self.batches)):
            self.timestep = ts
            self.hypothesis[ts] = np.zeros(np.shape(self.batches[ts])[0])




        
    

if __name__ == '__main__':
    COMPV1 = ComposeV1(classifier="qns3vm", method="gmm")
    # COMPV1.compose(data, 1)
    # COMPV1.drift_window()
    # COMPV1.set_cores()
    COMPV1.set_data()
    COMPV1.set_batch()
    COMPV1.run()



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


# class FastCompose(): 
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