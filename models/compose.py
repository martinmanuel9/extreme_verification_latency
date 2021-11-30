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
        self.data = []                    #  array of timesteps each containing a matrix N instances x D features
        self.labels = []                   #  array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = []
        self.hypothesis = []                #  array of timesteps each containing a N instances x 1 - Classifier hypothesis
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
        self.step = 0
        self.cse = cse.CSE(self.dataset)

    def compose(self, verbose):
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
        self.verbose = verbose

        # set object displayed info setting
        if self.verbose >= 0 and self.verbose <=2:
           self.verbose = verbose 
        else:
            print("Only 3 options to display information: 0 - No Info ; 1 - Command Line Progress Updates; 2 - Plots when possilbe and Command Line Progress")


        # set labels and unlabeles and dataset to process
        self.set_data()

        # set core support 
        self.core_support = self.data           # load the dataset in the core support property this includes labeled and unlabeled data from set_data

        # set cores
        self.set_cores()

        # set drift window
        self.set_drift_window()
   
    
    def set_drift_window(self):
        """
        Finds the lower and higher limits to determine drift
        """
        # find window where data will drift
        all_data = self.dataset
        
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
        
        # print("Available number of cores:", self.n_cores)
        # user_input = input("Enter the amount of cores you wish to begin processing: ")
        user_input = self.n_cores
        self.n_cores = user_input
        # print("User selected the following cores to process:", self.n_cores)

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
    #TODO: create batches one timestep per mxn of data 
    def set_data(self):
        """
        Method sets the dataset in its repespective bins, data with timesteps, gets labaled data and unlabeled data from dataset
        """
        # import from dataset generation
        # avail_data_opts = ['Unimodal','Multimodal','1CDT', '2CDT', 'Unimodal3D','1cht','2cht','4cr','4crev1','4crev2','5cvt','1csurr',
        #     '4ce1cf','fg2c2d','gears2c2d', 'keystroke', 'Unimodal5D', 'UnitTest']
        # print('The following datasets are available:\n' , avail_data_opts)
        # user_data_input = input('Enter dataset:')
        user_data_input = 'UnitTest'     # comment out whenever you can run it or determine if I want to run all sets
        data_gen = bmdg.Datagen()
        dataset_gen = data_gen.gen_dataset(user_data_input)

        self.dataset = dataset_gen
        timestep = 0
        data = []
        for i in range(len(self.dataset)):
            timestep += 1
            dat = self.dataset.iloc[i].to_numpy()
            temp_batch = np.append(timestep, dat)
            data.append(temp_batch)
        
        colmn_names = list(self.dataset)
        colmn_names.insert(0,'timestep')
        self.data = pd.DataFrame(data, columns=colmn_names)

        # get labeled data/unlabeled data 
        labels = []
        unlabeled = []
       
        for i in range(len(self.data)):
            if self.dataset['label'][i] == 1:
                lab_dat = self.data.iloc[i].to_numpy()
                labels.append(lab_dat)
            else:
                unlab_dat = self.data.iloc[i].to_numpy()
                unlabeled.append(unlab_dat)
        
        labeled_colmn = list(self.data)

        unlabeled_colm = list(self.data)
        unlabeled_colm[-1] = 'unlabeled'
        
        self.labels = pd.DataFrame(labels, columns=labeled_colmn)
        self.unlabeled = pd.DataFrame(unlabeled, columns=unlabeled_colm)



    def classify(self, ts):
        # sort data in descending so labeled data is at the top and unlabeled follows
        self.hypothesis[ts] = np.sort(self.hypothesis[ts])[::-1]

    def run(self):
        # start = self.timestep
        start = 3
        n = self.core_support[self.core_support['label']==1]['label'].count()
        print(n)

        for ts in range(len(self.data)): # loop from start to end of batches
            self.timestep = ts
            # self.hypothesis[ts] = np.zeros(np.shape(self.data[ts])[0])

            # check if at timestep we have labeled data
            if self.core_support.iloc[ts, -1] == 1: 
                # copy labels into hypothesis
                label = self.core_support.iloc[ts].to_numpy()
                self.hypothesis.append(label)

            
            # # add info for core supports from previous time step
            #TODO:Determine what to do with getting past core-supports. I want to understand why past
            if start != ts:                                              # if not initialized 
                n_cs = self.core_support[ts-1][0].count()                # find number of core supports from previous timesteps
                print(n_cs)
                # self.data[ts] = self.core_support                        # append the current data with the core supports
                
                
                
            #     self.hypothesis[ts] = self.hypthothesis[ts-1][self.core_support[ts-1]]
            #     self.labels[ts] = self.labels[ts-1][self.core_support[ts-1]]
            #     self.core_support[ts] = np.zeros(n_cs)

            # self.step = 1 

            # # plot labeled / unlabled data

            # unlabled_ind = self.classify(ts)




     

if __name__ == '__main__':
    COMPV1 = ComposeV1(classifier="qns3vm", method="gmm")
    COMPV1.compose(1)
    # COMPV1.drift_window()
    # COMPV1.set_cores()
    # COMPV1.set_data()
    COMPV1.set_drift_window()
    COMPV1.run()
    
    # COMPV1.compose(COMPV1.dataset, 1)
    # print(COMPV1.hypothesis)
    
    



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