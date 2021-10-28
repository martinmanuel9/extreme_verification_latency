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
from extreme_verification_latency.models.qns3vm import QN_S3VM
import qns3vm as ssl
import benchmark_datagen as bmdg

class ComposeV1(): 
    def __init__(self, 
                 classifier, 
                 method): 
        """
        Initialization of COMPOSEV1
        """
        self.timestep = 1                   # [INTEGER] The current timestep of the datase
        self.synthetic = 0                  # [INTEGER] 1 Allows synthetic data during cse and {0} does not allow synthetic data
        self.n_cores =  1                   # [INTEGER] Level of feedback displayed during run {default}
        self.verbose = 1                    #    0  : No Information Displayed
                                        #   {1} : Command line progress updates
                                        #    2  : Plots when possible and Command line progress updates
        self.data = {}           
        self.labels = []                    # [LIST] list array of timesteps each containing a vector N instances x 1 - Correct label
        self.unlabeled = []
        self.hypothesis =[]                 # [LIST] list array of timesteps each containing a N instances x 1 - Classifier hypothesis
        self.core_support = []              # [LIST] list array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.classifier_func = []
        self.classifier_opts = []           # [Tuple] Tuple of options for the selected classifer in ssl class
        self.learner = {}

        self.cse_func = []                  # [STRING] string corresponding to function in cse class
        self.cse_opts = []                  # [Tuple] tuple of options for the selected cse function in cse class

        self.performance = []               # [list] column vector of classifier performances at each timestep
        self.comp_time = []                 # [MATRIX] matrix of computation time for column 1 : ssl classification, column 2 : cse extraction

        self.classifier = classifier
        self.method = method
        self.dataset = []
        self.figure_xlim = []
        self.figure_ylim = []
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

        # set timesteps (data is the timesteps)
        self.data = dataset
        
        # set labels 
        self.labels = self.dataset['label']

        # set core support 
        self.core_support = self.dataset

        # set hypthothesis
        self.hypothesis = []     

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

    def set_classifier(self, user_selction, user_options):
        """
        Sets classifier by getting the classifier object from ssl module
        loads classifier based on user input
        """
        if not self.learner: 
            self.classifier = ssl.QN_S3VM(user_options)
            self.learner = (self.data[self.timestep], self.labels[self.timestep])
            
        self.classifier_func = user_selction
        self.classifier_opts = user_options

    def set_cse(self, user_selection, user_options ):
        if not self.cse_func:
            self.cse_func = cse.CSE(self.dataset)
            
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

    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier

if __name__ == '__main__':
    data = bmdg.Datagen.dataset("UnitTest")
    COMPV1 = ComposeV1(classifier="qns3vm", method="gmm")
    COMPV1.compose(data, 1)
    # COMPV1.drift_window()
    COMPV1.set_cores()




    

class ComposeV2(): 
    """
    """
    def __init__(self, 
                 classifier, 
                 method): 
        self.classifier = classifier 
    
    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier


class FastCompose(): 
    """
    """
    def __init__(self, 
                 classifier, 
                 method): 
        self.classifier = classifier 

    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier