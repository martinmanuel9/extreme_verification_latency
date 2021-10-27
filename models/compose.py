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
from joblib import Parallel, delayed
import multiprocessing
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
        self.data = []            
        self.lables = []                    # [LIST] list array of timesteps each containing a vector N instances x 1 - Correct label
        self.hypothesis =[]                 # [LIST] list array of timesteps each containing a N instances x 1 - Classifier hypothesis
        self.core_support = []              # [LIST] list array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        self.classifier_func = []           # [Tuple] Tuple of string corresponding to classifier in ssl class
        self.classifier_opts = []           # [Tuple] Tuple of options for the selected classifer in ssl class

        self.cse_func = []                  # [STRING] string corresponding to function in cse class
        self.cse_opts = []                  # [Tuple] tuple of options for the selected cse function in cse class

        self.performance = []               # [list] column vector of classifier performances at each timestep
        self.comp_time = []                 # [MATRIX] matrix of computation time for column 1 : ssl classification, column 2 : cse extraction

        self.classifier = classifier
        self.method = method
        self.dataset = []
        self.figure_xlim = []
        self.figure_ylim = []

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

        # set data 
        self.data = np.zeros(np.shape(self.dataset)[0])
        # set labels 

        # set core support 

        # set hypothesis

        # set performance

        # set comp_time


    def drift_window(self):
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
        print(num_cores)
        if self.n_cores > num_cores:
            print("You do not have enough cores on this machine. Cores have to be set to ", num_cores)
            self.n_cores = num_cores                   # sets number of cores to available 
        else:
            self.n_cores = num_cores                   # original number of cores to 1
        
        print(self.n_cores)
        process_features = Parallel(n_jobs=num_cores)(delayed(self.dataset)(i) for i in self.dataset)

    def set_classifier(self, user_selction, user_options, *args):
        """
        Sets classifier by getting the classifier object from ssl module
        loads classifier based on user input
        """
        max_args = 4
        # first check if user input a learner else create learner - ssl(0)
        if not self._learner:                                       # if we do not get the learner object from ssl 
            self.learner = ssl(0)
            # need to get call ssl class and set the data to load it to the learner
            set_data = ssl.set_data(self.data, self.timestep)     # create ssl(0) as learner 
            self.learner                                           # load first batch of data into learner object
        
        if len(*args) < max_args:
            self._learner = self.learner 
        else:
            self.set_classifier = self.set_classifier(learner, labels, timestep, data)

        self.classifer_func = self.set_classifier(learner, labels, timestep, data)

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