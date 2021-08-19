#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          compose.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021
COMPOSE Origin:     Muhammad Umer and Robi Polikar

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
from joblib import Parallel, delayed
import multiprocessing
import models.ssl as ssl

class ComposeV1(): 
    def __init__(self, 
                 classifier, 
                 method): 
        """
        Initialization of COMPOSEV1
        """
        _timestep = 1                   # [INTEGER] The current timestep of the datase
        _synthetic = 0                  # [INTEGER] 1 Allows synthetic data during cse and {0} does not allow synthetic data
        _n_cores =  1                   # [INTEGER] Level of feedback displayed during run {default}
        _verbose = 1                    #    0  : No Information Displayed
                                        #   {1} : Command line progress updates
                                        #    2  : Plots when possible and Command line progress updates
        _data = []                      # [LIST] list array of timesteps each containing a matrix N instances x D features
        _lables = []                    # [LIST] list array of timesteps each containing a vector N instances x 1 - Correct label
        _hypothesis =[]                 # [LIST] list array of timesteps each containing a N instances x 1 - Classifier hypothesis
        _core_support = []              # [LIST] list array of timesteps each containing a N instances x 1 - binary vector indicating if instance is a core support (1) or not (0)
        _classifier_func = []           # [Tuple] Tuple of string corresponding to classifier in ssl class
        _classifier_opts = []           # [Tuple] Tuple of options for the selected classifer in ssl class

        _cse_func = []                  # [STRING] string corresponding to function in cse class
        _cse_opts = []                  # [Tuple] tuple of options for the selected cse function in cse class

        _performance = []               # [list] column vector of classifier performances at each timestep
        _comp_time = []                 # [MATRIX] matrix of computation time for column 1 : ssl classification, column 2 : cse extraction

        _dataset = []
        _figure_xlim = []
        _figure_ylim = []

        self.classifier = classifier
        self.method = method

    def check_compose_dataset(self, dataset, verbose, *args):
        """
        Sets COMPOSE dataset and information processing options
        Check if the input parameters are not empty for compose
        This checks if the dataset is empty and checks what option of feedback you want
        Gets dataset and verbose (the command to display options as COMPOSE processes)
        Verbose: 0 : no info is displayed
                 1 : Command Line progress updates
                 2 : Plots when possible and Command Line progress updates
        """
        self._dataset = dataset
        self._verbose = verbose

        # need to limit arguements to 2 for dataset and verbose 
        max_args = 2
        try:
            len(*args) <= max_args
        except ValueError:
            print("Number of input parameters must be a min of two. Input valid dataset and valid option to display information")

        # set object displayed info setting
        if self._verbose >= 0 and self._verbose <=2:
           self._verbose = verbose 
        else:
            print("Only 3 options to display information: 0 - No Info ; 1 - Command Line Progress Updates; 2 - Plots when possilbe and Command Line Progress")

        if not self._dataset:
            print("Dataset is empty!")
        else:
            self._dataset = dataset

        return dataset, verbose

    def determine_drift(self, data, dataset):
        """
        Finds the lower and higher limits to determine drift
        """
        self._data = data
        self._data = dataset
        # find window where data will drift
        all_data = np.full_like(data, dataset)
        self._figure_xlim = [min(all_data[:1]) , max(all_data[:1])]
        self._figure_ylim = [min(all_data[:2]), max(all_data[:2])]

    def set_cores(self, cores, dataset):
        """
        Establishes number of cores to conduct parallel processing
        """
        self._dataset = dataset
        self._n_cores = cores
        num_cores = multiprocessing.cpu_count()         # determines number of cores
        if self._n_cores > num_cores:
            print("You do not have enough cores on this machine. Cores have to be set to ", num_cores)
            self._n_cores = num_cores                   # sets number of cores to available 
        else:
            self._n_cores = num_cores                   # original number of cores to 1
        
        process_features = Parallel(n_jobs=num_cores)(delayed(dataset)(i) for i in dataset)

        return process_features

    def set_classifier(self, learner, user_selection, user_opt):
        """
        Sets classifier by getting the classifier object from ssl module
        loads classifier based on user input
        """
        if not learner:         # if we do not get the learner object from ssl 
            self.learner = ssl(0)
            self.learner.set_data()



    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier
    

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