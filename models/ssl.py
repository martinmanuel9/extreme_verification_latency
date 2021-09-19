#!/usr/bin/env python 
"""
Application:        COMPOSE Framework 
File name:          ssl.py
Author:             Martin Manuel Lopez
Advisor:            Dr. Gregory Ditzler
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

from sys import getsizeof
from numpy.lib.type_check import real
import pandas as pd

class ssl():
    """
    ssl is a class of semi-supervise learning classifiers that may be used in stationary and non-stationary 
    environments. Depending on the classifier chosen a variety of class balancing techniques are available to
    reduce SSL problem of assigning all data to one class. 
    """
    _verbose = 2                #  controls output of screen which plots when possible and renders command line operations
                                #  0 : Suppress all output
                                #  1 : Give text updates to command window
                                #  2 : Plot data when dimensionality allows and give text updates to command window
    _data =[]                   #  N instances x D dimensions : Features of data with labeled data grouped at top of matrix
    _labels = []
    _classifier = []            # Type of SSL classifier to use
    _classifierOpts = []        # Options that correspond with SSL Classifier selected - see individual methods for options
    _balance = []               # Type of class balancing to use
    _balanceOpts = []           # Options that correspond with Balance Function selected - see individual methods for options

    n_features=[]               #  Number of features in data (i.e. dimensionality of data)
    n_classes=[]                # Number of classes different class labels
    n_instances=[]              # Number of instances in data
    n_labeled=[]                # Number of labeled instances in data
    n_unlabeled=[]              # Number of unlabeled instances in data
        
    input_label_format=[]       # Format of labels passed by user - 'integer' OR 'vector'
    input_label_ids=[]          # Records the class identifiers of the labels passed by user
        
    label_format=[]             # Current format of label

    # The cells below contain text strings that match the SSL
    # classifiers and class balance methods available in this object
    # If if other classifiers or balancing methods are added to this
    # class these cells must be modified to include those methods
    valid_classifier = ['s3vm', 'label_prop','label_spread', 'cluster_n_label', 'cluster_n_label_v2', 'label_prop_bal']
    valid_balance = ['none','mass','bid'] #,'reg'}  # may need to delete the reg as idk what it means here

    def set_ssl(self, verbose, *args):
        """
        Sets COMPOSE dataset and information processing options
        Check if the input parameters are not empty for compose
        This checks if the dataset is empty and checks what option of feedback you want
        Gets dataset and verbose (the command to display options as COMPOSE processes)
        Verbose: 0 : no info is displayed
                 1 : Command Line progress updates
                 2 : Plots when possible and Command Line progress updates
        """
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

        return verbose

    def set_data(self, data, labels, *args):
        """
        Load data and labels in ssl 
        """
        # check to see if the size of the data matches the size of the labels
        if getsizeof(data) == getsizeof(labels):
            self._data = data
            self._labels = labels

            # Obtain size information of data
            sizeData = getsizeof(data)                                  # Obtain size info from data
            df_unlabeled = pd.DataFrame.sum(self.n_unlabeled, axis=1)   # sum across each row  
            unlabeled = df_unlabeled['0'].valuecounts()                 # count the instances that have zero which are the unlabeled
            
            self.n_labeled = self.n_instances - self.n_unlabeled        # The remaining instances must be labeled











    
