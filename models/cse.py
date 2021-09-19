#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          cse.py - core support extraction (CSE)
Author:             Martin Manuel Lopez
Advisor:            Dr. Gregory Ditzler
Creation:           09/18/2021
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
from pandas.core.frame import DataFrame
import benchmark_datagen as bm_gen_data

class CSE():
    def __init__(self):
        _synthetic_data = 0
        _verbose = 1
        _data = []
        _nCores = 1 
        _boundary = []
        _boundary_opts = []
        _boundary_data = []
        _opts = []
        _n_Instances = []
        _n_features = []
        valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']

    # check to see if cse gets right inputs 
    def check_input(self, verbose, synthetic_data):
        self._verbose = verbose
        # set object displayed info setting
        if verbose < 0 or verbose > 2:
            print("Only 3 options to display information: 0 - No Info ; 1 - Command Line Progress Updates; 2 - Plots when possilbe and Command Line Progress")
            return False
        
        self._synthetic_data = synthetic_data

        if synthetic_data.empty:
            print("Dataset is empty!")
            return False

        return True

    # set data by getting inputs from benchmark_datagen
    def set_data(self, data): 
        self._data = data
        return data
    
    # set boundary 
    def set_boundary(self, boundary_selection, opts):
        self._opts = opts
        self._boundary_opts = boundary_selection
        valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']

        if boundary_selection not in valid_boundary:
            print(boundary_selection, " not in valid boundary.", boundary_selection,
                "is an invalid boundary construction method - choose from:  ", valid_boundary)
            
        return boundary_selection
    
    # Extract Core Supports using the boundary selected
    def extract(self):  
        if self._data.empty: 
            print("You must load data before extracting core supports")

        if self._boundary_opts.empty:
            print('Boundary construction type not set - default classifier and options loaded') 
            # sett gmm as defualt boundary
            self.set_boundary('gmm')  

        # set verbose                               

 ## unit tests        
if __name__ == '__main__' :
    gen_data = bm_gen_data.Datagen.dataset("UnitTest")
    
    # # check input
    # test_cse = CSE()
    # checkInputCse = test_cse.check_input(3, gen_data)

    # # test set_data 
    # test_set_data = CSE()
    # check_set_data = test_set_data.set_data(gen_data)
    # print(check_set_data)

    # # test set_boundary
    # test_set_boundary = CSE()
    # check_set_boundary = test_set_boundary.set_boundary('fake_boundary',[])

    
