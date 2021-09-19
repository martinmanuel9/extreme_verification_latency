#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          cse.py
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
    def set_boundary(boundary_selection, opts):

    # # extract core 
    # def extract():

    # # set defualt options
    # def set_default_opts(opts):

    # # set user options
    # def set_user_opts():
    

if __name__ == '__main__' :
    gen_data = bm_gen_data.Datagen.dataset("UnitTest")
    test_cse = CSE()
    # check input
    checkInputCse = test_cse.check_input(3, gen_data)
    print(checkInputCse)

    
