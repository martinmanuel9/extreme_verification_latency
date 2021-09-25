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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class CSE():
    def __init__(self):
        self._synthetic_data = 0
        self._verbose = 1
        self._data = []
        self._nCores = 1 
        self._boundary = []
        self._boundary_opts = []
        self._boundary_data = []
        self._opts = []
        self._n_Instances = []
        self._n_features = []
        self._indices = []
        self._valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']

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

    def set_verbose(self, verbose):

        if verbose > 2:
            verbose = 2
        if verbose < 0:
            verbose = 0
        self._verbose = math.floor(verbose)
        

    # set data by getting inputs from benchmark_datagen
    def set_data(self, data): 
        self._data = data
    
    # Set Boundary Construction Type and Options 
    def set_boundary(self, boundary_selection, opts):
        if not opts:
            self._boundary_opts= []

        self._boundary_opts = []                # clears any past boundary options

        # if boundary_selection not in valid_boundary:
        #     print(boundary_selection, " not in valid boundary.", boundary_selection,
        #         "is an invalid boundary construction method - choose from:  ", valid_boundary)
        
        if boundary_selection in self._valid_boundary:
            self._boundary = boundary_selection
            self.set_defualt_opts()
            if opts:                                            # if user passes in options
                self.set_user_opts(opts)                        # sets user options
        else:
           print(boundary_selection, " not in valid boundary.", boundary_selection,
                "is an invalid boundary construction method - choose from:  ", self._valid_boundary) 

    # Extract Core Supports using the boundary selected
    def indices(self):
        if self._data.empty: 
            print("You must load data before extracting core supports")
            return

        if not self._boundary:
            print('Boundary construction type not set - default classifier and options loaded') 
            # sett gmm as defualt boundary
            self.set_boundary('gmm', ['gmm'])

        # plot labeled and unlabeled data 
        if self._verbose == 2:
            self.plot_cse([])
        
        # run boundary constructor and extract indices of core supporting instances 
        # inds = obj.boundary(obj);
        

        if self._verbose == 2:
            self.plot_cse(self._indices)
        
    def set_defualt_opts(self): 

        # get n features
        df = pd.DataFrame(self._data)
        self._n_features = df.shape[1] - 1
        
        if self._boundary == "a_shape":
            alpha = 2
            p = 2
            self._boundary_opts.append(alpha)
            self._boundary_opts.append(p)
        if self._boundary == "gmm":
            kl = 10
            kh = 10
            p = 0.4
            self._boundary_opts.append(kl)
            self._boundary_opts.append(kh)
            self._boundary_opts.append(p)
        if self._boundary == "knn":
            k = 10
            p = 0.4
            self._boundary_opts.append(k)
            self._boundary_opts.append(p)
        if self._boundary == "parzen":
            win = np.ones((1, self._n_features))
            p = 0.4
            noise_thr = 0
            self._boundary_opts.append(win)
            self._boundary_opts.append(p)
            self._boundary_opts.append(noise_thr)

    def set_user_opts(self, opts):
        # must be an array input 
        if isinstance(opts, list):
            # self._valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']
            # need to determine if user inputs is the actual correct boundary 
            if any(i in self._valid_boundary for i in opts):
                self._boundary_opts = opts
            else:
                print("Warning: Option", self._boundary, "is not a valid option for boundary construction method.")
        else:
            print("Options must be entered as list: [options]")

    def plot_cse(self, indices):
        if not indices:                 # if no indices are specified
            indices = self._data
            color = 'r.'                # red dot marker
        else:
            color = 'k.'                # black dot marker
        
        df = pd.DataFrame(self._data)
        self._n_features = df.shape[1] - 1
        print(self._n_features)
        if self._n_features == 2: 
            print(self._data)              # command line progress
            plt.plot(self._data["column1"], self._data["column2"], color)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Boundary Constructor:" + self._boundary)
            plt.show()
        if self._n_features == 3:
            print(self._data)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = self._data["column1"]
            ys = self._data["column2"]
            zs = self._data["column2"]
            ax.scatter(xs, ys, zs, marker = '.')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            ax.set_title('Boundary Constructor: ' , self._boundary)
            plt.show()

    ## Alpha shape and Dependencies 
    def a_shape(self):
        ashape = self.a_shape
        if not ashape:
            print('No Alpha Shape could be constructed try different alpha or check data')
            return 
        









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

    # test set_boundary
    # test_set_boundary = CSE()
    # check_set_boundary = test_set_boundary.set_boundary('knn', ["knn", 1, [1, 1, 1]])

    # # test extract 
    # test_inds = CSE()
    # check_test_inds = test_inds.inds(gen_data,"",1)

    # # test default options
    # test_defualt_opts = CSE()
    # check_default = test_defualt_opts.set_defualt_opts("knn", gen_data)
    # print(check_default)

    # # test set user opts
    # test_set_user_opts = CSE()
    # check_set_usr_opts = test_set_user_opts.set_user_opts(["fake"])  ## ["fake", 1, [gen_data]] , ["gmm", 1, [gen_data]] 

    # test plot and indices
    # test_plot_ind = CSE()
    # test_plot_ind.set_verbose(2)
    # test_plot_ind.set_data(gen_data)
    # test_plot_ind.set_boundary("a_shape", ["a_shape"])
    # test_plot_ind.indices()