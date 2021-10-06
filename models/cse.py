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
from numpy.core.numeric import ones, zeros_like
from numpy.lib.function_base import diff
from pandas.core.frame import DataFrame
import benchmark_datagen as bm_gen_data
import numpy as np
import numpy.matlib as npm
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay

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
            setattr(self._boundary_opts, alpha, 2)
            self._boundary_opts.append(p)
            setattr(self._boundary_opts, p, 2)
        if self._boundary == "gmm":
            kl = 10
            kh = 10
            p = 0.4
            self._boundary_opts.append(kl)
            setattr(self._boundary_opts, kl, 10)
            self._boundary_opts.append(kh)
            setattr(self._boundary_opts, kh, 10)
            self._boundary_opts.append(p)
            setattr(self._boundary_opts, p, 0.4)
        if self._boundary == "knn":
            k = 10
            p = 0.4
            self._boundary_opts.append(k)
            setattr(self._boundary_opts, k, 10)
            self._boundary_opts.append(p)
            setattr(self._boundary_opts, p, 0.4)
        if self._boundary == "parzen":
            win = np.ones((1, self._n_features))
            p = 0.4
            noise_thr = 0
            self._boundary_opts.append(win)
            setattr(self._boundary_opts, win, np.ones((1,self._n_features)))
            self._boundary_opts.append(p)
            setattr(self._boundary_opts, p, 0.4)
            self._boundary_opts.append(noise_thr)
            setattr(self._boundary_opts, noise_thr, 0)

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

    
    ## Alpha Shape Construction 
    def ashape(self): 
        if self._n_Instances < self._n_features + 1:            # If a class does not have enought points to construct a tesselation 
            print("Warning::Alpha_Shape::Tesselation_Construction" +
            "Data of dimension", self._n_features, "requires a minimum of", (self._n_features + 1)," unique points.\n" +
            "Alpha shape was not constructed for this data.\n ")
            ashape = []
            return
        else:
            ashape_simplexes = Delaunay(self._data, qhull_options="Qbb Qc Qz Qx Q12")        # set the output simplexes to the Delaunay Triangulation 
                                                                                             # ”Qbb Qc Qz Qx Q12” for ndim > 4 gor qhull options
            ashape_include = np.zeros((pd.DataFrame.shape(ashape_simplexes)))
            for sID in pd.DataFrame.shape(ashape_simplexes):
                # if self._boundary_opts.alpha > calc_radius(obj.data(ashape.simplexes(sID, :), :)) 
                ashape_include[sID] = 1

    def calc_radius(self, points):
        points_Df = pd.DataFrame(points)                # should probably 2D to get points 
        nC = np.shape(points_Df)[1]                     # gets dimension - number of columns
        nR = np.shape(points_Df)[0]                     # gets dimension - number of rows

        # need to create a check to get an [nD+ 1 x nD ] matrix as points 
        if nR < nC:
            print("The dimension of the input points are not square the number of dimension of rows must be 1 more than the dimension of columns")
        
        rM = ((pd.DataFrame(points_Df))**2).sum(axis=1) # first column vector of M which is taking the points raising it by 2 and summing each row for a column vector
        oneColumn = np.array([1]*nR)
        M = np.column_stack((rM.values, points_Df.values, oneColumn))      # create matrix based on teh column of each array 

        # calculate minors
        m = np.zeros(np.shape(M)[0])
        colm_iter = len(m)
        for mID in range(len(m)):                        
            temp = M
            find_Det = np.delete(temp,colm_iter,1)              # deletes columns as it 
            m[mID] = np.linalg.det(find_Det)                    # iterates across the columns to find determinant
            colm_iter -= 1
        
        # calculate center of each dimension
        c = np.zeros(nC)            
        for j in range(len(c)):
            c[j]= (((-1)^(j+1))*0.5*m[j+1]/m[1]) 

        # determine radius 
        radius = math.sqrt(((c-(points_Df[:1].values))**2).sum(axis=1))     #  sqrt(sum(c-first row of points)^2))) 

    ## Alpha shape and Dependencies 
    # def a_shape_contraction(self):
    #     ashape = self.a_shape
    #     if not ashape:
    #         print('No Alpha Shape could be constructed try different alpha or check data')
    #         return 
        
    #     ## Compaction - shrinking of alpha shapes
    #     ashape_N_instances = pd.DataFrame.size(self._data)
    #     ashape_N_core_supports = math.ciel(pd.DataFrame.size(self._data)*self._boundary_opts[1]) # self._boundary_opts[1] is 
    #                                                                                               # p value when ashape is selected
        
    #     ashape_core_support = ones(self._data[0])           # binary vector indicating instance of core support is or not
    #     too_many_core_supports = True                       # Flag denoting if the target number of coresupports has been obtained
        

    #     # begin compaction and remove one layer of simplex at a time
    #     while sum(ashape_core_support) >= ashape_N_core_supports and too_many_core_supports == True:
    #         # find d-1 simplexes 
    #         Tip = npm.repmat(np.nonzero(ashape_include == 1), pd.DataFrame.size(ashape_simplexes[1])) # need to understand .include here

    #         edges = []
    #         nums = ashape.simplexes[1]
    #         for ic in pd.DataFrame.size(ashape.simplexes[1]):
    #             edges = [edges, ashape.simplexes(ashape.include==1, nums(pd.DataFrame.size(ashape.simplexes[1])-1))]
    #             nums = pd.DataFrame(nums).iloc[0, :].shift()        # shifts each row to the right 
            
    #         edges = pd.sort(edges)                          # sort the d-1 simplexes so small node is on left in each row
    #         Sid = edges.ravel().argsort()                   # sort by rows placing copies of d-1 simplexes in adjacent row
    #         Tid = Tid(Sid)                                  # sort the simplex identifiers to match

    #         consec_edges = pd.sum(diff(edges), axis=1)      # find which d-1 simplexes are duplicates - a zero in row N indicates row N and N+1 
    #         consec_edges.ravel().nonzero() + 1 = 0          # throw a zero mark on the subsequent row (N+1) as well
    #         ashape.include(Tid(consec_edges~=0)) = 0



                                          
        

 ## unit tests        
# if __name__ == '__main__' :
    # gen_data = bm_gen_data.Datagen.dataset("UnitTest")
    
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