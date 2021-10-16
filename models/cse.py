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

from numpy.lib.function_base import diff
from pandas.core.frame import DataFrame
import benchmark_datagen as bm_gen_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay, distance
# import trimesh
from sklearn.mixture import GaussianMixture as GMM


class CSE:
    def __init__(self, data) -> None:
        self.synthetic_data = []
        self.verbose = 1
        self.data = data
        self.boundary = []
        self.boundary_data = {}
        self.boundary_opts = {}                                             # creates dictionary 
        self.N_Instances = np.shape(self.data)[0]
        self.N_features = np.shape(self.data)[1]
        self.valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']
        self.ashape = {}                                                    # dictionary for ashape

    # check to see if cse gets right inputs 
    def check_input(self, verbose, synthetic_data):
        self.verbose = verbose
        # set object displayed info setting
        if verbose < 0 or verbose > 2:
            print("Only 3 options to display information: 0 - No Info ; 1 - Command Line Progress Updates;" 
                    + "2 - Plots when possilbe and Command Line Progress")
            return False
        
        self.synthetic_data = synthetic_data

        if synthetic_data.empty:
            print("Dataset is empty!")
            return False

        return True

    def set_verbose(self, verbose):

        if verbose > 2:
            verbose = 2
        if verbose < 0:
            verbose = 0
        self.verbose = math.floor(verbose)
        

    # set data by getting inputs from benchmark_datagen
    def set_data(self, data): 
        self.data = data
    
    # Set Boundary Construction Type and Options 
    def set_boundary(self, boundary_selection, opts):
        if not opts:
            self.boundary_opts= []

        self.boundary_opts = []                # clears any past boundary options

        # if boundary_selection not in valid_boundary:
        #     print(boundary_selection, " not in valid boundary.", boundary_selection,
        #         "is an invalid boundary construction method - choose from:  ", valid_boundary)
        
        if boundary_selection in self.valid_boundary:
            self.boundary = boundary_selection
            self.set_defualt_opts()
            if opts:                                            # if user passes in options
                self.set_user_opts(opts)                        # sets user options
        else:
           print(boundary_selection, " not in valid boundary.", boundary_selection,
                "is an invalid boundary construction method - choose from:  ", self.valid_boundary) 

    # Extract Core Supports using the boundary selected
    def indices(self):
        if self.data.empty: 
            print("You must load data before extracting core supports")
            return

        if not self.boundary:
            print('Boundary construction type not set - default classifier and options loaded') 
            # sett gmm as defualt boundary
            self.set_boundary('gmm', ['gmm'])

        # plot labeled and unlabeled data 
        if self.verbose == 2:
            self.plot_cse([])
        
        # run boundary constructor and extract indices of core supporting instances 
        # inds = obj.boundary(obj);
        

        if self._verbose == 2:
            self.plot_cse(self.Indices)
        
    def set_defualt_opts(self): 

        # get n features
        df = pd.DataFrame(self.data)
        self.N_features = df.shape[1] - 1
        
        if self.boundary == "a_shape":
            # alpha = 2
            # p = 2
            self.boundary_opts['alpha'] = 2
            self.boundary_opts['p'] = 2
        if self._boundary == "gmm":
            # kl = 10
            # kh = 10
            # p = 0.4
            self.boundary_opts['kl'] = 10
            self.boundary_opts['kh'] = 10
            self.boundary_opts['p'] = 0.4
        if self.boundary == "knn":
            # k = 10
            # p = 0.4
            self.boundary_opts['k'] = 10
            self.boundary_opts['p'] = 0.4
        if self.boundary == "parzen":
            # win = np.ones((np.shape(self.N_features)))
            # p = 0.4
            # noise_thr = 0
            self.boundary_opts['win'] = np.ones((np.shape(self.N_features)[0]))
            self.boundary_opts['p'] = 0.4
            self.boundary_opts['noise_thr'] = 0

    def set_user_opts(self, opts):
        # must be an array input 
        if isinstance(opts, list):
            # self.valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']
            # need to determine if user inputs is the actual correct boundary 
            if any(i in self.valid_boundary for i in opts):
                self.boundary_opts = opts
            else:
                print("Warning: Option", self.boundary, "is not a valid option for boundary construction method.")
        else:
            print("Options must be entered as list: [options]")

    def plot_cse(self, indices):
        if not indices:                 # if no indices are specified
            indices = self.data
            color = 'r.'                # red dot marker
        else:
            color = 'k.'                # black dot marker
        
        df = pd.DataFrame(self.data)
        self.N_features = df.shape[1] - 1
        print(self.N_features)
        if self.N_features == 2: 
            print(self.data)              # command line progress
            plt.plot(self.data["column1"], self.data["column2"], color)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Boundary Constructor:" + self.boundary)
            plt.show()
        if self._n_features == 3:
            print(self.data)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xs = self._data["column1"]
            ys = self._data["column2"]
            zs = self._data["column2"]
            ax.scatter(xs, ys, zs, marker = '.')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            ax.set_title('Boundary Constructor: ' , self.boundary)
            plt.show()
    
    ## Alpha shape and Dependencies Onion method
    def alpha_shape(self):  
        set = np.array(self.data)
        df = pd.DataFrame(self.data)
        self.N_Instances = np.shape(df)[0]
        self.N_features = np.shape(df)[1]

        set_data = [tuple(row) for row in set]
        uniques = np.unique(set_data, axis=0)
        self.data = np.array(uniques)
        
        if self.N_Instances < self.N_features + 1:            # If a class does not have enought points to construct a tesselation 
            print("Warning::Alpha_Shape::Tesselation_Construction" +
            "Data of dimension", self.N_features, "requires a minimum of", (self.N_features + 1)," unique points.\n" +
            "Alpha shape was not constructed for this data.\n ")
            self.ashape = {}                                                         # set output to empty dict
            return                                                                   # returns to calling function
        else:
            simplexes = Delaunay(self.data, qhull_options="Qbb Qc Qz Qx Q12")        # set the output simplexes to the Delaunay Triangulation 
                                                                                     # ”Qbb Qc Qz Qx Q12” for ndim > 4 for qhull options
            includes = np.zeros((np.shape(simplexes)[0]))
            for sID in range(len(simplexes)):
                if self.boundary_opts['alpha'] > self.calc_radius(simplexes[sID,:]):
                    includes[sID] = 1
            
        self.ashape['simplexes'] = simplexes                                    # adds tuple to simplexes and includes after Tesselation
        self.ashape['includes'] = includes
        
        # plot options for a-shape
        # if self.verbose == 2:
            
        #     if self.N_features == 2:
        #         trimesh()
        #     elif self.N_features == 2:
        #         trimesh()

    
    # calculate the radius 
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
    
    ## alpha shape compaction
    def a_shape_compaction(self):
        self.alpha_shape()      # construct alpha shape
        ashape = self.ashape

        if not ashape:
            print('No Alpha Shape could be constructed try different alpha or check data')
            return 
        
        ## missing plot methods 

        ## Compaction - shrinking of alpha shapes -- referred to as ONION Method -- peels layers
        self.ashape['N_start_instances'] = pd.shape(self.data)[0] 
        self.ashape['N_core_supports'] = math.ciel((pd.shape(self.data)[0])*self.boundary_opts['p'])        
        self.ashape['core_support'] = np.ones(np.shape(self.data)[0])  # binary vector indicating instance of core support is or not
        too_many_core_supports = True                                  # Flag denoting if the target number of coresupports has been obtained
        

        # Remove layers and compactions
        while sum(self.ashape['core_support']) >= self.ashape['N_core_supports'] and too_many_core_supports == True:
            # find d-1 simplexes 
            Tip = np.tile(np.argwhere(self.ashape['includes'] == 1), (np.shape(self.ashape['simplexes'])[0],1))

            edges = []
            nums = []
            for i in range(np.shape(self.ashape['simplexes'])[1]):
                nums.append(i)

            for ic in range(pd.shape(self.ashape['simplexes'][1])):  
                edges = [edges, self.ashape['simplexes'][self.ashape['includes'] == 1, (np.shape(self.ashape['simplexes'])[1]-1)]] # need to test this
                nums = pd.DataFrame(nums).iloc[0, :].shift()        # shifts each row to the right MATLAB is circshift
            
            edges = np.sort(edges)                                  # sort the d-1 simplexes so small node is on left in each row
            Sid = edges.ravel().argsort()                           # sort by rows placing copies of d-1 simplexes in adjacent row
            Tid = Tid(Sid)                                          # sort the simplex identifiers to match

            consec_edges = np.sum(diff(edges), axis=1)              # find which d-1 simplexes are duplicates - a zero in row N indicates row N and N+1 
            edge_single_vector = np.ravel(consec_edges)
            non_zero_edge = np.nonzero(edge_single_vector)
            consec_edges[non_zero_edge] = 0                         # throw a zero mark on the subsequent row (N+1) as well
            self.ashape['includes'][Tid[consec_edges!=0]] = 0   
            points_remaining = np.unique(self.ashape['simplexes'][self.ashape['includes']==1])
            if len(points_remaining) >= self.ashape['N_core_supports']:
                set_diff = self.ashape['N_start_instances'].difference(points_remaining)
                for i in range(set_diff):
                    self.ashape['core_support'] = 0
            else:
                too_many_core_supports = False


    ## GMM dependencies 
    def gmm(self):
        x_ul = self.data
        core_support_cutoff = math.ceil(self.N_Instances * self.boundary_opts['p'])
        BIC = []    #np.zeros(self.boundary_opts['kh'] - self.boundary_opts['kl'] + 1)         # Bayesian Info Criterion
        GM ={}
        if self.boundary_opts['kl'] > self.boundary_opts['kh'] or self.boundary_opts['kl'] < 0:
            print('the lower bound of k (kl) needs to be set less or equal to the upper bound of k (kh), k must be a positive number')
        
        if self.boundary_opts['kl'] == self.boundary_opts['kh']:
            gmm_range = self.boundary_opts['kl'] + 1
            for i in range(1,gmm_range):
                GM[i] = GMM(n_components = i).fit(x_ul)
                BIC.append(GM.bic(x_ul))
        else:
            upper_range = self.boundary_opts['kh'] + 1
            for i in range(self.boundary_opts['kl'], upper_range):
                GM[i] = GMM(n_components=i).fit(x_ul)
                BIC.append(GM[i].bic(x_ul))
        
        temp = self.boundary_opts['kl'] - 1
        minBIC = np.amin(BIC)                                         # minimum Baysian Information Criterion (BIC) - used to see if we fit under MLE
        numComponents = BIC.count(minBIC)                
        
        # need to calculate the Mahalanobis Distance for GMM
        D = distance.cdist(GM[temp+numComponents], x_ul, 'mahalanobis', VI=None)    # calculates Mahalanobis Distance - outlier detection

        minMahal = D.min(axis=1)
        I = np.where(D.min(axis=1))
        sortMahal = minMahal.sort()
        # [SortMahal,IX]=sort(MinMahal)
            
        # support_inds = IX(1:CoreSupportCutOff)
        self.boundary_data['BIC']= BIC
        self.boundary_data['num_components'] = numComponents + temp
        self.boundary_data['gmm'].gmm{obj.parent.timestep} = GM{numComponents+temp}


    
 ## unit tests        
if __name__ == '__main__':
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

    # # test plot and indices
    # test_plot_ind = CSE()
    # test_plot_ind.set_verbose(2)
    # test_plot_ind.set_data(gen_data)
    # test_plot_ind.set_boundary("a_shape", ["a_shape"])
    # test_plot_ind.indices()
    
    ## test the compaction and alpha shape
    test_alpha = CSE(gen_data)
    test_alpha.set_data(gen_data)
    test_alpha.alpha_shape()
    test_alpha.a_shape_compaction()

