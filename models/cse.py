#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          cse.py - core support extraction (CSE)
Author:             Martin Manuel Lopez
Creation:           09/18/2021

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

from random import random
from tkinter import N, Y
from numpy.lib.function_base import diff
from numpy.lib.twodim_base import diag
from pandas.core.frame import DataFrame
import benchmark_datagen as bm_gen_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay, distance
# import trimesh
from sklearn.mixture import GaussianMixture as GMM
import util
import knn
import scipy.special as sp
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import warnings

class CSE:
    def __init__(self, data=None, mode=None):
        # self.data must be updated as it is taking data as a dictionary 
        self.data = []
        self.boundary = []
        self.boundary_data = {}
        self.boundary_opts = {} 
        self.valid_boundary = ['a_shape','gmm','parzen','knn','no_cse']
        self.ashape = {} 
        self.ashape_includes = {}
        self.mode = mode
        utility = util.Util()

        if type(data) is dict:
            self.data = utility.makeDataFrame(data)     
            self.N_Instances = np.shape(self.data)[0]
            self.N_features = np.shape(self.data)[1] - 1
            
        elif type(data) is np.ndarray:
            self.data = data
            self.N_Instances = np.shape(self.data)[0]
            self.N_features =  np.shape(self.data)[1] - 1
        elif type(data) is list:
            self.data = data
            self.N_Instances = len(self.data)
            self.N_features = 1 
        else:
            print("Please ensure that you pass in data to extract core supports!")
            exit() 
        
        if self.mode == 'gmm':
            self.set_boundary('gmm')
            self.core_support = self.gmm()
        elif self.mode == 'parzen':
            self.set_boundary(self.mode)
            self.core_support = self.parzen()
        elif self.mode == 'a_shape':
            self.set_boundary(self.mode)
            self.core_support = self.a_shape_compaction()

    # Set Boundary Construction Type and Options 
    def set_boundary(self, boundary_selection, opts=None):
        if not opts:
            self.boundary_opts.clear()

        self.boundary_opts.clear()                          # clears any past boundary options

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
            # set gmm as defualt boundary
            self.set_boundary('gmm', opts='gmm')

        if self.verbose == 2:
            self.plot_cse(self.Indices)
        
    def set_defualt_opts(self): 
        """
        Sets classifier default options
        """    
        if self.boundary == "a_shape":
            self.boundary_opts["alpha"] = 2             # alpha parameter of alpha shape
            self.boundary_opts["p"] = 0.30              # set the percentage of points to be used for core supports 
        if self.boundary == "gmm":
            self.boundary_opts['kl'] = 10               # number of centers to find
            self.boundary_opts['kh'] = 10               
            self.boundary_opts['p'] = 0.4               # set percentage of points to be used for core supports 
            self.boundary_opts.update
        if self.boundary == "knn":
            self.boundary_opts['k'] = 10                # set number of k neighbors to compare
            self.boundary_opts['p'] = 0.4               # set percentage of points to be used for core supports 
        if self.boundary == "parzen":
            self.boundary_opts['win'] = np.ones(self.N_features)
            self.boundary_opts['p'] = 0.4
            self.boundary_opts['noise_thr'] = 0

    def set_user_opts(self, opts):
        # must be an array input 
        if isinstance(opts, list):
            # Determines if user inputs is the actual correct boundary 
            if any(i in self.valid_boundary for i in opts):
                self.boundary_opts = opts
                self.set_defualt_opts() 
            else:
                print("Warning: Option", self.boundary, "is not a valid option for boundary construction method.")
        else:
            print("Options must be entered as list: [options]")

    
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
            simplexes = Delaunay(self.data,qhull_options='Qbb Qc Qz Qx Q12' )   # set the output simplexes to the Delaunay Triangulation - WAS: qhull_options='Qbb Qc Qz Qx Q12'
                                                                                # ”Qbb Qc Qz Qx Q12” for ndim > 4 for qhull options
            self.ashape_includes = np.squeeze(np.zeros((1,len(simplexes.simplices))))

            for sID in range(0,len(simplexes.simplices)):
                if self.boundary_opts['alpha'] > self.calc_radius(simplexes.simplices[sID]):
                    self.ashape_includes[sID] = 1
        self.ashape['simplexes'] = simplexes.simplices                                    # adds tuple to simplexes and includes after Tesselation
        
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
        m = []
        for mID in range(0,np.shape(M)[0]):                        
            temp = M
            find_Det = np.delete(temp,0,axis=1)                # deletes columns as it 
            (a,b) = find_Det.shape
            if a>b:
                padding = ((0,0),(0,a-b))
            else:
                padding = ((0, b-a), (0,0))
            find_Det = np.pad(find_Det, padding, mode = 'constant', constant_values=1)

            m.append(np.linalg.det(find_Det))                   # iterates across the columns to find determinant
            
        # calculate center of each dimension
        c = []
        for j in range(0,nC):
            if m[0] != 0:
                c.append( ((-1)^(j+1)) * 0.5 * m[j+1]/m[0] )
            else:
                c.append( ((-1)^(j+1)) * 0.5 * (m[j+1]/1) )
        
        # determine radius 
        return math.sqrt(((c-(points_Df[:1].values))**2).sum(axis=1))     #  sqrt(sum(c-first row of points)^2))) 
        
    ## alpha shape compaction
    def a_shape_compaction(self):
        self.alpha_shape()      # construct alpha shape
        if not self.ashape['simplexes'].any():
            print('No Alpha Shape could be constructed try different alpha or check data')
            return 
        
        ## Compaction - shrinking of alpha shapes -- referred to as ONION Method -- peels layers
        self.ashape['N_start_instances'] = np.shape(self.data)[0] 
        self.ashape['N_core_supports'] = math.ceil((np.shape(self.data)[0]) * self.boundary_opts['p'])  # deseired core supports  
        # may not be needed
        self.ashape['core_support'] = np.array(self.data)
        # self.ashape['core_support'] = np.ones(self.ashape['N_start_instances'])  # binary vector indicating instance of core support is or not
        too_many_core_supports = True                                  # Flag denoting if the target number of coresupports has been obtained
        
        # Remove layers and compactions
        while len(self.ashape['core_support']) >= self.ashape['N_core_supports'] and too_many_core_supports == True:
            # find d-1 simplexes
            if not self.ashape_includes.all():
                self.ashape_includes[0] = 1 
            indx = np.squeeze(np.argwhere(self.ashape_includes == 1))
            simpx_shape = np.shape(self.ashape['simplexes'])[1]
            Tid = np.tile(indx , (simpx_shape , 1))
            edges = np.zeros((1, np.shape(self.ashape['simplexes'])[1]))
            nums = []
            sortID = np.squeeze(np.argwhere(self.ashape_includes == 1))
            for i in range(np.shape(self.ashape['simplexes'])[1]):
                nums.append(i)
            
            for ic in range(np.shape(self.ashape['simplexes'])[1]):
                edges = np.vstack((edges, self.ashape['simplexes'][sortID]))
                nums = nums[-1:] + nums[:-1]    # shifts each row to the right

            edges = np.delete(edges, 0, axis=0)
            num = np.shape(self.ashape['simplexes'])[1]-1
            edges = edges[:,:num]
            
            edges = np.sort(edges, axis=1)  # sort the d-1 simplexes so small node is on left in each row
            edges, Sid = np.sort(edges, kind='heapsort', axis=0) , np.argsort(edges, kind='heapsort', axis=0)   # sort rows of d-1 simplexes in adjacent row 
            Sid = Sid[:,0]
            sorter = []
            for i in Sid:
                if i >= len(Tid):
                    pass
                else:
                    sorter.append(i)
            Tid = Tid[sorter]                                          # sort the simplex identifiers to match 
            consec_edges = np.sum(diff(edges,n=1, axis=0), axis=1)     # find which d-1 simplexes are duplicates - a zero in row N indicates row N and N+1 
            consec_edge_indx = np.argwhere(consec_edges == 0)
            consec_edges = np.squeeze(consec_edges[consec_edge_indx])
            consec_edges = np.append(consec_edges, 0) # throw a zero mark on the subsequent row (N+1) as well
            indx = Tid[np.argwhere(consec_edges != 0)]
            self.ashape_includes[indx] = 0
            # determine how many points are remaining 
            points_remaining = np.unique(self.ashape['simplexes'][np.argwhere(self.ashape_includes == 1)])
            difference = np.setdiff1d(np.arange(self.ashape['N_start_instances']), points_remaining)
            if len(points_remaining) >= self.ashape['N_core_supports']:
                self.ashape['core_support'][difference] = 0
            else:
                too_many_core_supports = False
        # return core supports 
        indices= np.squeeze(np.argwhere(self.ashape['core_support'] == 1))
        indices = indices[:,0]
        support_indices = self.ashape['core_support'][indices]
        return support_indices
    
    ## GMM using for COMPOSE
    def gmm(self):
        x_ul = self.data
        core_support_cutoff = math.ceil(self.N_Instances * self.boundary_opts['p'])
        BIC = []    #np.zeros(self.boundary_opts['kh'] - self.boundary_opts['kl'] + 1)   # Bayesian Info Criterion
        GM = {}
        preds = {}
        if self.boundary_opts['kl'] > self.boundary_opts['kh'] or self.boundary_opts['kl'] < 0:
            print('the lower bound of k (kl) needs to be set less or equal to the upper bound of k (kh), k must be a positive number')
        
        # remove infs and NaN
        x_ul_df = pd.DataFrame(x_ul)
        x_ul_df = x_ul_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        
        # creates Gaussian Mixutre Model (GMM)
        for i in range(1, self.boundary_opts['kl']+1): 
            if len(x_ul_df) < i:
                break
            else:           
                GM[i] = GMM(n_components = i ).fit(x_ul_df)
            BIC.append(GM[i].bic(x_ul_df))
        
        # Plots GMM
        # plt.scatter(x_ul[:,0], x_ul[:,1], label='Stream @ current timestep') 
        # plt.scatter(y[0,:], y[1,:], c="orange", zorder=10, s=100, label="Train Data from timestep+1")
        # plt.legend()
        # plt.plot()
        # plt.show()
        
        temp = self.boundary_opts['kl'] - 1
        minBIC, bicIX = np.min(BIC), np.argmin(BIC)       # gets the index of minimal BIC
        numComponents = bicIX                             # lowest BIC score becomes numComponets  
        
        # # need to calculate the Mahalanobis Distance for GMM
        get_MD = util.Util(data=x_ul_df)
        
        GM_means = []
        GM_cov = []
        if numComponents <= 1:
            numComponents = 2
        
        GM_means = GM[numComponents].means_
        GM_means = GM_means.reshape((-1, np.shape(x_ul_df)[1]))
        GM_cov = GM[numComponents].covariances_
        GM_cov = GM_cov.reshape((-1, np.shape(x_ul_df)[1]))

        # needs to return the squared Mahalanobis Distance of each observation in x to the reference samples in data
        x_ul_df['mahalanobis'] = get_MD.MahalanobisDistance( x= x_ul_df , data= GM_means, cov=GM_cov)           # x= observations, data=distribution
        x_ul_df = x_ul_df.sort_values(by='mahalanobis')
        
        support_indices = x_ul_df.loc[:, x_ul_df.columns != 'mahalanobis']  # get all but mahalanobis distance
        support_indices = support_indices.loc[:core_support_cutoff]
        self.boundary_data['BIC'] = BIC
        return support_indices.to_numpy()
    
    # Parzen Window Clustering
    def parzen(self):
        core_support_cutoff = math.floor(self.N_Instances * self.boundary_opts['p'])
        data = self.data
        r = data.shape[0]
        ur = data.shape[0]
        uc = data.shape[1]

        scores = []
        
        for i in range(r):
            x_center = np.array(data.iloc[i]) 
            # box windows
            box_min = np.tile(x_center - self.boundary_opts['win']/2, (ur,1))
            box_max = np.tile(x_center + self.boundary_opts['win']/2, (ur,1))
        
            # find unlabeled
            x_in = np.array(data[np.sum((np.logical_and((data >= box_min), (data <= box_max))), axis=1)/uc==1])  
            n_in = np.shape(x_in)[0]
            
            if n_in > (self.boundary_opts['noise_thr'] * ur):
                sig = diag(self.boundary_opts['win']/2 ** 2)
                utility = util.Util(x_in)
                norm_euc = utility.quickMahal(x_in, x_center, sig)
                ul_dist_sum = np.mean(math.exp(-4*norm_euc))
            else:
                ul_dist_sum = 0

            scores.append(ul_dist_sum)
            
        sortMahal = np.sort(scores)[::-1]       # sort in descending order
        IX = np.where(sortMahal)
        support_indices = sortMahal[:core_support_cutoff]
        return support_indices

    ## KNN clustering
    def k_nn(self):
        
        core_support_cutoff = math.floor(self.N_Instances * self.boundary_opts['p'])
        
        kn = knn.KNN(self.data, 5)
        neighbors_dist = kn.knn_run('knn_dist')

        neighbors_dist = np.array(neighbors_dist)
        sort_neighbors = np.sort(neighbors_dist)
        return sort_neighbors
    
    def core_support_extract(self):
        return self.core_support
