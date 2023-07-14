#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency 
File name:          evl_util.py 
Author:             Martin Manuel Lopez
Creation:           09/18/2021

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
PhD Advisor: Dr. Gregory Ditzler and Dr. Salim Hariri 
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

from numpy import linalg
from numpy.lib import utils
from numpy.lib.twodim_base import diag
from pandas.core.tools.datetimes import DatetimeScalarOrArrayConvertible
from pandas.io.formats.format import return_docstring
from scipy.spatial import distance
from scipy.spatial.distance import mahalanobis
import benchmark_datagen_old as bm_gen_dat
import pandas as pd 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import random

class Util:
    def __init__(self, data=None) -> None:
        self.data = pd.DataFrame(data)
        self.N_features = np.shape(self.data)[1]

    def MahalanobisDistance(self, x=None, cov=None, data=None):
        """Compute the Mahalanobis Distance between each row of x and the data  
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """       
        x_minus_mean = x - np.mean(data) 
        
        if cov is None:
            cov = np.cov(data.T)
        
        # make cov a square matrix 
        if np.shape(cov)[0] > np.shape(cov)[1]:
            cov = list(cov)
            to_pop = np.shape(cov)[0] - np.shape(cov)[1]
            for k in range(to_pop):
                cov.pop()
        
        cov = np.array(cov)
        inv_cov = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mean, inv_cov)
        mahalDist = np.dot(left_term,x_minus_mean.T)
        return mahalDist.diagonal()

    def quickMahal(self, x, mu, sig):
        mu = np.tile(mu, (np.shape(x)[0], 1))
        x_minus_mu = (x-mu)
        inv_cov = np.linalg.inv(sig)
        left_term = np.dot(x_minus_mu,inv_cov)
        mahal = np.dot(left_term, x_minus_mu.T).diagonal()
        dist = np.sum(mahal)
        return dist

    def makeDataFrame(self, data):
        keys = data.keys()
        df = pd.DataFrame([data]).T
        # df[first column - only colm with data][array in row][row in array][first number of array]
        # df[0][1][j]
        dataReturn = pd.DataFrame()
        for key in keys:
            arrayRow = df[0][key]
            # print(len(arrayRow))              # this takes the first array - index begins at 1 since first timestep is 1
            for j in range(0, len(arrayRow)):
                row = pd.Series(df[0][key][j])
                dataReturn = dataReturn.append(row, ignore_index=True)

        return dataReturn

# if __name__ == '__main__':
#     gen_data = bm_gen_dat.Datagen()
#     data = gen_data.gen_dataset("UnitTest")
#     util = Util()
#     util.makeDataframe(data)
    # util = Util(gen_data)
    ## test Mahalanobis Distance
    # util = Util(gen_data)
    # gen_data['mahalanobis'] = util.MahalanobisDistance()
    # print(gen_data.head())

    ## test quickMahal
    # x_in = [ 2.8958, -7.4953,  1    ]
    # x_in = np.asfarray(x_in)
    # boundary_opts = 3
    # win = np.ones(boundary_opts)
    # util = Util(x_in)
    # gen_data['mahal'] = util.MahalanobisDistance()

    # for i in range(len(gen_data)):
    #     x_center = gen_data.iloc[i]
    #     # x_center = np.asfarray(x_center)
    #     sig = diag(win/2 ** 2)
    #     dist = util.quickMahal(x_in, x_center)
    #     gen_data['quickMahal'] = dist
    
    # print(gen_data.head())