#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          util.py 
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

from scipy.spatial.distance import mahalanobis
import benchmark_datagen as bm_gen_dat
import pandas as pd 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
from sklearn.covariance import EmpiricalCovariance, MinCovDet

class Util:
    def __init__(self, data) -> None:
        self.data = pd.DataFrame(data)

    def MahalanobisDistance(self, cov=None, data=None):
        """Compute the Mahalanobis Distance between each row of x and the data  
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """

        data = self.data

        colmn_mean = data.mean()
        x_mu = []
        for i in range(np.shape(data)[0]):
            x_mu.append(data.iloc[i] - colmn_mean)

        x_minus_mean = np.array(x_mu)
        if not cov:
            cov = np.cov(data.values.T)
        
        # print(cov)
        
        inv_cov = sp.linalg.pinv(cov)
        # print(inv_covmat)
        left_term = np.dot(x_minus_mean, inv_cov)
        mahalDist = np.dot(left_term,x_minus_mean.T)
        return mahalDist.diagonal()

if __name__ == '__main__':
    gen_data = bm_gen_dat.Datagen.dataset("UnitTest")
    util = Util(gen_data)
    gen_data['mahalanobis'] = util.MahalanobisDistance()
    print(gen_data.head())
