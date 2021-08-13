#!/usr/bin/env python 

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
from scipy import stats
from tqdm import tqdm

from sklearn.cluster import KMeans


class APT(): 
    def __init__(self, 
                 Xinit,
                 Yinit,
                 Kclusters:int=10,
                 resample:bool=True,   
                 T:int=50): 
        """
        """
        # total number of times we are going to run an experiment with .run()
        self.T = T 
        # number of unique classes in the data 
        self.nclasses = len(np.unique(Yinit))
        #
        self.resample=resample
        # set the intial data 
        self.Xinit = Xinit
        self.Yinit = Yinit
        # intialize the cluster model  
        self.Kclusers = Kclusters
        self.class_cluster = np.zeros((self.Kclusers,))
        self.M = len(Yinit)
        self._initialize()
        
    def _initialize(self): 
        """
        """
        # run the clustering algorithm on the training data then find the cluster 
        # assignment for each of the samples in the training data 
        self.cluster = KMeans(n_clusters=self.Kclusers).fit(self.Xinit)
        labels = self.cluster.predict(self.Xinit)

        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class. 
        for i in range(self.Kclusers): 
            yhat = self.Yinit[labels==i]
            mode_val,_ = stats.mode(yhat)
            self.class_cluster[i] = mode_val[0]


    
    def run(self, Xts, Yts): 
        """
        """
        self.T = np.min([self.T, len(Xts)])
        N = len(Xts[0])
        
        # check lens of the data 
        if self.M != N: 
            raise ValueError('N and M must be the same size')
        
        # run the experiment 
        for t in range(self.T-1):
            # get the data from time T and resample if required 
            Xt, Yt = Xts[t], Yts[t]
            if self.resample: 
                ii = np.random.randint(0, N, N)
                Xt, Yt = Xt[ii], Yt[ii]
            
            # step 4: associate each new instance to one previous example
            sample_assignment = np.zeros((N,))
            for n in range(N): 
                sample_assignment[n] = int(np.argmin(np.linalg.norm(Xt[n] - self.Xinit, axis=1)))
            
            # step 5: Compute instance-to-exemplar correspondence
            #yhat = Yt[sample_assignment]
            print(t, len(Yt), len(sample_assignment))
            print(Yt[sample_assignment[0]])
            # step 6: Pass the cluster assignment from the example to their 
            # assigned instances to achieve instance-to-cluster assignment 
            self.cluster = KMeans(n_clusters=self.Kclusers, init=self.cluster.cluster_centers_).fit(Xt)
            # step 7: pass the class of an example  