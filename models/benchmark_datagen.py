#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          benchmark_datagen.py
Author:             Martin Manuel Lopez
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

from typing import Match
import pandas as pd


class datagen():
    _dataset = []
    _data = []
    _labels=[]

    def dataset(self, data, labels, dataset):
        if data = 'Unimodal':
            # Unimodal
            UG_2C_2D =  pd.read_csv('UG_2C_2D.txt', delimiter="\t")
            l = 1
            m = 1
            limit = 1000
            self._data = data
            self._labels = labels
            self._dataset = dataset
            for i in zip(range(limit), len(UG_2C_2D)): # for loops in 1000s for clustering
                for j in (limit-1):
                    dataset.data[l:,0][m:, 1] = UG_2C_2D[j: 1]      # get first 2 columns of data
                    dataset.labels[l:,0][m:,1] = UG_2C_2D[j:, 2]    # get last column for labels
                    n_pt = 1000
                    # get dataset column first column and use zeroes for the n_pt 
                    