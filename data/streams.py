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


import pandas as pd 
import numpy as np

def stream_file_loader(experiment_name:str='1CDT', 
                       chunk_size:int=500): 
    """read data in from a file stream
    """
    df = pd.read_csv(''.join(['data/files/', experiment_name, '.txt']), header=None)
    X, Y = df.values[:,:-1], df.values[:,-1]
    N = len(Y)

    # set Xinit and Yinit
    Xinit, Yinit = X[:chunk_size,:], Y[:chunk_size]
    Xt, Yt = [], []

    for i in range(chunk_size, N-chunk_size, chunk_size): 
        Xt.append(X[i:i+chunk_size])
        Yt.append(Y[i:i+chunk_size])
    
    return Xinit, Yinit, Xt, Yt


def generate_stream(dataset_name:str='', 
                    dataset_params:dict={'val': None}):
    """
    """
    return None 