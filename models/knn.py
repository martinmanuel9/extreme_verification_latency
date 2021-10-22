#!/usr/bin/env python 

"""
Application:        COMPOSE Framework - K-Nearest Neighbors Algorithm
File name:          knn.py 
Author:             Martin Manuel Lopez
Creation:           10/20/2021

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

import pandas as pd 
import numpy as np
import scipy as sp
import math

class KNN:
    def __init__(self, data) -> None:
        self.data = pd.DataFrame(data)
        self.N_features = np.shape(self.data)[1]
 
## KNN algorithm
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1) :    
            distance += (row1[i] - row2[i])**2
        return math.sqrt(distance)
        
    def get_nearest_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self, train, test_row, num_neighbors):
        neighbors = self.get_nearest_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction
        
    def k_nearest_neighbors(self, train, test, num_neighbors):
        predictions = list()
        for row in test:
            output = self.predict_classification(train, row, num_neighbors)
            predictions.append(output)
        return(predictions)