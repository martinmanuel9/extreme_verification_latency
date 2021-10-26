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
import random as rd

class KNN:
    def __init__(self, data, n_folds) -> None:
        self.data = pd.DataFrame(data)
        self.N_features = np.shape(self.data)[1]
        self.n_folds = n_folds                      # 5 fold cross validation
 
## KNN algorithm

    # Find the min and max values for each column
    def dataset_minmax(self):
        dataset = np.array(self.data)
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax
    
    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = rd.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
            dataset_array = np.array(dataset_split)
        return dataset_array.tolist()
 
    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

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
        return predictions         

    def knn_run(self, option):
        dataset = np.array(self.data)
        folds = self.cross_validation_split(dataset, self.n_folds)
        scores = []
        knn_distances = []
        accuracies = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted_dist = self.k_nearest_neighbors(train_set, test_set, self.N_features)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted_dist)
            scores.append(accuracy)
            knn_distances.append(predicted_dist)
            accuracies.append(accuracy)
        if option == 'scores':
            return scores
        elif option == 'knn_dist':
            return knn_distances
        elif option == 'knn_accuracy':
            return accuracy
        else:
            return "KNN can only return: 'scores', 'knn_dist', or 'knn_accuracy'. Please reselect KNN options"