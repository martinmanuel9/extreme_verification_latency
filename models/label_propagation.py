#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          label_propagation.py
Author:             Martin Manuel Lopez
Creation:           03/11/2022

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""

# MIT License
#
# Copyright (c) 2021 Martin M Lopez
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

from sklearn.semi_supervised import LabelPropagation
import numpy as np

class Label_Propagation:
    def __init__(self, X_train, X_labeled, X_unlabeled):
        self.X = np.array(X_train)
        self.labels= np.array(X_labeled) 
        self.unlabeled = np.array(X_unlabeled)

    def ssl(self):
        labels = self.labels[:,-1]
        X = self.X
        # define model
        model = LabelPropagation(kernel='knn', n_neighbors=4, gamma=30, max_iter=2000)
        # fit model on training dataset
        model.fit(X, labels)
        # make predictions
        predicted_labels = model.predict(X)
        # to match structures 
        to_add = np.zeros((len(predicted_labels), np.shape(X)[1]-1))
        predicted_labels = np.column_stack((to_add, predicted_labels))
        return predicted_labels