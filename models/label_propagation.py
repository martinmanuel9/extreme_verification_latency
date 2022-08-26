#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          label_propagation.py
Author:             Martin Manuel Lopez
Creation:           03/11/2022

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
PhD Advisor: Dr. Gregory Ditzler
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


# # evaluate logistic regression fit on label propagation for semi-supervised learning
# from numpy import concatenate
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.semi_supervised import LabelPropagation
# from sklearn.linear_model import LogisticRegression
# # define dataset
# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=1)
# # split into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1, stratify=y)
# # split train into labeled and unlabeled
# X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.50, random_state=1, stratify=y_train)
# # create the training dataset input
# X_train_mixed = concatenate((X_train_lab, X_test_unlab))
# # create "no label" for unlabeled data
# nolabel = [-1 for _ in range(len(y_test_unlab))]
# # recombine training dataset labels
# y_train_mixed = concatenate((y_train_lab, nolabel))
# # define model
# model = LabelPropagation()
# # fit model on training dataset
# model.fit(X_train_mixed, y_train_mixed)
# # get labels for entire training dataset data
# tran_labels = model.transduction_
# # define supervised learning model
# model2 = LogisticRegression()
# # fit supervised learning model on entire training dataset
# model2.fit(X_train_mixed, tran_labels)
# # make predictions on hold out test set
# yhat = model2.predict(X_test)
# # calculate score for test set
# score = accuracy_score(y_test, yhat)
# # summarize score
# print('Accuracy: %.3f' % (score*100))

from socket import if_nametoindex
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn import preprocessing
from sklearn import utils
import random
np.seterr(invalid='ignore')
class Label_Propagation:
    def __init__(self, X_train, X_labeled, X_unlabeled):
        self.X = np.array(X_train)
        self.labels= np.array(X_labeled)
        self.unlabeled = np.array(X_unlabeled)
        self.actual_label = np.shape(X_labeled)[1]-1

    def ssl(self):
        
        labels = self.labels[:,self.actual_label]
        # labels_orig = np.copy(self.labels[:,self.actual_label])
        # labels = np.floor(labels)
        # labels_orig = np.floor(labels_orig)
        X = self.X
        
        # define model
        model = LabelPropagation(kernel='knn', n_neighbors=4, gamma=30, max_iter=2000)
        # fit model on training dataset
        
        if len(labels) < len(X):
            dif = len(X) - len(labels)
            labels_to_add = []
            for r in range(dif):
                rndm_label = random.choice(labels)
                labels_to_add = np.append(labels_to_add, rndm_label)
            labels = np.append(labels, labels_to_add)
        elif len(X) < len(labels):
            dif = len(labels) - len(X)
            X_to_add = np.ones(np.shape(X)[1])
            for k in range(dif):
                rdm_X = random.randint(0, len(X)-1)
                X_to_add = np.vstack((X_to_add, X[rdm_X]))
            X_to_add = np.delete(X_to_add, 0, axis=0)
            X = np.vstack((X, X_to_add))
            
        
        with np.errstate(divide='ignore'):
            model.fit(X, labels)
        
        # make predictions
        predicted_labels = model.predict(X)

        return predicted_labels