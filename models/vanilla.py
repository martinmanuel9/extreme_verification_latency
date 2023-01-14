#!/usr/bin/env python 

"""
Application:        Vanilla Application and ML - 
File name:          vanilla.py
Author:             Martin Manuel Lopez
Creation:           01/13/2023

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


import pandas as pd
import numpy as np
import unsw_nb15_datagen as unsw_data
import datagen_synthetic as synthetic_data
import classifier_performance as perf_metric
from skmultiflow.bayes import NaiveBayes
import time


class VanillaClassifier():
    def __init__(self, classifier, dataset) -> None:
        self.classifier = classifier
        self.method = classifier
        self.dataset = dataset
        self.train = {}
        self.test ={}
        self.predictions = {}
        self.perf_metric = {}

    def set_data(self):
        if self.dataset == 'unsw':
            unsw = unsw_data.UNSW_NB15_Datagen()
            train_data = unsw.allFeatTrain
            test_data = unsw.allFeatTest
            train, test = unsw.create_dataset(train=train_data, test= test_data)
            train = train['Data'] # create data set with timesteps with dictionary of 'Data'
            test = test['Data']
            ts = 0
            for i in range(0, len(train[0])):
                self.train[ts] = train[0][i]
                ts += 1
            ts = 0
            for i in range(0, len(test[0])):
                self.test[ts] = test[0][i]
                ts += 1
            assert len(self.train.keys()) == len(self.test.keys()) 

    def classify(self, ts, classifier, train, test):
        if self.classifier == 'naive_bayes':
            naive_bayes = NaiveBayes()
            t_start = time.time()
            naive_bayes.fit(train, test)
            self.predictions[ts] = naive_bayes.predict(test)
            t_end = time.time()
            naive_bayes.partial_fit(train, test)
            performance = perf_metric.PerformanceMetrics(timestep= ts, preds= self.predictions[ts], test= test, \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end) 
            self.perf_metric[ts] = performance.findClassifierMetrics(preds= self.predictions[ts], test= test)

            print(self.perf_metric)
        elif self.classifier == 'svm':
            pass
        elif self.classifier == 'knn':
            pass
    
    def run(self):
        self.set_data()
        timesteps = self.train.keys()
        for ts in range(0, len(timesteps) -1):
            self.classify(ts=ts, classifier=self.classifier, train = self.train[ts], test= self.train[ts])
            
    

van = VanillaClassifier(classifier='naive_bayes', dataset='unsw')
van.run()