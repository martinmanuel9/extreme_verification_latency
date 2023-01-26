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
import ton_iot_datagen as ton_iot
import bot_iot_datagen as bot_iot
import classifier_performance as perf_metric
from skmultiflow.bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, OneClassSVM
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
        self.avg_perf_metric = {}

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
        if self.dataset == 'unsw_stream':
            unsw_stream = unsw_data.UNSW_NB15_Datagen()
            train_data = unsw_stream.allFeatTrain
            test_data = unsw_stream.allFeatTest
            train, test = unsw_stream.create_dataset(train= train_data, test= test_data)
            train = train['Dataset']
            test = test['Dataset']

            if len(train) < len(test):
                min_length = len(train)
            elif len(test) < len(train):
                min_length = len(train)

            ts = 0
            for i in range(0, min_length):
                self.train[ts] = train[i]
                ts += 1
            ts = 0
            for i in range(0, min_length):
                self.test[ts] = test[i]
                ts += 1
        if self.dataset == 'ton_iot':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.fridgeTrainStepsize, test_stepsize=datagen.fridgeTestStepsize, 
                                                    train=datagen.fridgeTrainSet, test= datagen.fridgeTestSet)
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
            # assert len(self.train.keys()) == len(self.test.keys()) 
        if self.dataset == 'bot_iot':
            datagen = bot_iot.BOT_IoT_Datagen()
            trainSetFeat = datagen.botTrainSet
            testSetFeat = datagen.botTestSet
            train, test = datagen.create_dataset(train=trainSetFeat, test=testSetFeat)

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

    def classify(self, ts, classifier, train, test):
        if self.classifier == 'naive_bayes_stream':
            naive_bayes = NaiveBayes() # multiflow
            t_start = time.time()
            self.predictions[ts] = naive_bayes.predict(test)
            t_end = time.time()
            naive_bayes.fit(train[:,:-1], test[:,-1])
            performance = perf_metric.PerformanceMetrics(timestep= ts, preds= self.predictions[ts], test= test, \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end) 
            self.perf_metric[ts] = performance.findClassifierMetrics(preds= self.predictions[ts], test= test[:,-1])

        elif self.classifier == 'naive_bayes':
            naive_bayes = BernoulliNB()
            t_start = time.time()
            naive_bayes.fit(train[:,:-1], train[:,-1])
            self.predictions[ts] = naive_bayes.predict(test[:,:-1])
            t_end = time.time()
            performance = perf_metric.PerformanceMetrics(timestep= ts, preds= self.predictions[ts], test= test, \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end) 
            self.perf_metric[ts] = performance.findClassifierMetrics(preds = self.predictions[ts], test = test[:,-1])

        elif self.classifier == 'svm':
            ssl_svm = SVC(kernel='rbf')
            t_start = time.time()
            print(np.unique(train[:,-1]))
            ssl_svm.fit(train[:,:-1], train[:,-1]) 
            self.predictions[ts] = ssl_svm.predict(test[:,:-1])
            t_end = time.time()
            performance = perf_metric.PerformanceMetrics(timestep= ts, preds= self.predictions[ts], test= test, \
                                        dataset= self.dataset , method= self.method , \
                                        classifier= self.classifier, tstart=t_start, tend=t_end) 
            self.perf_metric[ts] = performance.findClassifierMetrics(preds = self.predictions[ts], test = test[:,-1])
    
    def run(self):
        total_start = time.time()
        self.set_data()
        timesteps = self.train.keys()
        for ts in range(0, len(timesteps) -1):
            self.classify(ts=ts, classifier=self.classifier, train = self.train[ts], test= self.train[ts])
        
        total_end = time.time()
        total_time = total_end - total_start 
        avg_perf_metric = perf_metric.PerformanceMetrics(tstart= total_start, tend= total_end)
        self.avg_perf_metric = avg_perf_metric.findAvePerfMetrics(total_time= total_time, perf_metrics= self.perf_metric)
        return self.avg_perf_metric


# van = VanillaClassifier(classifier='naive_bayes', dataset='unsw')
# van.run()
