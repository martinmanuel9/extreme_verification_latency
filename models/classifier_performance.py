#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
File name:          classifier_performance.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

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

import numpy as np 
import pandas as pd
import sklearn.metrics as metric

class ClassifierMetrics:

    def __init__(self, preds, test, timestep, dataset, method, classifier, time_to_predict ):
        self.preds = preds
        self.test = test
        self.timestep = timestep
        self.selected_dataset = dataset
        self.method = method
        self.classifier = classifier
        self.time_to_predict = time_to_predict
        self.avg_results_dict = {}
        self.avg_results = {}
        self.classifier_error = {}
        self.accuracy_score_sklearn = {}
        self.classifier_accuracy = {}

    def classification_error(self, preds, L_test):  
        self.classifier_error[self.timestep] =  np.sum(preds != L_test)/len(preds)
        self.classifier_accuracy[self.timestep] = 1 - self.classifier_error[self.timestep] * 100

    def results_logs(self):
            avg_error = np.array(sum(self.classifier_error.values()) / len(self.classifier_error))
            avg_accuracy = np.array(sum(self.classifier_accuracy.values()) / len(self.classifier_accuracy))
            avg_exec_time = np.array(sum(self.time_to_predict.values()) / len(self.time_to_predict))
            avg_results_df = pd.DataFrame({'Dataset': [self.selected_dataset], 'Classifier': [self.classifier],'Method': [self.method], 'Avg_Error': [avg_error], 'Avg_Accuracy': [avg_accuracy], 'Avg_Exec_time': [avg_exec_time]}, 
                                columns=['Dataset','Classifier','Method','Avg_Error', 'Avg_Accuracy', 'Avg_Exec_Time'])
            self.avg_results_dict['Dataset'] = self.selected_dataset
            self.avg_results_dict['Classifier'] = self.classifier
            self.avg_results_dict['Method'] = self.method
            self.avg_results_dict['Avg_Error'] = avg_error
            self.avg_results_dict['Avg_Accuracy'] = avg_accuracy
            self.avg_results_dict['Avg_Exec_Time'] = avg_exec_time
            run_method = self.selected_dataset + '_' + self.classifier + '_' + self.method
            self.avg_results[run_method] = avg_results_df

            df = pd.DataFrame.from_dict((self.classifier_accuracy.keys(), self.classifier_accuracy.values())).T
            accuracy_scores = pd.DataFrame(df.values, columns=['Timesteps', 'Accuracy'])
            x = accuracy_scores['Timesteps']
            y = accuracy_scores['Accuracy']