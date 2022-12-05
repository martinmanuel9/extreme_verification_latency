#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
File name:          classifier_performance.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

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

from random import random
import numpy as np 
import pandas as pd
import sklearn.metrics as metric
import random

class PerformanceMetrics:
    """
    Intent of Classification Class is to get the classification metrics for either 
    a single timestep or an overtime performance of the a classifier 
    Initialization requires: 
        timestep = what time step you want get the prediction, can be single or array 
        preds = predictions from classifier 
        test = expected results 
        dataset = what is the dataset you are running 
        method = the algorithm running COMPOSE or Fast COMPOSE or MClass etc 
        classifier = classifying algorithm; SSL ; QN_S3VM; Label Propogation
        tstart = time started classification
        tend = end time classification
    """
    def __init__(self, tstart, tend, timestep= None, preds= None, test= None, dataset= None, method= None, classifier= None):
        self.preds = preds
        self.test = test
        self.selected_dataset = dataset
        self.method = method
        self.classifier = classifier
        self.ts = timestep
        self.total_time = tend - tstart
        # classification metrics
        self.classifier_error = {}
        self.classifier_accuracy = {}
        self.roc_auc_score = {}
        self.roc_auc_plot = {}
        self.f1_score = {}
        self.mathews_corr_coeff = {}
        # batch results - creates dict of all avg results 
        self.avg_results = {}
        # results
        self.perf_metrics = {}

    def findClassifierMetrics(self, preds, test):
        with np.errstate(divide='ignore'):
            self.classifier_error[self.ts] =  np.sum(preds != test) / len(preds)
        self.classifier_accuracy[self.ts] = 1 - self.classifier_error[self.ts]
        # class_report = metric.classification_report(test, preds)

        # roc curve
        try:
            self.roc_auc_score[self.ts] = metric.roc_auc_score(test, preds)
            fpr, tpr, _ = metric.roc_curve(test, preds, pos_label=1)
            self.roc_auc_plot[self.ts] = [fpr, tpr]
        except ValueError:
            self.roc_auc_score[self.ts] = 'Only one class found'
            self.roc_auc_plot[self.ts] = 'Only one class found'
        # F1-score
        self.f1_score[self.ts] = metric.f1_score(test.astype(int), preds.astype(int), average=None) 
        # Mathews Correlation Coefficient 
        self.mathews_corr_coeff[self.ts] = metric.matthews_corrcoef(test.astype(int), preds.astype(int))
        # add to dict
        self.perf_metrics['Dataset'] = self.selected_dataset
        self.perf_metrics['Classifier'] = self.classifier
        self.perf_metrics['Method'] = self.method
        self.perf_metrics['Classifier_Error'] = self.classifier_error[self.ts]
        self.perf_metrics['Classifier_Accuracy'] = self.classifier_accuracy[self.ts]
        self.perf_metrics['ROC_AUC_Score'] = self.roc_auc_score[self.ts]
        self.perf_metrics['ROC_AUC_Plotter'] = self.roc_auc_plot[self.ts]
        self.perf_metrics['F1_Score'] = self.f1_score[self.ts]
        self.perf_metrics['Matthews_CorrCoef'] = self.mathews_corr_coeff[self.ts]
        self.perf_metrics['Total_Time_Seconds'] = self.total_time
        self.perf_metrics['Total_Time_Min'] = self.total_time / 60
        perf_metric_df = pd.DataFrame.from_dict((self.perf_metrics.keys(), self.perf_metrics.values())).T
        performance_metrics = pd.DataFrame(perf_metric_df.values, columns=['Metrics', 'Scores'])
        
        return self.perf_metrics

    def findAvePerfMetrics(self, total_time, perf_metrics):
        self.selected_dataset = []
        self.method = []
        self.classifier = []
        self.total_time = {}

        metrics = perf_metrics
        total_time_sec = total_time
        total_time_min = total_time/60
        self.selected_dataset = metrics[0]['Dataset']
        self.classifier = metrics[0]['Classifier']
        self.method = metrics[0]['Method']
        for k in metrics.keys():
            self.classifier_error[k] = metrics[k]['Classifier_Error']
            self.classifier_accuracy[k] = metrics[k]['Classifier_Accuracy']
            self.roc_auc_score[k] = metrics[k]['ROC_AUC_Score']
            self.f1_score[k] = metrics[k]['F1_Score']
            self.mathews_corr_coeff[k] = metrics[k]['Matthews_CorrCoef']
            self.total_time[k] = metrics[k]['Total_Time_Seconds']
            
        avg_error = np.array(sum(self.classifier_error.values()) / len(self.classifier_error))
        avg_accuracy = np.array(sum(self.classifier_accuracy.values()) / len(self.classifier_accuracy))
        avg_exec_time_sec = np.array(sum(self.total_time.values()) / len(self.total_time))
        avg_exec_time_min = avg_exec_time_sec / 60
        roc_auc_scores = []
        for c in self.roc_auc_score.keys():
            if self.roc_auc_score[c] == 'Only one class found':
                break
            else:
                roc_auc_scores.append(self.roc_auc_score[c])
        if len(roc_auc_scores) < 1 :
            avg_roc_auc_score = 'Only one class found'
        else:
            avg_roc_auc_score = np.array(sum(roc_auc_scores) / len(roc_auc_scores))
        # print(np.divide(list(self.f1_score.values()), len(self.f1_score)))
        f1_scores = []
        first = np.shape(self.f1_score[0])[0]
        for s in self.f1_score.values():
            if np.shape(s)[0] > first:
                break
            else:
                f1_scores.append(s)
        f1_scores = np.array(f1_scores)
        # print(type(np.divide(list(self.f1_score.values()), len(self.f1_score))))
        # avg_f1_score = np.array(sum(self.f1_score.values()/ len(self.f1_score)))
        avg_f1_score = np.array(sum(f1_scores/ len(self.f1_score)))
        avg_matt_corrcoeff = np.array(sum(self.mathews_corr_coeff.values())/ len(self.mathews_corr_coeff))
        self.avg_results['Dataset'] = self.selected_dataset
        self.avg_results['Classifier'] = self.classifier
        self.avg_results['Method'] = self.method
        self.avg_results['Avg_Error'] = avg_error
        self.avg_results['Avg_Accuracy'] = avg_accuracy
        self.avg_results['Avg_Exec_Time_Sec'] = avg_exec_time_sec
        self.avg_results['Avg_Exec_Time_Min'] = avg_exec_time_min
        self.avg_results['Avg_ROC_AUC_Score'] = avg_roc_auc_score
        self.avg_results['Avg_F1_Score'] = avg_f1_score
        self.avg_results['Avg_Matthews_Corr_Coeff'] = avg_matt_corrcoeff
        self.avg_results['Total_Exec_Time_Sec'] = total_time_sec
        self.avg_results['Total_Exec_Time_Min'] = total_time_min
        timesteps = self.classifier_accuracy.keys()
        timesteps = list(timesteps)
        timesteps = np.array(timesteps)
        accuracy = self.classifier_accuracy.values()
        accuracy = list(accuracy)
        accuracy = np.array(accuracy)
        self.avg_results['Timesteps'] = timesteps
        self.avg_results['Accuracy'] = accuracy
        
        # avg_perf_metrics = pd.DataFrame({'Dataset': [self.selected_dataset], 'Classifier': [self.classifier],'Method': [self.method], 'Avg_Error': [avg_error], 
        #                                 'Avg_Accuracy': [avg_accuracy], 'Avg_Exec_Time_Sec': [avg_exec_time_sec], 'Avg_Exec_Time_Min': [avg_exec_time_min],
        #                                 'Avg_ROC_AUC_Score': [avg_roc_auc_score], 'Avg_F1_Score': [avg_f1_score], 'Avg_Matthews_Corr_Coeff': [avg_matt_corrcoeff], 
        #                                 'Total_Exec_Time_Sec': [total_time_sec], 'Total_Exec_Time_Min': [total_time_min], 'Timesteps':[self.classifier_accuracy.keys()],
        #                                 'Accuracy': [self.classifier_accuracy.values()]}, 
        #                     columns=['Dataset','Classifier','Method','Avg_Error', 'Avg_Accuracy', 'Avg_Exec_Time_Sec','Avg_Exec_Time_Min',
        #                             'Avg_ROC_AUC_Score', 'Avg_F1_Score', 'Avg_Matthews_Corr_Coeff','Total_Exec_Time_Sec','Total_Exec_Time_Min', 'Timesteps', 'Accuracy'])
        return self.avg_results