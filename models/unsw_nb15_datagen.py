#!/usr/bin/env python 

"""
Application:        Cyber Attacks Data Generation from USNW - NB15 dataset 
File name:          unsw_nb15_datagen.py 
Author:             Martin Manuel Lopez
Creation:           12/5/2022

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

import random
from typing import Match
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path


class UNSW_NB15_Datagen:   
    def __init__(self) -> None:
        self.import_data()

    def change_directory(self):
        path = str(Path.home())
        path = path + '/extreme_verification_latency/data/UNSW_NB15/'
        os.chdir(path)

    def import_data(self):
        self.change_directory()
        self.trainSet = pd.read_csv('UNSW_NB15_training-set.csv')
        self.testSet = pd.read_csv('UNSW_NB15_testing-set.csv')
        self.flow_features()
        self.basic_features()
        self.content_features()
        self.time_features()
        self.generated_features()
        self.all_features()
        
    # cannot use flow features for COMPOSE as it has strings 
    def flow_features(self): 
        flow_features = ['proto','label'] # ; ['scrip', 'sport','dstip','dsport'] were not found in the csv file
        self.flowFeatTrain = self.trainSet[flow_features]
        self.flowFeatTest = self.testSet[flow_features]

    def basic_features(self):
        # basic featues has 'state' & 'service' removed for EVL as they are str
        basic_features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss','dloss','sload', 'dload', 'spkts', 'dpkts', 'label']
        self.basicFeatTrain = self.trainSet[basic_features]
        self.basicFeatTest = self.testSet[basic_features]

    def content_features(self):
        contet_features = ['swin','dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth','label'] # ['res_bdy_len] was not found in csv
        self.contentFeatTrain = self.trainSet[contet_features]
        self.contentFeatTest = self.testSet[contet_features]

    def time_features(self):
        time_features =['sjit', 'djit',  'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'label']   # ['stime', 'ltime'] not found in csv
        self.timeFeatTrain = self.trainSet[time_features]
        self.timeFeatTest = self.testSet[time_features]

    def generated_features(self):
        generated_features = ['is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
                                'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'label' ]
        self.generateFeatTrain = self.trainSet[generated_features]
        self.generateFeatTest = self.testSet[generated_features]

    def all_features(self):
        all_features = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss','dloss','sload', 'dload', 'spkts', 'dpkts',
                        'swin','dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth','sjit', 'djit',  'sinpkt', 'dinpkt',  
                        'tcprtt', 'synack', 'ackdat','is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',  
                        'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst','ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'label']
        self.allFeatTrain = self.trainSet[all_features]
        self.allFeatTest = self.testSet[all_features]
    
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield np.array(iterable[ndx:min(ndx + n, l)])

    def create_dataset(self, train, test):
        self.trainDict = {}
        self.testDict = {}
        train_stepsize = 1750 
        test_stepsize = 820
        trainSet = train.to_numpy()
        testSet = test.to_numpy()
        
        a = []
        indx = []
        for d in range(test_stepsize-1):
            a.append(d)
        for v in range(int(0.5 * len(a))):
            rnd = random.choice(a)
            indx.append(rnd)

        self.trainDataset = trainSet
        self.trainData = trainSet
        self.trainLabels = trainSet[:,-1]
        self.trainUse = trainSet[:train_stepsize]
        self.trainUse[:,-1][indx] = 1

        self.testDataset = testSet
        self.testData = testSet
        self.testLabels = testSet[:,-1]
        self.testUse = testSet[:test_stepsize]
        self.testUse[:,-1][indx] = 1

        trainDataset = []
        X_train = []
        for i in self.batch(self.trainData, train_stepsize):
            trainDataset.append(i)
        X_train.append(trainDataset)
        self.trainData = np.array(X_train, dtype=object)

        testDataset = []
        y_test = []
        for i in self.batch(self.testData, test_stepsize):
            testDataset.append(i)
        y_test.append(testDataset)
        self.testData = np.array(y_test, dtype=object)

        trainLabels = []
        lblTrainData = []
        for i in self.batch(self.trainLabels, train_stepsize):
            trainLabels.append(i)
        lblTrainData.append(trainLabels)
        self.trainLabels = lblTrainData

        testLabels = []
        lblTestData = []
        for i in self.batch(self.testLabels, test_stepsize):
            testLabels.append(i)
        lblTestData.append(trainLabels)
        self.testLabels = lblTestData

        self.trainDict['Dataset'] = self.trainDataset
        self.trainDict['Data'] = self.trainData
        self.trainDict['Labels'] = self.trainLabels
        self.trainDict['Use'] = self.trainUse

        self.testDict['Dataset'] = self.testDataset
        self.testDict['Data'] = self.testData
        self.testDict['Labels'] = self.testLabels
        self.testDict['Use'] = self.testUse

        return self.trainDict, self.testDict

# set up the dataset generation 
# dataset = UNSW_NB15_Datagen()
# gen_train_features = dataset.flowFeatTrain
# gen_test_features =dataset.generateFeatTest 
# print(gen_train_features)
# X, y = dataset.create_dataset(train=gen_train_features, test=gen_test_features)




