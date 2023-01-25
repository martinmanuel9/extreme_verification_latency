#!/usr/bin/env python 

"""
Application:        Cyber Attacks Data Generation of IoT Devices  
File name:          bot_iot_datagen.py
Author:             Martin Manuel Lopez
Creation:           1/19/2023

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
import pandas as pd
import numpy as np
import os
from pathlib import Path  
from category_encoders import OrdinalEncoder 
class BOT_IoT_Datagen():
    def __init__(self) -> None:
        self.import_data()

    def change_directory(self):
        path = str(Path.home())
        path = path + '/extreme_verification_latency/data/BoT_IoT_Data/'
        os.chdir(path)

    def import_data(self):
        self.change_directory()
        self.updateDataset()

    def updateDataset(self):
        trainData = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
        mapping = [{'col': 'proto',  'mapping': {'arp': 1, 'icmp': 2, 'ipv6-icmp': 3, 'tcp': 4, 'tcp': 5, 'udp': 6}},
                        {'col': 'category', 'mapping': {'Normal':0, 'DDoS': 1, 'DoS': 2, 'Reconnaissance':3, 'Theft':4}},
                        {'col' : 'subcategory', 'mapping': {'Normal': 0, 'HTTP': 1, 'Keylogging': 2, 'OS_Fingerprint': 3, 'Service_Scan': 4, 'TCP': 5, 'UDP': 6}}]
        trainData = OrdinalEncoder(cols=['proto', 'category', 'subcategory'], mapping=mapping).fit(trainData).transform(trainData)
        testData = pd.read_csv('UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')
        testData = OrdinalEncoder(cols=['proto', 'category', 'subcategory'], mapping=mapping).fit(testData).transform(testData)
        self.botTrainSet = trainData[['pkSeqID','proto','sport','dport',
                        'seq','stddev', 'N_IN_Conn_P_SrcIP', 'min','state_number','mean',
                        'N_IN_Conn_P_DstIP','drate','srate','max','category','subcategory','attack']] # removing IP addresses 
        self.botTestSet = testData[['pkSeqID','proto','sport','dport',
                        'seq','stddev', 'N_IN_Conn_P_SrcIP', 'min','state_number','mean',
                        'N_IN_Conn_P_DstIP','drate','srate','max','category','subcategory','attack']]
        print(self.botTrainSet)

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

datagen = BOT_IoT_Datagen()
trainSetFeat = datagen.botTrainSet
testSetFeat = datagen.botTestSet
trainSet, testSet = datagen.create_dataset(train=trainSetFeat, test=testSetFeat)
print(np.shape(trainSet['Data']), np.shape(testSet['Data']))
