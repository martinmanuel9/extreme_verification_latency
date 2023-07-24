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
        # path +=  '/extreme_verification_latency/data/BoT_IoT_Data/'
        path =  '../data/BoT_IoT_Data/'
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

        self.botTrainSet = trainData[['proto','seq','stddev', 'N_IN_Conn_P_SrcIP', 'min','state_number','mean',
                        'N_IN_Conn_P_DstIP','drate','srate','max','category','subcategory','attack']] # removing IP addresses and pkSeqID, 'sport','dport'
        self.botTestSet = testData[['proto','seq','stddev', 'N_IN_Conn_P_SrcIP', 'min','state_number','mean',
                        'N_IN_Conn_P_DstIP','drate','srate','max','category','subcategory','attack']]

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield np.array(iterable[ndx:min(ndx + n, l)])

    def create_dataset(self, train, test):
        self.trainDict = {}
        self.testDict = {}
        train_stepsize = 29348 
        test_stepsize = 7337
        trainSet = train.to_numpy()
        testSet = test.to_numpy()

        # randomize the order of the of test set
        N = len(trainSet)
        V = len(testSet)
        ii = np.random.randint(0, N, N)
        jj = np.random.randint(0, V, V)
        trainSet = trainSet[ii] 
        testSet =  testSet[jj]

        # numAttacks = sum( num > 0 for num in trainSet[:,-1])
        # noAttacks = sum( num < 1 for num in trainSet[:,-1])

        trainAtckLblSort = np.argwhere(trainSet[:,-1]>0)
        testAtckLblSort = np.argwhere(testSet[:,-1]>0)
        trainAttackSet = trainSet[trainAtckLblSort]
        testAttackSet = testSet[testAtckLblSort]

        trainZeroLblSort = np.argwhere(trainSet[:,-1]<1)
        testZeroLblSort = np.argwhere(testSet[:,-1]<1)

        trainZeroLbls = np.squeeze(trainSet[trainZeroLblSort])
        testZeroLbls = np.squeeze(testSet[testZeroLblSort])
        
        attackTrainStep = train_stepsize - len(trainZeroLbls)
        attackTestStep = test_stepsize - len(testZeroLbls)
        # make batches to later concatenante so that all batches have attacks and no attacks 
        trainAttack = []
        lblAttack = []
        for i in self.batch(trainAttackSet, attackTrainStep):
            trainAttack.append(i)
        lblAttack.append(trainAttack)
        trnAttackSet = np.array(lblAttack, dtype=object)
        
        for j in range(0, len(trnAttackSet[0])-1):
            trnAttackSet[0][j] = np.vstack((np.squeeze(trnAttackSet[0][j]), trainZeroLbls))
            K = len(trnAttackSet[0][j]) 
            kk = np.random.randint(0, K, K)
            trnAttackSet[0][j] = trnAttackSet[0][j][kk]

        testAttack =[]
        lblTestAttack = []
        for i in self.batch(testAttackSet, attackTestStep):
            testAttack.append(i)
        lblTestAttack.append(testAttack)
        tstAttackSet = np.array(lblTestAttack, dtype=object)

        for j in range(0, len(tstAttackSet[0])-1):
            tstAttackSet[0][j] = np.vstack((np.squeeze(tstAttackSet[0][j]), testZeroLbls))
            K = len(tstAttackSet[0][j])
            kk = np.random.randint(0, K, K)
            tstAttackSet[0][j] = tstAttackSet[0][j][kk]

        # set batched and randomized sets to train and test data
        self.trainData = trnAttackSet
        self.testData = tstAttackSet

        # get the training lables 
        trainLbls = []
        for i in range(0, len(self.trainData[0])-1):
            trainLbls.append(self.trainData[0][i][:,-1])
        self.trainLabels = trainLbls

        testLbls = []
        for i in range(0, len(self.testData[0])-1):
            testLbls.append(self.testData[0][i][:,-1])
        self.testLabels = testLbls 

        # get the datasets as a single array
        trainDataset = []
        trainDataset.append(np.squeeze(self.trainData[0][0]))
        for t in range(1, len(self.trainData[0])-1):
            trainDataset = np.vstack((np.squeeze(trainDataset), np.squeeze(self.trainData[0][t])))
        self.trainDataset = trainDataset

        testDataset =[]
        testDataset.append(np.squeeze(self.testData[0][0]))
        for t in range(1, len(self.testData[0])-1):
            testDataset = np.vstack((np.squeeze(testDataset), np.squeeze(self.testData[0][t])))
        self.testDataset = testDataset

        # get use sets:
        self.trainUse = self.trainData[0][0]
        self.testUse = self.testData[0][0]

        
        self.trainDict['Dataset'] = self.trainDataset
        self.trainDict['Data'] = self.trainData
        self.trainDict['Labels'] = self.trainLabels
        self.trainDict['Use'] = self.trainUse

        self.testDict['Dataset'] = self.testDataset
        self.testDict['Data'] = self.testData
        self.testDict['Labels'] = self.testLabels
        self.testDict['Use'] = self.testUse

        return self.trainDict, self.testDict

# datagen = BOT_IoT_Datagen()
# trainSetFeat = datagen.botTrainSet
# testSetFeat = datagen.botTestSet
# trainSet, testSet = datagen.create_dataset(train=trainSetFeat, test=testSetFeat)
# print(np.shape(trainSet['Data']), np.shape(testSet['Data']))