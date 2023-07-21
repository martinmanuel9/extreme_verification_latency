#!/usr/bin/env python 
"""
Application:        SCARGC
File name:          scargc.py
Author:             Martin Manuel Lopez
Creation:           05/30/2021

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""
# MIT License
#
# Copyright (c) 2022
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

from cProfile import label
import warnings
# warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import statistics
import numpy as np 
from scipy import stats
from sklearn.svm import SVC, SVR
from tqdm import tqdm
import math
import benchmark_datagen_old as bdg
import ton_iot_datagen as ton_iot
import bot_iot_datagen as bot_iot
import unsw_nb15_datagen as unsw
import classifier_performance as cp
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
from sklearn import tree
# from skmultiflow.bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import time
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from knn import knn as Bknn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class SCARGC: 
    def __init__(self, 
                datasource,
                dataset,  
                Kclusters:int=10,
                maxpool:int=25, 
                resample:bool=True, 
                T:int=100,
                classifier:str=''): 
        """
        Removed Xinit, Yinit
        """
        # set the classifier that is used [eg 1nn or svm]
        self.classifier = classifier 
        # set the number of clusters for kmeans
        self.Kclusters = Kclusters
        # this will associate a cluster to a class in the data 
        # self.class_cluster = np.zeros((self.Kclusters,))
        self.class_cluster = {}
        # set the data 
        self.X = {} # Xinit
        self.Y = {} # Yinit
        self.Xinit = {}
        self.Yinit = {}
        self.data = {}
        self.labeled = {}
        self.all_data = []
        self.train_model = []
        self.datasource = datasource
        self.dataset = dataset
        # set resample 
        self.resample = resample
        # set max pool size 
        self.maxpool = maxpool
        # establish dataset 
        self.set_data()
        # initialize the cluster model
        self.initialize(Xinit= self.Xinit, Yinit = self.Yinit)
        self.T = 0
        self.performance_metric = {}
        self.avg_perf_metric = {}
        self.preds = {}
        self.n_cores = []
 
    def set_data(self):
        if self.datasource == 'synthetic':
            set_data = bdg.Datagen()
            data_gen = set_data.gen_dataset(self.dataset)
            data ={}
            labeled = {}
            unlabeled = {}
            ts = 0

            # set a self.data dictionary for each time step 
            # self.dataset[0][i] loop the arrays and append them to dictionary
            # data is the datastream 
            for i in range(0, len(data_gen[0])):
                data[ts] = data_gen[0][i]
                ts += 1

            # filter out labeled and unlabeled from of each timestep
            for i in data:
                len_of_batch = len(data[i])
                label_batch = []
                unlabeled_batch = []            
                for j in range(0, len_of_batch - 1):
                    if data[i][j][2] == 1:              # will want to say that label == 1
                        label_batch.append(data[i][j])
                        labeled[i] = label_batch
                    else:
                        unlabeled_batch.append(data[i][j])
                        unlabeled[i] = unlabeled_batch

            # convert labeled data to match self.data data structure
            labeled_keys = labeled.keys()
            for key in labeled_keys:        
                if len(labeled[key]) > 1:
                    len_of_components = len(labeled[key])
                    array_tuple = []
                    for j in range(0, len_of_components):
                        array = np.array(labeled[key][j])
                        arr_to_list = array.tolist()
                        array_tuple.append(arr_to_list)
                        array = []
                        arr_to_list = []
                    concat_tuple = np.vstack(array_tuple)
                    labeled[key] = concat_tuple
            
            self.X = labeled    # set of all labels as a dict per timestep ; we only need X[0] for initial labels
            self.Y = data

            self.Xinit = self.X
            self.Yinit = self.Y

        # elif self.datasource == 'unsw':
        #     # dataset = UNSW_NB15_Datagen()
        #     # gen_train_features = dataset.generateFeatTrain
        #     # gen_test_features =dataset.generateFeatTest 
        #     # X, y = dataset.create_dataset(train=gen_train_features, test=gen_test_features)
        #     # we have the following categoires : flow, basic, time, content, generated 
        #     unsw_gen = unsw.UNSW_NB15_Datagen()
        #     # type of unsw features : generated, time, content, basic, allFeatures
        #     gen_train_features = unsw_gen.allFeatTrain
        #     gen_test_features = unsw_gen.allFeatTest
        #     train , test = unsw_gen.create_dataset(train = gen_train_features, test = gen_test_features)
        #     data = train['Data']
        #     dataset = train['Dataset']
        #     labels = train['Labels']
        #     testDataset = train['Dataset']
        #     testData = test['Data']
        #     testLabels = test['Labels']
            
        #     ts = 0
        #     # set data (all the features)
        #     for i in range(0, len(data[0])):
        #         self.data[ts] = data[0][i]
        #         ts += 1
        #     # set all the labels 
        #     ts = 0
        #     for k in range(0, len(labels[0])):
        #         self.labeled[ts] = labels[0][k]
        #         ts += 1


        if self.dataset == 'ton_iot_fridge':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.fridgeTrainStepsize, test_stepsize=datagen.fridgeTestStepsize, 
                                                    train=datagen.fridgeTrainSet, test= datagen.fridgeTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1
            
            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'ton_iot_garage':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.garageTrainStepsize, test_stepsize=datagen.garageTestStepsize, 
                                                    train=datagen.garageTrainSet, test= datagen.garageTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'ton_iot_gps':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.gpsTrainStepsize, test_stepsize=datagen.gpsTestStepsize, 
                                                    train=datagen.gpsTrainSet, test= datagen.gpsTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']


        elif self.dataset == 'ton_iot_modbus':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.modbusTrainStepsize, test_stepsize=datagen.modbusTestStepsize, 
                                                    train=datagen.modbusTrainSet, test= datagen.modbusTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'ton_iot_light':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.lightTrainStepsize, test_stepsize=datagen.lightTestStepsize, 
                                                    train=datagen.lightTrainSet, test= datagen.lightTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'ton_iot_thermo':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.thermoTrainStepsize, test_stepsize=datagen.thermoTestStepsize, 
                                                    train=datagen.thermoTrainSet, test= datagen.thermoTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'ton_iot_weather':
            datagen = ton_iot.TON_IoT_Datagen()
            # need to select what IoT data you want fridge, garage, GPS, modbus, light, thermostat, weather 
            train, test =  datagen.create_dataset(train_stepsize=datagen.weatherTrainStepsize, test_stepsize=datagen.weatherTestStepsize, 
                                                    train=datagen.weatherTrainSet, test= datagen.weatherTestSet)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels[0])):
                self.labeled[ts] = labels[0][k]
                ts += 1

            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']

        elif self.dataset == 'bot_iot':
            datagen = bot_iot.BOT_IoT_Datagen()
            trainSetFeat = datagen.botTrainSet
            testSetFeat = datagen.botTestSet
            train, test = datagen.create_dataset(train=trainSetFeat, test=testSetFeat)
            data = train['Data']
            labels = train['Labels']
            core_supports = train['Use']
            dataset = train['Dataset']
            testData = test['Data']
            testLabels = test['Labels']
            testCoreSupports = test['Use']
            ts = 0
            # set data (all the features)
            for i in range(0, len(data[0])):
                self.data[ts] = data[0][i]
                ts += 1
            # set all the labels 
            ts = 0
            for k in range(0, len(labels)):
                self.labeled[ts] = labels[k]
                ts += 1
            
            dict_train = {}
            for i in range(0, len(train['Data'][0])):
                dict_train[i] = train['Data'][0][i]
            
            dict_test = {}
            for j in range(0, len(test['Data'][0])):
                dict_test[j] = test['Data'][0][j]

            self.Xinit = dict_train
            self.Yinit = dict_test

            self.X = dict_train
            self.Y = dict_test
            self.all_data = train['Dataset']


        # get the number of classes in the dataset 
        self.nclasses = len(np.unique(self.Y))

    def initialize(self, Xinit, Yinit): 
        """
        """
        # run the clustering algorithm on the training data then find the cluster 
        # assignment for each of the samples in the training data
        self.set_cores()
        with ProcessPoolExecutor(max_workers=self.n_cores): 
            if self.datasource == 'synthetic':
                self.cluster = KMeans(n_clusters=self.Kclusters).fit(Xinit[0])
                labels = self.cluster.predict(Xinit[1])
                
                # for each of the clusters, find the labels of the data samples in the clusters
                # then look at the labels from the initially labeled data that are in the same
                # cluster. assign the cluster the label of the most frequent class. 
                for i in range(self.Kclusters):
                    yhat = Yinit[i][labels]
                    mode_val,_ = stats.mode(yhat)
                    self.class_cluster[i] = mode_val
            elif self.datasource == 'UNSW':
                self.cluster = KMeans(n_clusters=self.Kclusters).fit(Xinit[0])    
                labels = self.cluster.predict(Yinit[0])
                
                # for each of the clusters, find the labels of the data samples in the clusters
                # then look at the labels from the initially labeled data that are in the same
                # cluster. assign the cluster the label of the most frequent class. 
                for i in range(self.Kclusters):
                    yhat = Yinit[i][labels]
                    mode_val,_ = stats.mode(yhat)
                    self.class_cluster[i] = mode_val

    def set_cores(self):
        """
        Establishes number of cores to conduct parallel processing
        """
        num_cores = multiprocessing.cpu_count()         # determines number of cores
        percent_cores = math.ceil( num_cores)
        self.n_cores = int(percent_cores)                   # original number of cores to 1
        
    def run(self): 
        '''
        Xts = Initial Training data
        Yts = Data stream
        '''
        Xts = self.X
        Yts = self.Y
        

        self.set_cores()
        with ProcessPoolExecutor(max_workers=self.n_cores):
            total_time_start = time.time()
            # Build Classifier 
            if self.classifier == '1nn':
                if self.datasource == 'synthetic':
                    knn = KNeighborsRegressor(n_neighbors=1).fit(Yts[0], Xts[0])           # KNN.fit(train_data, train label)
                    predicted_label = knn.predict(Yts[1])
                elif self.datasource == 'UNSW':
                    knn = KNeighborsClassifier(n_neighbors=1).fit(self.all_data[:,:-1], self.all_data[:,-1])           # KNN.fit(train_data, train label)
                    self.train_model = knn
                    predicted_label = knn.predict(Yts[0][:,:-1]) 
                    self.preds[0] = predicted_label

            elif self.classifier == 'svm':
                if self.datasource == 'synthetic':
                    svn_clf = SVC(gamma='auto').fit(Xts[0][:,:-1], Yts[0][:,-1])
                    predicted_label = svn_clf.predict(Yts[1][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'UNSW':
                    svn_clf = SVC(kernel='rbf').fit(self.all_data[:,:-1], self.all_data[:,-1]) # use the entire training data
                    self.train_model = svn_clf
                    predicted_label = svn_clf.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label

            elif self.classifier == 'logistic_regression':
                if self.datasource == 'UNSW':
                    lg_rg = LogisticRegression()
                    lg_rg.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = lg_rg
                    predicted_label = lg_rg.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    # TODO: Need to develop synthetic 
                    exit()
            
            elif self.classifier == 'random_forest':
                if self.datasource == 'UNSW':
                    rf = RandomForestClassifier()
                    rf.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = rf
                    predicted_label = rf.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()
            
            elif self.classifier == 'adaboost':
                if self.datasource == 'UNSW':
                    ada = AdaBoostClassifier()
                    ada.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = ada
                    predicted_label = ada.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()
            
            elif self.classifier == 'decision_tree':
                if self.datasource == 'UNSW':
                    dt = tree.DecisionTreeClassifier()
                    dt.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = dt
                    predicted_label = dt.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()

            elif self.classifier == 'knn':
                if self.datasource == 'UNSW':
                    knn = KNeighborsClassifier(n_neighbors=50)
                    knn.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = knn
                    predicted_label = knn.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()
            
            elif self.classifier == 'mlp':
                if self.datasource == 'UNSW':
                    mlp = MLPClassifier(random_state=1, max_iter=300)
                    mlp.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = mlp
                    predicted_label = mlp.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()
            
            elif self.classifier == 'naive_bayes':
                if self.datasource == 'UNSW':
                    nb = BernoulliNB()
                    nb.fit(self.all_data[:,:-1], self.all_data[:,-1])
                    self.train_model = nb
                    predicted_label = nb.predict(Yts[0][:,:-1])
                    self.preds[0] = predicted_label
                elif self.datasource == 'synthetic':
                    exit()

            elif self.classifier == 'lstm':
                if self.datasource == 'UNSW':
                    num_classes = len(set(self.all_data[:,-1]))
                    trainLabel = tf.keras.utils.to_categorical(self.all_data[:,-1], num_classes=num_classes)
                    # Define the input shapeinput_shape = (timesteps, input_dim)  
                    # adjust the values according to your data
                    tsteps = 1000 
                    input_dim = np.shape(self.all_data[:,:-1])[1]
                    input_shape = (tsteps, input_dim)

                    # Define the LSTM model
                    model = Sequential()
                    model.add(LSTM(128, input_shape=input_shape))
                    model.add(Dense(num_classes, activation='softmax'))

                    # Compile the model
                    model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])

                    # Print the model summary
                    model.summary()
                    # Train the model
                    trainDataReshaped = np.expand_dims(self.all_data[:,:-1], axis=1)
                    model.fit(trainDataReshaped, trainLabel, batch_size=32, epochs=10, validation_split=0.2)
                    self.train_model = model
                    testDataReshaped = np.expand_dims(Yts[0][:,:-1], axis=1)
                    predicted_label = model.predict(testDataReshaped)
                    self.preds[0] = tf.argmax(predicted_label, axis=1).numpy() 
                    

                elif self.datasource == 'synthetic':
                    exit()

            elif self.classifier == 'gru':
                if self.datasource == 'UNSW':
                    num_classes = len(set(self.all_data[:,-1]))
                    trainLabel = tf.keras.utils.to_categorical(self.all_data[:,-1], num_classes=num_classes)
                    sequence_length = 1000
                    input_dim = np.shape(self.all_data[:,:-1])[1] 
                    # Define the input shape and number of hidden units
                    input_shape = (sequence_length, input_dim)  # e.g., (10, 32)
                    hidden_units = 64
                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.GRU(hidden_units, input_shape=input_shape))
                    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

                    # Compile the model
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                    # Train the model
                    trainDataReshaped = np.expand_dims(self.all_data[:,:-1], axis=1)
                    model.fit(trainDataReshaped, trainLabel, batch_size=32, epochs=10, validation_split=0.2)
                    self.train_model = model
                    testDataReshaped = np.expand_dims(Yts[0][:,:-1], axis=1)
                    predicted_label = model.predict(testDataReshaped)
                    self.preds[0] = tf.argmax(predicted_label, axis=1).numpy()

                elif self.datasource == 'synthetic':
                    exit()

                
            self.T = len(Yts)      

            # empty sets for pool and labels
            pool_data = []
            pool_label = []
            pool_index = 0
            past_centroid = self.cluster.cluster_centers_


            labeled_data_labels = Xts
            labeled_data = Yts
            
            # run the experiment 
            for t in tqdm(range(self.T-1), position=0, leave=True): 
                # get the data from time T and resample if required
                # it seems that the algo takes in the labeled data labels and the labeled data as inputs 
                if self.datasource == 'synthetic':
                    if self.classifier == '1nn':
                        if t == 0: 
                            Xt, Yt = np.array(labeled_data_labels[t]), np.array(labeled_data[t])       # Xt = train labels ; Yt = train data
                            Xe, Ye = np.array(labeled_data_labels[t+1]), np.array(Yts[t+1])            # Xe = test labels ; Ye = test data
                        else: 
                            Xt, Yt = np.array(labeled_data_labels), np.array(labeled_data)             # Xt = train labels ; Yt = train data
                            Xe, Ye = np.array(labeled_data_labels), np.array(Yts[t+1])                 # Xe = test labels ; Ye = test data
                    elif self.classifier == 'svm': 
                        if t == 0:
                            Xt, Yt = np.array(labeled_data_labels[t]), np.array(Yts[t])                # Xt = train labels ; Yt = train data
                            Xe, Ye = np.array(Xts), np.array(Yts[t+1])                                 # Xe = test labels ; Ye = test data
                        else:
                            Xt, Yt = np.array(labeled_data_labels), np.array(labeled_data)             # Xt = train labels ; Yt = train data
                            Xe, Ye = np.array(labeled_data_labels), np.array(Yts[t+1])                 # Xe = test labels ; Ye = test data
                elif self.datasource == 'UNSW':
                    if t == 0: 
                        Xt, Yt = np.array(labeled_data_labels[t]), np.array(Yts[t])       # Xt = train labels ; Yt = train data
                        Xe, Ye = np.array(Xts), np.array(Yts[t])            # Xe = test labels ; Ye = test data
                    else: 
                        Xt, Yt = np.array(labeled_data_labels), np.array(labeled_data)             # Xt = train labels ; Yt = train data
                        Xe, Ye = np.array(labeled_data_labels), np.array(Yts[t])                 # Xe = test labels ; Ye = test data

                t_start = time.time()            

                if self.resample == True:
                    N = len(Yt)
                    V = len(Xt)
                    ii = np.random.randint(0, N, N)
                    jj = np.random.randint(0, V,V)
                    Xt = Xt[jj] 
                    Yt =  Yt[ii]

                if t == 0:
                    pool_data = Ye
                    pool_label = np.array(predicted_label)
                    pool_index += 1
                else:
                    if self.datasource == 'synthetic':
                        if self.classifier == '1nn':
                            knn_mdl = KNeighborsClassifier(n_neighbors=1).fit(Yt, Xt)    # fit(train_data, train_label)
                            predicted_label = knn_mdl.predict(Ye)
                        elif self.classifier == 'svm':
                            svm_mdl = SVC(kernel='rbf').fit(Yt[:,:-1], Yt[:,-1])        # fit(Xtrain, X_label_train)
                            predicted_label = svm_mdl.predict(Ye[:,:-1])
                    elif self.datasource == 'UNSW':
                        if self.classifier == 'lstm':
                            YeReshaped = np.expand_dims(Ye[:,:-1], axis=1)
                            preds = self.train_model.predict(YeReshaped)
                            predicted_label = tf.argmax(preds, axis=1).numpy() 
                        elif self.classifier == 'gru': 
                            YeReshaped = np.expand_dims(Ye[:,:-1], axis=1)
                            preds = self.train_model.predict(YeReshaped)
                            predicted_label = tf.argmax(preds, axis=1).numpy() 
                        else:
                            predicted_label = self.train_model.predict(Ye[:,:-1])
                    
                    pool_data = np.vstack((pool_data, Ye))

                    # remove extra dimensions from pool label
                    pool_label = np.squeeze(pool_label)
                    predicted_label = np.squeeze(predicted_label)
                    self.preds[t] = predicted_label
                    
                    pool_label = np.concatenate((pool_label, predicted_label))
                    
                    if t > 0:
                        sbrt_pool_lbl = list(pool_label)
                        sbrt_pool_lbl.pop(0)
                    pool_label = np.array(pool_label)
                    
                    pool_index += 1
                concordant_label_count = 0

                # if |pool| == maxpoolsize
                if len(pool_label) > self.maxpool:
                    # C <- Clustering(pool, k)
                    temp_current_centroids = KMeans(n_clusters=self.Kclusters, init=past_centroid, n_init='auto').fit(pool_data).cluster_centers_
                    # find the label for the current centroids               
                    # new labeled data
                    new_label_data = np.zeros(np.shape(temp_current_centroids)[1])
                    for k in range(self.Kclusters):
                        if self.classifier == '1nn':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = KNeighborsClassifier(n_neighbors=1).fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)
                            
                            # _,new_label_data = nearestData.kneighbors([temp_current_centroids[k]])
                            # new_label_data = np.vstack(new_label_data[0][0])
                            # nearestData = Bknn(k=0, problem=1, metric=0)
                            # nearestData.fit(past_centroid, temp_current_centroids)
                            # centroid_label = nearestData.predict(temp_current_centroids[k])[0]
                            # new_label_data = np.vstack((new_label_data[0], centroid_label))
                            
                        elif self.classifier == 'svm':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = SVC(kernel='rbf').fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)
                        
                        elif self.classifier == 'logistic_regression':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = LogisticRegression().fit(X=past_centroid[:,:-1], y= t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)

                        elif self.classifier == 'random_forest':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = RandomForestClassifier().fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)
                        
                        elif self.classifier == 'adaboost':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = RandomForestClassifier().fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)

                        elif self.classifier == 'decision_tree':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = tree.DecisionTreeClassifier().fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)

                        elif self.classifier == 'knn':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = KNeighborsClassifier(n_neighbors=10).fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)

                        elif self.classifier == 'mlp':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = MLPClassifier(random_state=1, max_iter=300).fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)

                        elif self.classifier == 'naive_bayes':
                            label_encoder = preprocessing.LabelEncoder()
                            t_cur_centroid = label_encoder.fit_transform(temp_current_centroids[:,-1])
                            nearestData = BernoulliNB().fit(past_centroid[:,:-1], t_cur_centroid)
                            centroid_label = nearestData.predict(temp_current_centroids[k:,:-1])
                            new_label_data = np.vstack(centroid_label)
                        
                        elif self.classifier == 'lstm':
                            num_classes = len(set(temp_current_centroids[:,-1]))
                            trainLabel = tf.keras.utils.to_categorical(temp_current_centroids[:,-1], num_classes=num_classes)
                            # Define the input shapeinput_shape = (timesteps, input_dim)  
                            # adjust the values according to your data
                            tsteps = 1
                            input_dim = np.shape(past_centroid[:,:-1])[1]
                            input_shape = (tsteps, input_dim)

                            # Define the LSTM model
                            nearestData = Sequential()
                            nearestData.add(LSTM(128, input_shape=input_shape))
                            nearestData.add(Dense(num_classes, activation='softmax'))

                            # Compile the model
                            nearestData.compile(loss='categorical_crossentropy',
                                        optimizer='adam',
                                        metrics=['accuracy'])

                            # Print the model summary
                            nearestData.summary()
                            # Train the model
                            trainDataReshaped = np.expand_dims(past_centroid[:,:-1], axis=1)
                            nearestData.fit(trainDataReshaped, trainLabel, batch_size=32, epochs=10, validation_split=0.2)
                            testDataReshaped = np.expand_dims(temp_current_centroids[k:,:-1], axis=1)
                            centroid_label = nearestData.predict(testDataReshaped)
                            new_label_data = tf.argmax(centroid_label, axis=1).numpy()
                            
                            # new_label_data = np.vstack(predicted_label)
                        
                        elif self.classifier == 'gru':
                            num_classes = len(set(temp_current_centroids[:,-1]))
                            trainLabel = tf.keras.utils.to_categorical(temp_current_centroids[:,-1], num_classes=num_classes)
                            sequence_length = 1 
                            input_dim = np.shape(past_centroid[:,:-1])[1] 
                            # Define the input shape and number of hidden units
                            input_shape = (sequence_length, input_dim)  # e.g., (10, 32)
                            hidden_units = 64
                            nearestData = tf.keras.Sequential()
                            nearestData.add(tf.keras.layers.GRU(hidden_units, input_shape=input_shape))
                            nearestData.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

                            # Compile the model
                            nearestData.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                            # Train the model
                            trainDataReshaped = np.expand_dims(past_centroid[:,:-1], axis=1)
                            nearestData.fit(trainDataReshaped, trainLabel, batch_size=32, epochs=10, validation_split=0.2)
                            testDataReshaped = np.expand_dims(temp_current_centroids[k:,:-1], axis=1)
                            centroid_label = nearestData.predict(testDataReshaped)
                            new_label_data = tf.argmax(centroid_label, axis=1).numpy()
                            # new_label_data = np.vstack(predicted_label)
                    
                    new_label_data = list(new_label_data)
                    new_label_data.pop(0)
                    new_label_data = np.array(new_label_data)
                    
                    # concordant data 
                    for l in range(0, len(pool_data)):
                        if pool_data[l][-1] == 1:
                            concordant_label_count += 1
                    
                    if concordant_label_count != 1:
                        labeled_data = pool_data
                        labeled_data_labels = new_label_data
                        past_centroid = temp_current_centroids
                    
                    Ye = np.squeeze(Ye)
                    Ye = np.array(Ye[:,-1])

                    # reset 
                    pool_data = np.zeros(np.shape(pool_data)[1])
                    pool_label = np.zeros(np.shape(pool_data))
                    pool_index = 0    
                
                t_end = time.time() 
                # needed to have same shape as preds and test
                indx = np.arange(np.shape(self.preds[t])[0])
                indx = np.squeeze(indx) 
                perf_metric = cp.PerformanceMetrics(timestep= t, preds= self.preds[t], test= Ye[indx], \
                                                    dataset= self.dataset , method= '' , \
                                                    classifier= self.classifier, tstart=t_start, tend=t_end)
                self.performance_metric[t] = perf_metric.findClassifierMetrics(preds= self.preds[t], test= Ye[indx])

            total_time_end = time.time()

            self.total_time = total_time_end - total_time_start
            avg_metrics = cp.PerformanceMetrics(tstart= total_time_start, tend= total_time_end)
            self.avg_perf_metric = avg_metrics.findAvePerfMetrics(total_time=self.total_time, perf_metrics= self.performance_metric)
            return self.avg_perf_metric



# run_scargc_svm = SCARGC(classifier = 'gru', dataset= 'ton_iot_fridge', datasource='UNSW').run()
# print(run_scargc_svm)