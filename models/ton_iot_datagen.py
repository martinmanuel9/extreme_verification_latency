#!/usr/bin/env python 

"""
Application:        Cyber Attacks Data Generation of IoT Devices  
File name:          ton_iot_datagen.py
Author:             Martin Manuel Lopez
Creation:           1/17/2023

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
from sklearn.model_selection import train_test_split   

class TON_IoT_Datagen():
    def __init__(self) -> None:
        self.import_data()

    def change_directory(self):
        path = str(Path.home())
        path = path + '/extreme_verification_latency/data/TON_IoT_Data/'
        os.chdir(path)

    def import_data(self):
        self.change_directory()
        self.fridge_data()
        self.garage_data()
        self.gps_data()
        self.modbus_data()
        self.light_data()
        self.thermostat_data() 
        self.weather_data()

    def fridge_data(self):
        fridge_dataset = pd.read_csv('Train_Test_IoT_Fridge.csv')
        fridge_dataset = fridge_dataset[['ts','date','time','fridge_temperature','temp_condition','type','label']]
        train_fridge, test_fridge = train_test_split(fridge_dataset, test_size=0.33)
        # print('fridge:', len(train_fridge), len(test_fridge))
        self.fridgeTrainStepsize = 400
        self.fridgeTestStepsize = 197
        self.fridgeTrainSet = train_fridge
        self.fridgeTestSet = test_fridge

    
    def garage_data(self):
        garage_dataset = pd.read_csv('Train_Test_IoT_Garage_Door.csv')
        garage_dataset = garage_dataset[['ts','date','time','door_state','sphone_signal','type', 'label']]
        train_garage, test_garage = train_test_split(garage_dataset, test_size=0.33)
        # print('garage:', len(train_garage), len(test_garage))
        self.garageTrainStepsize = 399
        self.garageTestStepsize = 196
        self.garageTrainSet = train_garage
        self.garageTestSet = test_garage

    def gps_data(self):
        gps_dataset = pd.read_csv('Train_Test_IoT_GPS_Tracker.csv')
        gps_dataset = gps_dataset[['ts','date','time','latitude','longitude','type', 'label']]
        train_gps, test_gps = train_test_split(gps_dataset, test_size=0.33)
        # print('gps:', len(train_gps), len(test_gps))
        self.gpsTrainStepsize = 395
        self.gpsTestStepsize = 194
        self.gpsTrainSet = train_gps
        self.gosTestSet = test_gps 

    def modbus_data(self):
        modbus_dataset = pd.read_csv('Train_Test_IoT_Modbus.csv')
        features  = ['ts','date','time','FC1_Read_Input_Register','FC2_Read_Discrete_Value','FC3_Read_Holding_Register','FC4_Read_Coil','type','label']
        train_modbus, test_modbus = train_test_split(modbus_dataset, test_size=0.33)
        train_modbus = train_modbus[features]
        test_modbus = test_modbus[features]
        # print('modbus:', len(train_modbus), len(test_modbus))
        self.modbusTrainStepsize = 342
        self.modbusTestStepsize = 168
        self.modbusTrainSet = train_modbus
        self.modbusTestSet = test_modbus

    
    def light_data(self):
        light_dataset = pd.read_csv('Train_Test_IoT_Motion_Light.csv')
        light_dataset = light_dataset[['ts','date','time','motion_status','light_status','type','label']]
        train_light, test_light = train_test_split(light_dataset, test_size=0.33)
        # print('light:', len(train_light), len(test_light))
        self.lightTrainStepsize = 398
        self.lightTestStepsize = 196
        self.lightTrainSet = train_light
        self.lightTestSet = test_light

    def thermostat_data(self):
        thermostat_dataset = pd.read_csv('Train_Test_IoT_Thermostat.csv')
        thermostat_dataset = thermostat_dataset[['ts','date','time','current_temperature','thermostat_status','type','label']]
        train_thermo, test_thermo = train_test_split(thermostat_dataset, test_size=0.33)
        # print('thermo', len(train_thermo), len(test_thermo))
        self.thermoTrainStepsize = 353
        self.thermoTestStepsize = 174
        self.thermoTrainSet = train_thermo
        self.thermoTestSet = test_thermo

    def weather_data(self):
        weather_dataset = pd.read_csv('Train_Test_IoT_Weather.csv')
        weather_dataset = weather_dataset[['ts','date','time','temperature','pressure','humidity','type','label']]
        train_weather, test_weather = train_test_split(weather_dataset, test_size=0.33)
        # print('weather:', len(train_weather), len(test_weather))
        self.weatherTrainStepsize = 397
        self.weatherTestStepsize = 195
        self.weatherTrainSet = train_weather
        self.weatherTestSet = test_weather
    
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield np.array(iterable[ndx:min(ndx + n, l)])

    def create_dataset(self, train_stepsize, test_stepsize, test, train):
        self.trainDict = {}
        self.testDict = {}
        trainSet = test.to_numpy()
        testSet = train.to_numpy()

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


# datagen = TON_IoT_Datagen()
# fridge_train, fridge_test =  datagen.create_dataset(train_stepsize=datagen.fridgeTrainStepsize, test_stepsize=datagen.fridgeTestStepsize, 
#             train=datagen.fridgeTrainSet, test= datagen.fridgeTestSet)
# print(fridge_train)

# weather_train, weather_test = datagen.create_dataset(train_stepsize=datagen.weatherTrainStepsize, test_stepsize=datagen.weatherTestStepsize, 
#                                 train= datagen.weatherTrainSet, test = datagen.weatherTestSet)
# print(weather_train)



