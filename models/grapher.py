#%%
#!/usr/bin/env python 
"""
Application:        EVL Graph results
File name:          grapher.py
Author:             Martin Manuel Lopez
Creation:           02/01/2023

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
import pickle as pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import os 


class Grapher():
    def graph_bot_iot(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/bot_iot"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            if item == 'COMPOSE':
                plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item, linewidth=3)
            else:
                plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item, linewidth=0.8)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of BOT IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_fridge(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_fridge"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            if item == 'COMPOSE':
                plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item, linewidth=3)
            else:
                plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item, linewidth=0.8)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Fridge IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_garage(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_garage"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Garage IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_gps(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_gps"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON GPS IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_light(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_light"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Motion Light IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_modbus(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_modbus"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Modbus IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_thermo(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_thermo"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Thermostat IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10) 
        plt.show()

    def graph_ton_iot_weather(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/ton_iot_weather"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Naive Bayes', 'SVM', 'COMPOSE Label Propagation', 'Fast COMPOSE Label Propagation', 'COMPOSE Naive Bayes', 'Fast COMPOSE Naive Bayes',
                        'COMPOSE SVM', 'Fast COMPOSE SVM']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON Weather Activity IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10)  
        plt.show()

    def graph_scargc(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/Paper_Plots/scargc"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[12] for time steps 
            # loc[13] for accuracy 
            # loc[14] for Experiment
            # loc[0] for dataset
            # result[0].loc[12].at[1]
        experiments = ['Fridge', 'Garage','GPS', 'Light','Modbus', 'Thermostat', 'Weather']
        timesteps = {}
        accuracy = {}
        classifier = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
        
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            sns.set_theme(style='darkgrid')
            # sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'], data[item]['Accuracy'], label=item)
                
        
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison of EVL Classifiers of TON IoT Activity IoT Dataset')
        plt.legend()
        plt.gcf().set_size_inches(15,10)  
        plt.show()

    def run(self):
        bot_result = Grapher()
        bot_result.graph_bot_iot()
        ton_iot_fridge = Grapher()
        ton_iot_fridge.graph_ton_iot_fridge()
        ton_iot_garage = Grapher()
        ton_iot_garage.graph_ton_iot_garage()
        ton_iot_gps = Grapher()
        ton_iot_gps.graph_ton_iot_gps()
        ton_iot_light = Grapher()
        ton_iot_light.graph_ton_iot_light()
        ton_iot_modbus = Grapher()
        ton_iot_modbus.graph_ton_iot_modbus()
        ton_iot_thermo = Grapher()
        ton_iot_thermo.graph_ton_iot_thermo()
        ton_iot_weather = Grapher()
        ton_iot_weather.graph_ton_iot_weather()
        scargc = Grapher()
        scargc.graph_scargc()

graph = Grapher()
graph.run()

# %%
