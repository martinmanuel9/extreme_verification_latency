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
    def grapher(self, path_name, labels, title, linestyle ):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/" + path_name
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
        experiments = labels
        style = linestyle
        timesteps = {}
        accuracy = {}
        classifier = {}
        plotline = {}
        for i in range(len(result)):
            timesteps[i] = result[i].loc[12].at[1]
            accuracy[i] = result[i].loc[13].at[1]
            classifier[i] = experiments[i]
            plotline[i] = style[i]
            
        data = {}
        for i, item in enumerate(classifier.values()):
            data[item] = pd.DataFrame({'Timesteps':timesteps[i], 'Accuracy': accuracy[i]})

        for i, item in enumerate(data.keys()):
            # sns.set_theme(style='darkgrid')
            sns.set_style('whitegrid')
            plt.plot(data[item]['Timesteps'][::8], data[item]['Accuracy'][::8], label=item, linestyle= plotline[i])
                
        plt.xlabel('Timesteps')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.gcf().set_size_inches(15,10)  
        plt.xlim([0,90])
        plt.xticks(range(0, 90, 10))
        plt.show()

    def run(self):
        mlp_bot = Grapher()
        mlp_bot.grapher(path_name='plotter_scargc/MLP/Bot', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of Bot IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_Fridge = Grapher()
        mlp_Fridge.grapher(path_name='plotter_scargc/MLP/Fridge', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Fridge IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_Garage = Grapher()
        mlp_Garage.grapher(path_name='plotter_scargc/MLP/Garage', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Garage IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_GPS = Grapher()
        mlp_GPS.grapher(path_name='plotter_scargc/MLP/GPS', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN GPS IoT Datasets',
        linestyle= ['solid', 'dotted'])
        
        mlp_Light = Grapher()
        mlp_Light.grapher(path_name='plotter_scargc/MLP/Light', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Light IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_modbus = Grapher()
        mlp_modbus.grapher(path_name='plotter_scargc/MLP/Modbus', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Modbus IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_thermo = Grapher()
        mlp_thermo.grapher(path_name='plotter_scargc/MLP/Thermo', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Thermostat IoT Datasets',
        linestyle= ['solid', 'dotted'])

        mlp_weather = Grapher()
        mlp_weather.grapher(path_name='plotter_scargc/MLP/Weather', 
        labels=['SCARGC', 'Base Classifer'],
        title='Multi-Layer Perceptron Accuracy Comparison of ToN Weather IoT Datasets',
        linestyle= ['solid', 'dotted'])

        lr_bot = Grapher()
        lr_bot.grapher(path_name='plotter_scargc/Logistic_Regression/Bot', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of Bot IoT Datasets',
        linestyle= ['solid', 'dotted'])

        lr_fridge = Grapher()
        lr_fridge.grapher(path_name='plotter_scargc/Logistic_Regression/Fridge', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Fridge Datasets',
        linestyle= ['solid', 'dotted'])

        lr_garage = Grapher()
        lr_garage.grapher(path_name='plotter_scargc/Logistic_Regression/Garage', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Garage Datasets',
        linestyle= ['solid', 'dotted'])

        lr_gps = Grapher()
        lr_gps.grapher(path_name='plotter_scargc/Logistic_Regression/GPS', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN GPS Datasets',
        linestyle= ['solid', 'dotted'])

        lr_light = Grapher()
        lr_light.grapher(path_name='plotter_scargc/Logistic_Regression/Light', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Light Datasets',
        linestyle= ['solid', 'dotted'])

        lr_modbus = Grapher()
        lr_modbus.grapher(path_name='plotter_scargc/Logistic_Regression/Modbus', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Modbus Datasets',
        linestyle= ['solid', 'dotted'])

        lr_thermo = Grapher()
        lr_thermo.grapher(path_name='plotter_scargc/Logistic_Regression/Thermo', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Thermostat Datasets',
        linestyle= ['solid', 'dotted'])

        lr_weather= Grapher()
        lr_weather.grapher(path_name='plotter_scargc/Logistic_Regression/Weather', 
        labels=['SCARGC', 'Base Classifer'],
        title='Logistic Regression Accuracy Comparison of ToN Weather Datasets',
        linestyle= ['solid', 'dotted'])

graph = Grapher()
graph.run()

# %%
