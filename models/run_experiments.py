#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          run_experiments.py
Author:             Martin Manuel Lopez
Creation:           03/26/2022

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

from tkinter import E
import compose
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import math
import pickle5 as pickle 
import time


class RunExperiment:

    def __init__(self, experiements = [], classifier = [], verbose =[], datasets=[], num_cores= 0.8):
        self.experiments = experiements
        self.classifier = classifier
        self.verbose = verbose
        self.datasets = datasets
        self.results = {}
        self.num_cores = num_cores
        

    def plot_results(self):
        experiments = self.results.keys()
        fig_handle = plt.figure()
        for experiment in experiments:
            df = pd.DataFrame(self.results[experiment])
            df.plot(label=experiment, x='Timesteps', y='Accuracy')
        plt.title('Accuracy over timesteps')
        plt.xlabel('Timesteps')
        plt.legend
        plt.ylabel('% Accuracy')
        plt.show()
        
        with open('results_plot.pkl', 'wb') as result_plot:
            pickle.dump(fig_handle, result_plot)

    def run(self):
        
        for i in self.experiments:
            for j in self.classifier:
                for dataset in self.datasets:
                    experiment = dataset + '_' + j + '_' + i
                    if i == 'fast_compose' and j == 'QN_S3VM':
                        fast_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", method="gmm", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        start_time = time.time()
                        self.results[experiment] = fast_compose_QNS3VM.run()
                        end_time = time.time()
                        total_time = end_time - start_time
                        fast_compose_label_prop.avg_results_dict['Total_Time'] = total_time
                        results_df = pd.DataFrame.from_dict((fast_compose_QNS3VM.avg_results_dict.keys(), fast_compose_QNS3VM.avg_results_dict.values())).T
                        results_df.to_pickle('results_fast_compose_QN_S3VM.pkl')
                        with open('total_time_fast_compose_QN_S3VM.pkl', 'wb') as f:
                            total_time_pkl = pickle.dump(total_time,f)
                        with open('total_time_fast_compose_QN_S3VM.pkl', 'rb') as file:
                            loaded_total_time_pkl = pickle.load(file)
                        results_pkl = pd.read_pickle('results_fast_compose_QN_S3VM.pkl')
                        print("time pickle: \n", loaded_total_time_pkl )
                        print("results pickle:\n " , results_pkl )

                    elif i == 'fast_compose' and j == 'label_propagation':
                        fast_compose_label_prop = compose.COMPOSE(classifier="label_propagation", method="gmm", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        start_time = time.time()
                        self.results[experiment] = fast_compose_label_prop.run()
                        end_time = time.time()
                        total_time = end_time - start_time 
                        fast_compose_label_prop.avg_results_dict['Total_Time'] = total_time
                        results_df = pd.DataFrame.from_dict((fast_compose_label_prop.avg_results_dict.keys(), fast_compose_label_prop.avg_results_dict.values())).T
                        results_df.to_pickle('results_fast_compose_label_propagation.pkl')
                        with open('total_time_fast_compose_label_propagation.pkl', 'wb') as f:
                            total_time_pkl = pickle.dump(total_time,f)
                        with open('total_time_fast_compose_label_propagation.pkl', 'rb') as file:
                            loaded_total_time_pkl = pickle.load(file)
                        results_pkl = pd.read_pickle('results_fast_compose_label_propagation.pkl')
                        print("time pickle: \n", loaded_total_time_pkl )
                        print("results pickle:\n" , results_pkl )
                        
                    elif i == 'compose' and j == 'QN_S3VM':
                        reg_compose_label_QN_S3VM = compose.COMPOSE(classifier="QN_S3VM", method="a_shape", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        start_time = time.time()
                        self.results[experiment] = reg_compose_label_QN_S3VM.run()
                        end_time = time.time()
                        total_time = end_time - start_time
                        fast_compose_label_prop.avg_results_dict['Total_Time'] = total_time
                        results_df = pd.DataFrame.from_dict((reg_compose_label_QN_S3VM.avg_results_dict.keys(), reg_compose_label_QN_S3VM.avg_results_dict.values())).T
                        results_df.to_pickle('results_compose_QN_S3VM.pkl')
                        with open('total_time_compose_QN_S3VM.pkl', 'wb') as f:
                            total_time_pkl = pickle.dump(total_time,f)
                        with open('total_time_compose_QN_S3VM.pkl', 'rb') as file:
                            loaded_total_time_pkl = pickle.load(file)
                        results_pkl = pd.read_pickle('results_compose_QN_S3VM.pkl')
                        print("Total Time : \n", loaded_total_time_pkl )
                        print("Result:\n" , results_pkl )

                    elif i == 'compose' and j == 'label_propagation':
                        reg_compose_label_prop = compose.COMPOSE(classifier="label_propagation", method="a_shape", verbose = self.verbose ,num_cores= self.num_cores, selected_dataset = dataset)
                        start_time = time.time()
                        self.results[experiment] = reg_compose_label_prop.run()
                        end_time = time.time()
                        total_time = end_time - start_time     
                        fast_compose_label_prop.avg_results_dict['Total_Time'] = total_time       
                        results_df = pd.DataFrame.from_dict((reg_compose_label_prop.avg_results_dict.keys(), reg_compose_label_prop.avg_results_dict.values())).T
                        results_df.to_pickle('results_compose_label_propagation.pkl')
                        with open('total_time_fast_compose_label_propagation.pkl', 'wb') as f:
                            total_time_pkl = pickle.dump(total_time,f)
                        with open('total_time_fast_compose_label_propagation.pkl', 'rb') as file:
                            loaded_total_time_pkl = pickle.load(file)
                        results_pkl = pd.read_pickle('results_compose_label_propagation.pkl')
                        print("Total Time : \n", loaded_total_time_pkl )
                        print("Result:\n" , results_pkl )
        
        self.plot_results()


run_experiment = RunExperiment(experiements=['fast_compose', 'compose'], classifier=['label_propagation'], verbose=0, datasets=[ 'UG_2C_2D','MG_2C_2D','1CDT', '2CDT'], num_cores=0.8)
run_experiment.run()


