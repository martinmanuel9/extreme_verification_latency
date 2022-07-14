#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
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
import scargc
import os
import time
from pathlib import Path  

class RunExperiment:

    def __init__(self, experiements = [], classifier = [], verbose =[], datasets=[], num_cores= 0.8):
        self.experiments = experiements
        self.classifier = classifier
        self.verbose = verbose
        self.datasets = datasets
        self.results = {}
        self.num_cores = num_cores
        
    def change_dir(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/results"
        os.chdir(path)

    def plot_results(self):
        experiments = self.results.keys()
        fig_handle = plt.figure()

        fig, ax = plt.subplots()
        for experiment in experiments:
            df = pd.DataFrame(self.results[experiment])
            
            df.plot(ax=ax, label=experiment, x='Timesteps', y='Accuracy')
        
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        plt.title('Accuracy over timesteps' + time_stamp )
        plt.xlabel('Timesteps')
        plt.tight_layout()
        plt.legend
        plt.ylabel('% Accuracy')
        plt.show()
        
        # change the directory to your particular files location
        self.change_dir()
        result_plot_pkl = 'results_plot_' + time_stamp + '.pkl'
        path = str(Path.home())
        path = path + "/extreme_verification_latency/plots"
        os.chdir(path) 
        with open(result_plot_pkl, 'wb') as result_plot:
            pickle.dump(fig_handle, result_plot)

    def run(self):
        
        for i in self.experiments:
            for j in self.classifier:
                for dataset in self.datasets:
                    experiment = dataset + '_' + j + '_' + i
                    if i == 'fast_compose' and j == 'QN_S3VM':
                        fast_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", method="fast_compose", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = fast_compose_QNS3VM.run()
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                        fast_compose_QNS3VM.avg_results_dict['Time_Stamp'] = time_stamp 
                        results_df = pd.DataFrame.from_dict((fast_compose_QNS3VM.avg_results_dict.keys(), fast_compose_QNS3VM.avg_results_dict.values())).T
                        # change the directory to your particular files location
                        self.change_dir()
                        results_fast_compose_qns3vm = 'results_fast_compose_QN_S3VM_'+ time_stamp +'.pkl'
                        results_df.to_pickle(results_fast_compose_qns3vm)
                        results_pkl = pd.read_pickle(results_fast_compose_qns3vm)
                        print("Results:\n " , results_pkl )

                    elif i == 'fast_compose' and j == 'label_propagation':
                        fast_compose_label_prop = compose.COMPOSE(classifier="label_propagation", method="fast_compose", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = fast_compose_label_prop.run()
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                        fast_compose_label_prop.avg_results_dict['Time_Stamp'] = time_stamp
                        results_df = pd.DataFrame.from_dict((fast_compose_label_prop.avg_results_dict.keys(), fast_compose_label_prop.avg_results_dict.values())).T
                        # change the directory to your particular files location
                        self.change_dir()
                        results_fast_compose_lbl_prop = 'results_fast_compose_label_propagation_'+ time_stamp + '.pkl'
                        results_df.to_pickle(results_fast_compose_lbl_prop)
                        results_pkl = pd.read_pickle(results_fast_compose_lbl_prop)
                        print("Results:\n" , results_pkl )
                        
                    elif i == 'compose' and j == 'QN_S3VM':
                        reg_compose_label_QN_S3VM = compose.COMPOSE(classifier="QN_S3VM", method="a_shape", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = reg_compose_label_QN_S3VM.run()
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                        reg_compose_label_QN_S3VM.avg_results_dict['Time_Stamp'] = time_stamp
                        results_df = pd.DataFrame.from_dict((reg_compose_label_QN_S3VM.avg_results_dict.keys(), reg_compose_label_QN_S3VM.avg_results_dict.values())).T
                        # change the directory to your particular files location
                        self.change_dir()
                        results_compose_qns3vm = 'results_compose_QN_S3VM_'+ time_stamp +'.pkl'
                        results_df.to_pickle(results_compose_qns3vm)
                        results_pkl = pd.read_pickle(results_compose_qns3vm)
                        print("Results:\n" , results_pkl )

                    elif i == 'compose' and j == 'label_propagation':
                        reg_compose_label_prop = compose.COMPOSE(classifier="label_propagation", method="a_shape", verbose = self.verbose ,num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = reg_compose_label_prop.run()
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")   
                        reg_compose_label_prop.avg_results_dict['Time_Stamp'] = time_stamp  
                        results_df = pd.DataFrame.from_dict((reg_compose_label_prop.avg_results_dict.keys(), reg_compose_label_prop.avg_results_dict.values())).T
                        # change the directory to your particular files location
                        self.change_dir()
                        results_compose_lbl_prop = 'results_compose_label_propagation_' + time_stamp + '.pkl' 
                        results_df.to_pickle(results_compose_lbl_prop)
                        results_pkl = pd.read_pickle(results_compose_lbl_prop)
                        print("Results:\n" , results_pkl )
                    
                    elif i == 'scargc' and j == '1nn': 
                        scargc_1nn_data = scargc.SetData(dataset= dataset)
                        run_scargc_1nn = scargc.SCARGC(Xinit= scargc_1nn_data.X, Yinit= scargc_1nn_data.Y , classifier = '1nn', dataset= dataset)
                        self.results[experiment] = run_scargc_1nn.run(Xts = scargc_1nn_data.X, Yts = scargc_1nn_data.Y)
                        results_df = pd.DataFrame.from_dict((run_scargc_1nn.avg_results.keys(), run_scargc_1nn.avg_results.values())).T
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                        # change the directory to your particular files location
                        self.change_dir()
                        results_scargc_1nn = 'results_scargc_1nn_'+ time_stamp + '.pkl'
                        results_df.to_pickle(results_scargc_1nn)
                        results_pkl = pd.read_pickle(results_scargc_1nn)
                        print("Results:\n", results_df)

                    elif i == 'scargc' and j == 'svm': 
                        scargc_svm_data = scargc.SetData(dataset= dataset)
                        run_scargc_svm = scargc.SCARGC(Xinit= scargc_svm_data.X[0], Yinit= scargc_svm_data.Y , classifier = 'svm', dataset= dataset)
                        self.results[experiment] = run_scargc_svm.run(Xts = scargc_svm_data.X[0], Yts = scargc_svm_data.Y)              # .X[0] only get initial training set
                        results_df = pd.DataFrame.from_dict((run_scargc_svm.avg_results.keys(), run_scargc_svm.avg_results.values())).T
                        time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                        # change the directory to your particular files location
                        self.change_dir()
                        results_scargc_svm = 'results_scargc_svm_'+ time_stamp + '.pkl' 
                        results_df.to_pickle(results_scargc_svm)
                        results_pkl = pd.read_pickle(results_scargc_svm)
                        print("Results:\n", results_df)
        
        self.plot_results()

run_experiment = RunExperiment(experiements=['fast_compose', 'compose'], classifier=['label_propagation'], verbose=0, datasets=[ 'UG_2C_2D','MG_2C_2D','1CDT', '2CDT'], num_cores=0.9)
run_experiment.run()

run_experiment = RunExperiment(experiements=['scargc'], classifier=['svm'], verbose=0, datasets=[ 'UG_2C_2D','MG_2C_2D','1CDT', '2CDT'], num_cores=0.9)
run_experiment.run()