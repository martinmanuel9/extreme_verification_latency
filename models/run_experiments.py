#%%
#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
File name:          run_experiments.py
Author:             Martin Manuel Lopez
Creation:           03/26/2022

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

import compose
from matplotlib import pyplot as plt
import pandas as pd
import pickle as pickle 
import time
import scargc
import os
import time
from pathlib import Path 

class RunExperiment:

    def __init__(self, experiements = [], classifier = [], datasets=[], modes=[], num_cores= 0.8):
        self.experiments = experiements
        self.classifier = classifier
        self.datasets = datasets
        self.results = {}
        self.num_cores = num_cores
        self.modes = modes 
        
    def change_dir(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/results"
        os.chdir(path)

    def plot_results(self):
        # change the directory to your particular files location
        self.change_dir()
        path = str(Path.home())
        path = path + "/extreme_verification_latency/plots"
        os.chdir(path)
        experiments = self.results.keys()
        fig_handle = plt.figure()
        fig, ax = plt.subplots()
        result_plot = {}
        for experiment in experiments:
            result_plot['Timesteps'] = self.results[experiment]['Timesteps']
            result_plot['Accuracy'] = self.results[experiment]['Accuracy']
            df = pd.DataFrame(result_plot)
            df.plot(ax=ax, label=experiment, x='Timesteps', y='Accuracy')
            
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        plt.title('Accuracy over timesteps' + time_stamp )
        plt.xlabel('Timesteps')
        plt.tight_layout()
        plt.legend
        plt.ylabel('% Accuracy')
        plt.gcf().set_size_inches(15,10)  
        plt.show()
        results_pkl = 'result_plot_data_' + f'{time_stamp}' + '.pkl'
        with open(f'{results_pkl}', 'wb') as result_data:
            pickle.dump(self.results, result_data)

    def run(self):
        
        for i in self.experiments:
            for j in self.classifier:
                for dataset in self.datasets:
                    for mode in self.modes:
                        experiment = dataset + '_' + j + '_' + i
                        if i == 'fast_compose' and j == 'QN_S3VM':
                            fast_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", mode= mode,  method="fast_compose", num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = fast_compose_QNS3VM.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                            fast_compose_QNS3VM.avg_perf_metric['Time_Stamp'] = time_stamp 
                            results_df = pd.DataFrame.from_dict((fast_compose_QNS3VM.avg_perf_metric.keys(), fast_compose_QNS3VM.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_fast_compose_qns3vm = 'results_fast_compose_QN_S3VM_'+ time_stamp +'.pkl'
                            results_df.to_pickle(results_fast_compose_qns3vm)
                            results_pkl = pd.read_pickle(results_fast_compose_qns3vm)
                            print("Results:\n " , results_pkl )

                        elif i == 'fast_compose' and j == 'label_propagation':
                            fast_compose_label_prop = compose.COMPOSE(classifier="label_propagation", mode= mode, method="fast_compose",  num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = fast_compose_label_prop.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                            fast_compose_label_prop.avg_perf_metric['Time_Stamp'] = time_stamp
                            results_df = pd.DataFrame.from_dict((fast_compose_label_prop.avg_perf_metric.keys(), fast_compose_label_prop.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_fast_compose_lbl_prop = 'results_fast_compose_label_propagation_'+ time_stamp + '.pkl'
                            results_df.to_pickle(results_fast_compose_lbl_prop)
                            results_pkl = pd.read_pickle(results_fast_compose_lbl_prop)
                            print("Results:\n" , results_pkl )
                        
                        elif i == 'fast_compose' and j == 'svm':
                            fast_compose_svm = compose.COMPOSE(classifier="svm", mode= mode, method="fast_compose",  num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = fast_compose_svm.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                            fast_compose_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                            results_df = pd.DataFrame.from_dict((fast_compose_svm.avg_perf_metric.keys(), fast_compose_svm.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_fast_compose_svm = 'results_fast_compose_svm_'+ time_stamp + '.pkl'
                            results_df.to_pickle(results_fast_compose_svm)
                            results_pkl = pd.read_pickle(results_fast_compose_svm)
                            print("Results:\n" , results_pkl )

                            
                        elif i == 'compose' and j == 'QN_S3VM':
                            reg_compose_label_QN_S3VM = compose.COMPOSE(classifier="QN_S3VM", mode= mode ,method="compose", num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = reg_compose_label_QN_S3VM.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                            reg_compose_label_QN_S3VM.avg_perf_metric['Time_Stamp'] = time_stamp
                            results_df = pd.DataFrame.from_dict((reg_compose_label_QN_S3VM.avg_perf_metric.keys(), reg_compose_label_QN_S3VM.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_compose_qns3vm = 'results_compose_QN_S3VM_'+ time_stamp +'.pkl'
                            results_df.to_pickle(results_compose_qns3vm)
                            results_pkl = pd.read_pickle(results_compose_qns3vm)
                            print("Results:\n" , results_pkl )

                        elif i == 'compose' and j == 'label_propagation':
                            reg_compose_label_prop = compose.COMPOSE(classifier="label_propagation", mode= mode, method="compose", num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = reg_compose_label_prop.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")   
                            reg_compose_label_prop.avg_perf_metric['Time_Stamp'] = time_stamp  
                            results_df = pd.DataFrame.from_dict((reg_compose_label_prop.avg_perf_metric.keys(), reg_compose_label_prop.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_compose_lbl_prop = 'results_compose_label_propagation_' + time_stamp + '.pkl' 
                            results_df.to_pickle(results_compose_lbl_prop)
                            results_pkl = pd.read_pickle(results_compose_lbl_prop)
                            print("Results:\n" , results_pkl )

                        elif i == 'compose' and j == 'svm':
                            reg_compose_svm = compose.COMPOSE(classifier="svm", mode= mode ,method="compose", num_cores= self.num_cores, selected_dataset = dataset)
                            self.results[experiment] = reg_compose_svm.run()
                            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                            reg_compose_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                            results_df = pd.DataFrame.from_dict((reg_compose_svm.avg_perf_metric.keys(), reg_compose_svm.avg_perf_metric.values())).T
                            # change the directory to your particular files location
                            self.change_dir()
                            results_compose_svm = 'results_compose_QN_S3VM_'+ time_stamp +'.pkl'
                            results_df.to_pickle(results_compose_svm)
                            results_pkl = pd.read_pickle(results_compose_svm)
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

run_experiment = RunExperiment(experiements=['compose'], classifier=['label_propagation'], modes=['a_shape'], datasets=['UG_2C_2D','MG_2C_2D','1CDT', '2CDT'], num_cores=0.95)
run_experiment.run()
# run_experiment = RunExperiment(experiements=['scargc'], classifier=['svm'], datasets=[ 'UG_2C_2D','MG_2C_2D','1CDT', '2CDT'], num_cores=0.9)
# run_experiment.run()
#%%
