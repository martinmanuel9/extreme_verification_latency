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
import mclassification as mclass
import vanilla
import os
import time
from pathlib import Path 

class RunExperiment:

    def __init__(self, experiements = [], classifier = [], datasets=[], modes=[], datasources = [],  num_cores= 0.8):
        self.experiments = experiements
        self.classifier = classifier
        self.datasets = datasets
        self.results = {}
        self.num_cores = num_cores
        self.modes = modes 
        self.datasources = datasources 
        
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
                        for datasource in self.datasources:
                            experiment = dataset + '_' + mode + '_' + j + '_' + i
                            if i == 'fast_compose' and j == 'QN_S3VM':
                                fast_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", mode= mode,  method="fast_compose", num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = fast_compose_QNS3VM.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                fast_compose_QNS3VM.avg_perf_metric['Experiment'] = i + '_' + j
                                fast_compose_QNS3VM.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((fast_compose_QNS3VM.avg_perf_metric.keys(), fast_compose_QNS3VM.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_fast_compose_qns3vm = 'results_'+ f'{experiment}' + '.pkl'
                                results_df.to_pickle(results_fast_compose_qns3vm)
                                results_pkl = pd.read_pickle(results_fast_compose_qns3vm)
                                print("Results:\n " , results_pkl )

                            elif i == 'fast_compose' and j == 'label_propagation':
                                fast_compose_label_prop = compose.COMPOSE(classifier="label_propagation", mode= mode, method="fast_compose",  num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = fast_compose_label_prop.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                fast_compose_label_prop.avg_perf_metric['Experiment'] = i + '_' + j
                                fast_compose_label_prop.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((fast_compose_label_prop.avg_perf_metric.keys(), fast_compose_label_prop.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_fast_compose_lbl_prop = 'results_'+ f'{experiment}' +  '.pkl'
                                results_df.to_pickle(results_fast_compose_lbl_prop)
                                results_pkl = pd.read_pickle(results_fast_compose_lbl_prop)
                                print("Results:\n" , results_pkl )
                            
                            elif i == 'fast_compose' and j == 'svm':
                                fast_compose_svm = compose.COMPOSE(classifier="svm", mode= mode, method="fast_compose",  num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = fast_compose_svm.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                fast_compose_svm.avg_perf_metric['Experiment'] = i + '_' + j
                                fast_compose_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((fast_compose_svm.avg_perf_metric.keys(), fast_compose_svm.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_fast_compose_svm = 'results_'+ f'{experiment}' + '.pkl'
                                results_df.to_pickle(results_fast_compose_svm)
                                results_pkl = pd.read_pickle(results_fast_compose_svm)
                                print("Results:\n" , results_pkl )

                            elif i == 'fast_compose' and j == 'naive_bayes':
                                fast_compose_nb = compose.COMPOSE(classifier="naive_bayes", mode= mode, method="fast_compose",  num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = fast_compose_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                fast_compose_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                fast_compose_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((fast_compose_nb.avg_perf_metric.keys(), fast_compose_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_fast_compose_nb = 'results_'+ f'{experiment}' + '.pkl'
                                results_df.to_pickle(results_fast_compose_nb)
                                results_pkl = pd.read_pickle(results_fast_compose_nb)
                                print("Results:\n" , results_pkl )
                                
                            elif i == 'compose' and j == 'QN_S3VM':
                                reg_compose_label_QN_S3VM = compose.COMPOSE(classifier="QN_S3VM", mode= mode ,method="compose", num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = reg_compose_label_QN_S3VM.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                reg_compose_label_QN_S3VM.avg_perf_metric['Experiment'] = i + '_' + j
                                reg_compose_label_QN_S3VM.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((reg_compose_label_QN_S3VM.avg_perf_metric.keys(), reg_compose_label_QN_S3VM.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_compose_qns3vm = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_compose_qns3vm)
                                results_pkl = pd.read_pickle(results_compose_qns3vm)
                                print("Results:\n" , results_pkl )

                            elif i == 'compose' and j == 'label_propagation':
                                reg_compose_label_prop = compose.COMPOSE(classifier="label_propagation", mode= mode, method="compose", num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = reg_compose_label_prop.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")  
                                reg_compose_label_prop.avg_perf_metric['Experiment'] = i + '_' + j 
                                reg_compose_label_prop.avg_perf_metric['Time_Stamp'] = time_stamp  
                                results_df = pd.DataFrame.from_dict((reg_compose_label_prop.avg_perf_metric.keys(), reg_compose_label_prop.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_compose_lbl_prop = 'results_' + f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_compose_lbl_prop)
                                results_pkl = pd.read_pickle(results_compose_lbl_prop)
                                print("Results:\n" , results_pkl )

                            elif i == 'compose' and j == 'svm':
                                reg_compose_svm = compose.COMPOSE(classifier="svm", mode= mode ,method="compose", num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = reg_compose_svm.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                reg_compose_svm.avg_perf_metric['Experiment'] = i + '_' + j 
                                reg_compose_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((reg_compose_svm.avg_perf_metric.keys(), reg_compose_svm.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_compose_svm = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_compose_svm)
                                results_pkl = pd.read_pickle(results_compose_svm)
                                print("Results:\n" , results_pkl )

                            elif i == 'compose' and j == 'naive_bayes':
                                reg_compose_nb = compose.COMPOSE(classifier="naive_bayes", mode= mode ,method="compose", num_cores= self.num_cores, selected_dataset = dataset, datasource= datasource)
                                self.results[experiment] = reg_compose_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                reg_compose_nb.avg_perf_metric['Experiment'] = i + '_' + j 
                                reg_compose_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((reg_compose_nb.avg_perf_metric.keys(), reg_compose_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_compose_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_compose_nb)
                                results_pkl = pd.read_pickle(results_compose_nb)
                                print("Results:\n" , results_pkl )
                            
                            elif i == 'scargc' and j == '1nn': 
                                # scargc_svm_data = SetData(dataset= 'UG_2C_2D')
                                # run_scargc_svm = SCARGC(Xinit= scargc_svm_data.X[0], Yinit= scargc_svm_data.Y , classifier = 'svm', dataset= 'UG_2C_2D')
                                # results = run_scargc_svm.run(Xts = scargc_svm_data.X, Yts = scargc_svm_data.Y)
                                scargc_1nn = scargc.SCARGC(classifier = '1nn', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_1nn.run(Xts = scargc_1nn.X, Yts = scargc_1nn.Y)
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_1nn.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_1nn.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_1nn.avg_perf_metric.keys(), scargc_1nn.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_1nn = 'results_'+ f'{experiment}' + '.pkl'
                                results_df.to_pickle(results_scargc_1nn)
                                results_pkl = pd.read_pickle(results_scargc_1nn)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'svm': 
                                scargc_svm = scargc.SCARGC(classifier = 'svm', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_svm.run(Xts = scargc_svm.X, Yts = scargc_svm.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_svm.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_svm.avg_perf_metric.keys(), scargc_svm.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_svm = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_svm)
                                results_pkl = pd.read_pickle(results_scargc_svm)
                                print("Results:\n", results_df)
                            elif i == 'scargc' and j == 'knn': 
                                scargc_knn = scargc.SCARGC(classifier = 'knn', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_knn.run(Xts = scargc_knn.X, Yts = scargc_knn.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_knn.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_knn.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_knn.avg_perf_metric.keys(), scargc_knn.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_knn = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_knn)
                                results_pkl = pd.read_pickle(results_scargc_knn)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'logistic_regression': 
                                scargc_lr = scargc.SCARGC(classifier = 'logistic_regression', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_lr.run(Xts = scargc_lr.X, Yts = scargc_lr.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_lr.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_lr.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_lr.avg_perf_metric.keys(), scargc_lr.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_lr = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_lr)
                                results_pkl = pd.read_pickle(results_scargc_lr)
                                print("Results:\n", results_df)
                            
                            elif i == 'scargc' and j == 'random_forest': 
                                scargc_rf = scargc.SCARGC(classifier = 'random_forest', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_rf.run(Xts = scargc_rf.X, Yts = scargc_rf.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_rf.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_rf.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_rf.avg_perf_metric.keys(), scargc_rf.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_rf = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_rf)
                                results_pkl = pd.read_pickle(results_scargc_rf)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'adaboost': 
                                scargc_ab = scargc.SCARGC(classifier = 'adaboost', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_ab.run(Xts = scargc_ab.X, Yts = scargc_ab.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_ab.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_ab.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_ab.avg_perf_metric.keys(), scargc_ab.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_ab = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_ab)
                                results_pkl = pd.read_pickle(results_scargc_ab)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'decision_tree': 
                                scargc_ab = scargc.SCARGC(classifier = 'decision_tree', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_ab.run(Xts = scargc_ab.X, Yts = scargc_ab.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_ab.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_ab.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_ab.avg_perf_metric.keys(), scargc_ab.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_ab = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_ab)
                                results_pkl = pd.read_pickle(results_scargc_ab)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'mlp': 
                                scargc_ab = scargc.SCARGC(classifier = 'mlp', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_ab.run(Xts = scargc_ab.X, Yts = scargc_ab.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_ab.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_ab.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_ab.avg_perf_metric.keys(), scargc_ab.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_ab = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_ab)
                                results_pkl = pd.read_pickle(results_scargc_ab)
                                print("Results:\n", results_df)

                            elif i == 'scargc' and j == 'naive_bayes': 
                                scargc_ab = scargc.SCARGC(classifier = 'naive_bayes', dataset= dataset, datasource= datasource)
                                self.results[experiment] = scargc_ab.run(Xts = scargc_ab.X, Yts = scargc_ab.Y)     # .X[0] only get initial training set
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                scargc_ab.avg_perf_metric['Experiment'] = i + '_' + j
                                scargc_ab.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((scargc_ab.avg_perf_metric.keys(), scargc_ab.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_scargc_ab = 'results_'+ f'{experiment}' + '.pkl' 
                                results_df.to_pickle(results_scargc_ab)
                                results_pkl = pd.read_pickle(results_scargc_ab)
                                print("Results:\n", results_df)

                            elif i == 'mclassification' and j == 'knn':
                                mclass_knn = mclass.MClassification(classifier= 'knn', dataset= dataset, method = 'kmeans', datasource= datasource)
                                self.results[experiment] = mclass_knn.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                mclass_knn.avg_perf_metric['Experiment'] = i + '_' + j
                                mclass_knn.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((mclass_knn.avg_perf_metric.keys(), mclass_knn.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_mclass_knn = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_mclass_knn)
                                results_pkl = pd.read_pickle(results_mclass_knn)
                                print("Results:\n" , results_pkl )

                            elif i == 'mclassification' and j == 'svm':
                                mclass_svm = mclass.MClassification(classifier= 'svm', dataset= dataset, method = 'kmeans', datasource= datasource)
                                self.results[experiment] = mclass_svm.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                mclass_knn.avg_perf_metric['Experiment'] = i + '_' + j
                                mclass_svm.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((mclass_svm.avg_perf_metric.keys(), mclass_svm.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_mclass_svm = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_mclass_svm)
                                results_pkl = pd.read_pickle(results_mclass_svm)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'svm':
                                van = vanilla.VanillaClassifier(classifier= 'svm', dataset=dataset)
                                self.results[experiment] = van.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van.avg_perf_metric['Experiment'] = i + '_' + j
                                van.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van.avg_perf_metric.keys(), van.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_svm = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_svm)
                                results_pkl = pd.read_pickle(results_van_svm)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'naive_bayes_stream':
                                van_nb_stream = vanilla.VanillaClassifier(classifier= 'naive_bayes_stream', dataset=dataset)
                                self.results[experiment] = van_nb_stream.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb_stream.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb_stream.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb_stream.avg_perf_metric.keys(), van_nb_stream.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nbs = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nbs)
                                results_pkl = pd.read_pickle(results_van_nbs)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'naive_bayes':
                                van_nb = vanilla.VanillaClassifier(classifier= 'naive_bayes', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )
                            
                            elif i == 'vanilla' and j == 'logistic_regression':
                                van_nb = vanilla.VanillaClassifier(classifier= 'logistic_regression', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'random_forest':
                                van_nb = vanilla.VanillaClassifier(classifier= 'random_forest', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )
                            
                            elif i == 'vanilla' and j == 'adaboost':
                                van_nb = vanilla.VanillaClassifier(classifier= 'adaboost', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )
                            
                            elif i == 'vanilla' and j == 'decision_tree':
                                van_nb = vanilla.VanillaClassifier(classifier= 'decision_tree', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'mlp':
                                van_nb = vanilla.VanillaClassifier(classifier= 'mlp', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == '1nn':
                                van_nb = vanilla.VanillaClassifier(classifier= '1nn', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )

                            elif i == 'vanilla' and j == 'knn':
                                van_nb = vanilla.VanillaClassifier(classifier= 'knn', dataset=dataset)
                                self.results[experiment] = van_nb.run()
                                time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
                                van_nb.avg_perf_metric['Experiment'] = i + '_' + j
                                van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
                                results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
                                # change the directory to your particular files location
                                self.change_dir()
                                results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
                                results_df.to_pickle(results_van_nb)
                                results_pkl = pd.read_pickle(results_van_nb)
                                print("Results:\n" , results_pkl )
        self.plot_results()

# run scargc 
# run_scargc = RunExperiment(experiements=['scargc'], classifier=['1nn'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['svm'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['knn'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['logistic_regression'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['random_forest'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['adaboost'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['decision_tree'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

run_scargc = RunExperiment(experiements=['scargc'], classifier=['mlp'], modes=[''],
                    datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
                    datasources= ['unsw'])
run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['naive_bayes'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc'], classifier=['1nn'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['vanilla'], classifier=['svm', 'knn', 'logistic_regression', 'random_forest', 
#                                                                 'adaboost', 'decision_tree', 'mlp', 'naive_bayes','1nn'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather'], 
#                     datasources= ['unsw'])
# run_scargc.run()

# run_scargc = RunExperiment(experiements=['scargc',], classifier=['knn'], modes=[''],
#                     datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', 'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather','bot_iot'], 
#                     datasources= ['unsw'])
# run_scargc.run()



#%%
