#%%
#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
File name:          run_experiment.py
Author:             Martin Manuel Lopez
Creation:           07/13/2023

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


from matplotlib import pyplot as plt
import pandas as pd
import pickle as pickle 
import time
import scargc
import mclassification
import vanilla
import os
import time
from pathlib import Path 

class RunExperiment:
    def __init__(self, experiements = [], classifiers = [], datasets=[], datasources = [], methods=[]):
            self.experiments = experiements
            self.classifier = classifiers
            self.datasets = datasets
            self.results = {}
            self.datasources = datasources 
            self.methods = methods
            
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

    def createExperiment(self, experiment, classifier, datasource, dataset, method):

        if experiment == 'vanilla':
            experiment = experiment + '_' + dataset + '_' + classifier + '_' + datasource
            van_nb = vanilla.VanillaClassifier(classifier= classifier, dataset= dataset)
            self.results[experiment] = van_nb.run()
            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
            van_nb.avg_perf_metric['Experiment'] = experiment 
            van_nb.avg_perf_metric['Time_Stamp'] = time_stamp
            results_df = pd.DataFrame.from_dict((van_nb.avg_perf_metric.keys(), van_nb.avg_perf_metric.values())).T
            # change the directory to your particular files location
            self.change_dir()
            results_van_nb = 'results_'+ f'{experiment}' +'.pkl'
            results_df.to_pickle(results_van_nb)
            results_pkl = pd.read_pickle(results_van_nb)
            print("Results:\n" , results_pkl )
        elif experiment == 'scargc':
            experiment = experiment + '_' + dataset + '_' + classifier + '_' + datasource
            scargc_ab = scargc.SCARGC(classifier = classifier, dataset= dataset, datasource= datasource)
            self.results[experiment] = scargc_ab.run()
            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
            scargc_ab.avg_perf_metric['Experiment'] = experiment + '_' + classifier
            scargc_ab.avg_perf_metric['Time_Stamp'] = time_stamp
            results_df = pd.DataFrame.from_dict((scargc_ab.avg_perf_metric.keys(), scargc_ab.avg_perf_metric.values())).T
            # change the directory to your particular files location
            self.change_dir()
            results_scargc_ab = 'results_'+ f'{experiment}' + '.pkl' 
            results_df.to_pickle(results_scargc_ab)
            results_pkl = pd.read_pickle(results_scargc_ab)
            print("Results:\n", results_df)
        elif experiment == 'mclass':
            experiment = experiment + '_' + dataset + '_' + classifier + '_' + datasource
            mclass = mclassification.MClassification(classifier= classifier, method= method, dataset= dataset, datasource= datasource, graph= True) 
            print(mclass)
            self.results[experiment] = mclass
            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
            mclass.avg_perf_metric['Experiment'] = experiment + '_' + classifier
            mclass.avg_perf_metric['Time_Stamp'] = time_stamp
            results_df = pd.DataFrame.from_dict((mclass.avg_perf_metric.keys(), mclass.avg_perf_metric.values())).T
            # change the directory to your particular files location
            self.change_dir()
            results_mclass = 'results_'+ f'{experiment}' + '.pkl' 
            results_df.to_pickle(results_mclass)
            results_pkl = pd.read_pickle(results_mclass)
            print("Results:\n", results_df)

    def run(self):
        
        for experiment in self.experiments:
            for classifier in self.classifier:
                for datasource in self.datasources:
                    for dataset in self.datasets:
                        for method in self.methods:
                            self.createExperiment(experiment= experiment, classifier= classifier, datasource= datasource, dataset= dataset, method= method)
        
        self.plot_results()



run = RunExperiment(experiements=['scargc'], classifiers=['mlp'], methods=['kmeans'],
                    datasets= ['ton_iot_fridge','ton_iot_garage' ,'ton_iot_gps','ton_iot_modbus', \
                               'ton_iot_light', 'ton_iot_thermo', 'ton_iot_weather','bot_iot'], 
                    datasources= ['UNSW'])
run.run()