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
import multiprocessing

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
            scargc_ab.avg_perf_metric['Experiment'] = experiment 
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
            mclass = mclassification.MClassification(classifier= classifier, method= method, dataset= dataset, datasource= datasource, graph= False) 
            self.results[experiment] = mclass.run()
            time_stamp = time.strftime("%Y%m%d-%H:%M:%S")
            mclass.avg_perf_metric['Experiment'] = experiment 
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


# lstm 
scargc_dnn_lstm_fridge = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_fridge'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_garage = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_garage'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_gps = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_gps'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_modbus = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_modbus'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_light = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_light'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_thermo = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_thermo'],
                    datasources= ['UNSW'])


scargc_dnn_lstm_weather = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['ton_iot_weather'],
                    datasources= ['UNSW'])

scargc_dnn_lstm_bot = RunExperiment(experiements=['scargc'], classifiers=['lstm'],
                    datasets= ['bot_iot'],
                    datasources= ['UNSW'])

# gru 
scargc_dnn_gru_fridge = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_fridge'],
                    datasources= ['UNSW'])

scargc_dnn_gru_garage = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_garage'],
                    datasources= ['UNSW'])

scargc_dnn_gru_gps = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_gps'],
                    datasources= ['UNSW'])

scargc_dnn_gru_modbus = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_modbus'],
                    datasources= ['UNSW'])

scargc_dnn_gru_light = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_light'],
                    datasources= ['UNSW'])

scargc_dnn_gru_thermo = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_thermo'],
                    datasources= ['UNSW'])      

scargc_dnn_gru_weather = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['ton_iot_weather'],
                    datasources= ['UNSW'])

scargc_dnn_gru_bot = RunExperiment(experiements=['scargc'], classifiers=['gru'],
                    datasets= ['bot_iot'],
                    datasources= ['UNSW'])

# lstm
mclass_dnn_lstm_fridge = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_fridge'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_garage = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_garage'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_gps = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_gps'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_modbus = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_modbus'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_light = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_light'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_thermo = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_thermo'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_weather = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['ton_iot_weather'],
                    datasources= ['UNSW'])

mclass_dnn_lstm_bot = RunExperiment(experiements=['mclass'], classifiers=['lstm'],
                    datasets= ['bot_iot'],
                    datasources= ['UNSW'])

# gru
mclass_dnn_gru_fridge = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_fridge'],
                    datasources= ['UNSW'])

mclass_dnn_gru_garage = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_garage'],
                    datasources= ['UNSW'])

mclass_dnn_gru_gps = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_gps'],
                    datasources= ['UNSW'])

mclass_dnn_gru_modbus = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_modbus'],
                    datasources= ['UNSW'])

mclass_dnn_gru_light = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_light'],
                    datasources= ['UNSW'])

mclass_dnn_gru_thermo = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_thermo'],
                    datasources= ['UNSW'])

mclass_dnn_gru_weather = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['ton_iot_weather'],
                    datasources= ['UNSW'])

mclass_dnn_gru_bot = RunExperiment(experiements=['mclass'], classifiers=['gru'],
                    datasets= ['bot_iot'],
                    datasources= ['UNSW'])

if __name__ == '__main__':
    # Create Process objects for each script
    process1 = multiprocessing.Process(target=scargc_dnn_lstm_fridge.run())
    process2 = multiprocessing.Process(target=scargc_dnn_lstm_garage.run())
    process3 = multiprocessing.Process(target=scargc_dnn_lstm_gps.run())
    process4 = multiprocessing.Process(target=scargc_dnn_lstm_modbus.run())
    process5 = multiprocessing.Process(target=scargc_dnn_lstm_light.run())
    process6 = multiprocessing.Process(target=scargc_dnn_lstm_thermo.run())
    process7 = multiprocessing.Process(target=scargc_dnn_lstm_weather.run())
    process8 = multiprocessing.Process(target=scargc_dnn_lstm_bot.run())
    process9 = multiprocessing.Process(target=scargc_dnn_gru_fridge.run())
    process10 = multiprocessing.Process(target=scargc_dnn_gru_garage.run())
    process11 = multiprocessing.Process(target=scargc_dnn_gru_gps.run())
    process12 = multiprocessing.Process(target=scargc_dnn_gru_modbus.run())
    process13 = multiprocessing.Process(target=scargc_dnn_gru_light.run())
    process14 = multiprocessing.Process(target=scargc_dnn_gru_thermo.run())
    process15 = multiprocessing.Process(target=scargc_dnn_gru_weather.run())
    process16 = multiprocessing.Process(target=scargc_dnn_gru_bot.run())
    process17 = multiprocessing.Process(target=mclass_dnn_lstm_fridge.run())
    process18 = multiprocessing.Process(target=mclass_dnn_lstm_garage.run())
    process19 = multiprocessing.Process(target=mclass_dnn_lstm_gps.run())
    process20 = multiprocessing.Process(target=mclass_dnn_lstm_modbus.run())
    process21 = multiprocessing.Process(target=mclass_dnn_lstm_light.run())
    process22 = multiprocessing.Process(target=mclass_dnn_lstm_thermo.run())
    process23 = multiprocessing.Process(target=mclass_dnn_lstm_weather.run())
    process24 = multiprocessing.Process(target=mclass_dnn_lstm_bot.run())
    process25 = multiprocessing.Process(target=mclass_dnn_gru_fridge.run())
    process26 = multiprocessing.Process(target=mclass_dnn_gru_garage.run())
    process27 = multiprocessing.Process(target=mclass_dnn_gru_gps.run())
    process28 = multiprocessing.Process(target=mclass_dnn_gru_modbus.run())
    process29 = multiprocessing.Process(target=mclass_dnn_gru_light.run())
    process30 = multiprocessing.Process(target=mclass_dnn_gru_thermo.run())
    process31 = multiprocessing.Process(target=mclass_dnn_gru_weather.run())
    process32 = multiprocessing.Process(target=mclass_dnn_gru_bot.run())



    # Start the processes
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()
    process7.start()
    process8.start()
    process9.start()
    process10.start()
    process11.start()
    process12.start()
    process13.start()
    process14.start()
    process15.start()
    process16.start()
    process17.start()
    process18.start()
    process19.start()
    process20.start()
    process21.start()
    process22.start()
    process23.start()
    process24.start()
    process25.start()
    process26.start()
    process27.start()
    process28.start()
    process29.start()
    process30.start()
    process31.start()
    process32.start()


    # Wait for the processes to finish
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
    process7.join()
    process8.join()
    process9.join()
    process10.join()
    process11.join()
    process12.join()
    process13.join()
    process14.join()
    process15.join()
    process16.join()
    process17.join()
    process18.join()
    process19.join()
    process20.join()
    process21.join()
    process22.join()
    process23.join()
    process24.join()
    process25.join()
    process26.join()
    process27.join()
    process28.join()
    process29.join()
    process30.join()
    process31.join()
    process32.join()
    