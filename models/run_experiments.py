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

import compose
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import math
import seaborn as sns 

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
        colors = sns.color_pallette('husl', n_colors=len(experiments))
        cmap = dict(zip(experiments, colors))

        col_nums = 3 
        row_nums = math.ceil(len(experiments) / col_nums)
        plt.figure(fig_size = (10,4))
        for i, (k,v) in enumerate(self.results.items(), 1):
            plt.subplot(row_nums, col_nums, i)
            p = sns.scatterplot(data=v , x='Timesteps' , y='Accuracy', pallete=cmap)
            plt.title(f'Experiement: {k}')
        
        plt.tight_layout()

        patches = [Patch(color=v, label=k) for k, v in cmap.items()]
        
        plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)
        plt.show()


    def run(self):
        
        for i in self.experiments:
            for j in self.classifier:
                for dataset in self.datasets:
                    experiment = dataset + '_' + j + '_' + i
                    if i == 'fast_compose' and j == 'QN_S3VM':
                        fast_compose_QNS3VM = compose.COMPOSE(classifier="QN_S3VM", method="gmm", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = fast_compose_QNS3VM.run()
                        results_df = pd.DataFrame.from_dict((fast_compose_QNS3VM.avg_results_dict.keys(), fast_compose_QNS3VM.avg_results_dict.values())).T
                        print(results_df)
                    elif i == 'fast_compose' and j == 'label_propagation':
                        fast_compose_label_prop = compose.COMPOSE(classifier="label_propagation", method="gmm", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = fast_compose_label_prop.run()
                        results_df = pd.DataFrame.from_dict((fast_compose_label_prop.avg_results_dict.keys(), fast_compose_label_prop.avg_results_dict.values())).T
                        print(results_df)
                    elif i == 'compose' and j == 'QN_S3VM':
                        reg_compose_label_prop = compose.COMPOSE(classifier="QN_S3VM", method="a_shape", verbose = self.verbose, num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = reg_compose_label_prop.run()
                        results_df = pd.DataFrame.from_dict((reg_compose_label_prop.avg_results_dict.keys(), reg_compose_label_prop.avg_results_dict.values())).T
                        print(results_df)
                    elif i == 'compose' and j == 'label_propagation':
                        reg_compose_QNS3VM = compose.COMPOSE(classifier="label_propagation", method="a_shape", verbose = self.verbose ,num_cores= self.num_cores, selected_dataset = dataset)
                        self.results[experiment] = reg_compose_QNS3VM.run()
                        results_df = pd.DataFrame.from_dict((reg_compose_QNS3VM.avg_results_dict.keys(), reg_compose_QNS3VM.avg_results_dict.values())).T
                        print(results_df)
        
        self.plot_results()

run_experiment = RunExperiment(experiements=['fast_compose','compose'], classifier=['label_propagation', 'QN_S3VM'], 
                                            verbose=0, datasets=[ 'UG_2C_2D' ,'2CDT', 'MG_2C_2D','1CDT'], num_cores=0.8)
run_experiment.run()
