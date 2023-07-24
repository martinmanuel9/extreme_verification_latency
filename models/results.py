#%%
#!/usr/bin/env python 
"""
Application:        Results
File name:          results.py
Author:             Martin Manuel Lopez
Creation:           02/14/2023

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
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

class Results:
    def provide_results(self):
        path = str(Path.home())
        path = path + "/extreme_verification_latency/results"
        os.chdir(path)
        list_dir = os.listdir(path)
        result = {}
        for i in range(len(list_dir)):
            result[i] = pickle.load(open(list_dir[i], "rb"))
            # print(result[i], "\n")
            # loc[0] for dataset
            # loc[1] for classifier
            # loc[14] for Experiment
            # loc[4]  for avg accuracy
            # loc[10]  for total exec time
            # result[0].loc[12].at[1]

        dataset = {}
        experiment = {}
        accuracy = {}
        roc_auc = {}
        total_exec_time_sec = {}
        for i in range(len(result)):
            dataset[i] = result[i].loc[0].at[1]
            experiment[i] = result[i].loc[14].at[1]
            accuracy[i] = result[i].loc[4].at[1]
            roc_auc[i] = result[i].loc[7].at[1]
            total_exec_time_sec[i] = result[i].loc[10].at[1]

        data = {}
        data['dataset'] = dataset.values()
        data['experiment'] = experiment.values()
        data['accuracy'] = accuracy.values()
        data['roc_auc'] = roc_auc.values()
        data['total_exec_time'] = total_exec_time_sec.values()

        to_excel = pd.DataFrame(data) 
        to_excel.to_excel('Results.xlsx')

        fig = go.Figure(data=[go.Table(header= dict(values= ['Dataset', 'Experiment', 'Avg Accuracy', 'Avg ROC AUC','Total Exec Time in Seconds']),
                                        cells= dict(values= [list(dataset.values()), list(experiment.values()),
                                                            list(accuracy.values()), list(roc_auc.values()), list(total_exec_time_sec.values())]))
                            ])

        fig.show()

res = Results()
res.provide_results()
# %%
