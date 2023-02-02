#%%
#!/usr/bin/env python 

"""
Application:        Online Learning in Extreme Verification Latency
File name:          openpickel.py
Author:             Martin Manuel Lopez
Creation:           07/05/2022

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

import pickle as pickle 
import os
import matplotlib
import matplotlib.backend_bases
import matplotlib.backends
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

class OpenResults():
    def run(self):
        # change the directory to your particular files location
        path = str(Path.home())
        path = path + "/extreme_verification_latency/results"
        os.chdir(path)
        list_dir = os.listdir(path)
        for i in range(len(list_dir)):
            result = pickle.load(open(list_dir[i], "rb"))
            print(result, "\n")

        ## plot results 
        plot_path = str(Path.home())
        plot_path = plot_path + "/extreme_verification_latency/plots"
        os.chdir(plot_path)
        plot_dir = os.listdir(plot_path)
        for j in range(len(plot_dir)):
            plot_data = pickle.load(open(plot_dir[j], 'rb'))
            fig_handle = plt.figure()
            fig, ax = plt.subplots()
            experiments = plot_data.keys()
            result_plot = {}
            for experiment in experiments:
                result_plot['Timesteps'] = plot_data[experiment]['Timesteps']
                result_plot['Accuracy'] = plot_data[experiment]['Accuracy']
                df = pd.DataFrame(result_plot)
                df.plot(ax=ax, label=experiment, x='Timesteps', y='Accuracy')
            plt.title('Accuracy over Timesteps of ' + plot_dir[j] )
            plt.xlabel('Timesteps')
            plt.tight_layout()
            plt.legend
            plt.ylabel('% Accuracy')
            plt.gcf().set_size_inches(15,10)  
            plt.show()


open_results = OpenResults()
open_results.run()
# %%
