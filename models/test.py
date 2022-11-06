#%%
import pickle as pickle 
import os
import matplotlib
import matplotlib.backend_bases
import matplotlib.backends
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import time

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
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    plt.title('Accuracy over Timesteps of ' + plot_dir[j] )
    plt.xlabel('Timesteps')
    plt.tight_layout()
    plt.legend
    plt.ylabel('% Accuracy')
    plt.gcf().set_size_inches(15,10)  
    plt.show()

# %%
