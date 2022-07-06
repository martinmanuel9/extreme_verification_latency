#!/usr/bin/env python 

"""
Application:        Online Learning Extreme Verification Latency 
File name:          openpickel.py
Author:             Martin Manuel Lopez
Creation:           07/05/2022

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

import pickle5 as pickel 
import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

# change the directory to your particular files location
path = str(Path.home())
path = path + "/extreme_verification_latency/results"
os.chdir(path)

list_dir = os.listdir(path)

for i in range(len(list_dir)):
    result = pickel.load(open(list_dir[i], "rb"))
    print(result, "\n")

plot_path = str(Path.home())
plot_path = plot_path + "/extreme_verification_latency/plots"
os.chdir(plot_path)

plot_dir = os.listdir(plot_path)

for j in range(len(plot_dir)):
    plotter = pickel.load(open(plot_dir[j], "rb"))
    plotter.show()