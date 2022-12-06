#!/usr/bin/env python 

"""
Application:        Cyber Attacks Data Generation from USNW - NB15 dataset 
File name:          
Author:             Martin Manuel Lopez
Creation:           12/5/2022

The University of Arizona
Department of Electrical and Computer Engineering
College of Engineering
"""

# MIT License
#
# Copyright (c) 2021
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

import random
from typing import Match
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path


class UNSW_NB15_Datagen:   
    def __init__(self) -> None:
        self.data_gen()

    def change_directory(self):
        path = str(Path.home())
        path = path + '/extreme_verification_latency/data/UNSW_NB15/'
        os.chdir(path)

    def data_gen(self):
        self.change_directory()
        self.trainSet = pd.read_csv('UNSW_NB15_training-set.csv')
        self.testSet = pd.read_csv('UNSW_NB15_testing-set.csv')
        self.flow_features()
        self.basic_features()
        self.content_features()
        self.time_features()
        self.generated_features()
        
    def flow_features(self): 
        flow_features = ['proto', 'label'] # ['scrip', 'sport','dstip','dsport'] were not found in the csv file
        self.flowFeatTrain = self.trainSet[flow_features]
        self.flowFeatTest = self.testSet[flow_features]
        

    def basic_features(self):
        basic_features = ['state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss','dloss','service','sload', 'dload', 'spkts', 'dpkts', 'label']
        self.basicFeatTrain = self.trainSet[basic_features]
        self.basicFeatTest = self.testSet[basic_features]

    def content_features(self):
        contet_features = ['swin','dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth', 'label'] # ['res_bdy_len] was not found in csv
        self.contentFeatTrain = self.trainSet[contet_features]
        self.contentFeatTest = self.testSet[contet_features]

    def time_features(self):
        time_features =['sjit', 'djit',  'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'label']   # ['stime', 'ltime'] not found in csv
        self.timeFeatTrain = self.trainSet[time_features]
        self.timeFeatTest = self.testSet[time_features]

    def generated_features(self):
        generated_features = ['is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
                                'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'label' ]
        self.generateFeatTrain = self.trainSet[generated_features]
        self.generateFeatTest = self.testSet[generated_features]
        


