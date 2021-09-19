#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          benchmark_datagen.py
Author:             Martin Manuel Lopez
Advisor:            Dr. Gregory Ditzler
Creation:           08/05/2021
COMPOSE Origin:     Muhammad Umer and Robi Polikar

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

from typing import Match
import pandas as pd
import numpy as np
import os
import math

class datagen():
    def __init__(self, data, dataset):
        _data = []                              # Unimodal, Multimodal, 1cdt, 2cdt, Unimodal3D,  1cht, 2cht, 4cr, 4crev1,4crev2
                                                # 5cvt, 1csurr, 4ce1cf, fg2c2d, gears2c2d, keystroke, Unimodal5D, noaa, custom
                                                # custom_mm, spirals, corners, cic, hk, nss_ext, nss, nss1, ol, diff_clus, 
                                                # mov_dat, cyc, vio_dcr, const_nvio_dcr, const_vio_dcr, gwac, vt, custom_vt
        _dataset=[]

        self._data = data
        self._dataset = dataset

    def dataset(datatype):
        # change the directory to your particular files location
        os.chdir('/Users/martinlopez/extreme_verification_latency_lopez/extreme_verification_latency/data/files/')

        if datatype == 'Unimodal':
            # Unimodal option
            UG_2C_2D =  pd.read_csv('UG_2C_2D.txt', delimiter="," , names=['column1', 'column2', 'column3'])                              
            l = 0                                                             
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(UG_2C_2D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(1000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1
    
        if datatype == 'Multimodal':           
            # multimodal option
            MG_2C_2D = pd.read_csv("MG_2C_2D.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(MG_2C_2D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(2000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1

        if datatype == '1CDT':           
            X1CDT = pd.read_csv("1CDT.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X1CDT)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(400)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1

        if datatype == '2CDT':           
            X2CDT = pd.read_csv("2CDT.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X2CDT)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(400)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1
        
        if datatype == 'Unimodal3D':           
            UG_2C_3D = pd.read_csv("UG_2C_3D.txt", delimiter=",", names=['column1', 'column2', 'column3', 'column4'])                         
            l = 0                                                             
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(UG_2C_3D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:3]                             
                    zero[l] = np.zeros_like(df.iloc[j,:3])
                    a = np.random.permutation(2000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1        
        
        if datatype == '1cht':           
            X1CHT = pd.read_csv("1CHT.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X1CHT)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(400)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1
        
        if datatype == '2cht':           
            X2CHT = pd.read_csv("4CR.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X2CHT)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(400)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1       

        if datatype == '4cr':           
            X4CR = pd.read_csv("4CR.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CR)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(400)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1   

        if datatype == '4crev1':           
            X4CRE_V1 = pd.read_csv("4CRE-V1.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CRE_V1)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(1000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 

        if datatype == '4crev2':           
            X4CRE_V2 = pd.read_csv("4CRE-V2.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CRE_V2)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(1000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 

        if datatype == '5cvt':           
            X5CVT = pd.read_csv("5CVT.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X5CVT)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(1000)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1

        if datatype == '1csurr':           
            X1Csurr = pd.read_csv("1Csurr.txt", delimiter=",", names=['column1', 'column2', 'column3'])
            num = 600 * math.floor(len(X1Csurr)/600)
            X1Csurr = X1Csurr[0:num]                 
            l = 0                                                             
            step = 600
            data = pd.DataFrame() 
            df = pd.DataFrame(X1Csurr)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(600)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 

        if datatype == '4ce1cf':           
            X4CE1CF = pd.read_csv("4CE1CF.txt", delimiter=",", names=['column1', 'column2', 'column3'])
            drift_no = 750
            num = drift_no * math.floor(len(X4CE1CF)/drift_no)
            X4CE1CF = X4CE1CF[0:num]                 
            l = 0                                                             
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CE1CF)
            zero = pd.DataFrame()
            for i in range(0, len(df), drift_no):                           
                for j in range(0, drift_no):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(drift_no)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 

        if datatype == 'fg2c2d':           
            FG_2C_2D = pd.read_csv("FG_2C_2D.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(FG_2C_2D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(step)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1  

        if datatype == 'gears2c2d':           
            GEARS_2C_2D = pd.read_csv("GEARS_2C_2D.txt", delimiter=",", names=['column1', 'column2', 'column3'])                         
            l = 0                                                             
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(GEARS_2C_2D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:2]                             
                    zero[l] = np.zeros_like(df.iloc[j,:2])
                    a = np.random.permutation(step)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 


        if datatype == 'keystroke':           
            keystroke = pd.read_csv("keystroke.txt", delimiter=",", names=['column1', 'column2', 'column3','column4', 
                                    'column5', 'column6', 'column7', 'column8', 'column9', 'column10'])                         
            l = 0                                                             
            step = 200
            data = pd.DataFrame() 
            df = pd.DataFrame(keystroke)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:10]                             
                    zero[l] = np.zeros_like(df.iloc[j,:10])
                    a = np.random.permutation(step)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1 
    
        if datatype == 'Unimodal5D':           
            UG_2C_5D = pd.read_csv("UG_2C_5D.txt", delimiter=",", names=['column1', 'column2', 'column3',
                                    'column4', 'column5'])                         
            l = 0                                                             
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(UG_2C_5D)
            zero = pd.DataFrame()
            for i in range(0, len(df), step):                           
                for j in range(0, step):                                
                    data[l] = df.iloc[j,:5]                             
                    zero[l] = np.zeros_like(df.iloc[j,:5])
                    a = np.random.permutation(step)
                    aT = a.T
                    if l == 0:
                        data[l] = aT[j]
                    dataset = data.T
                    dataset["label"] = 1
                l += 1

        return dataset

# if __name__ == '__main__':
#     # 'Unimodal', 'Multimodal', '1CDT', '2CDT','Unimodal3D', '1cht', '2cht', '4cr', '4crev1','4crev2','5cvt','1csurr',
#     testArray = ['Unimodal', 'Multimodal', '1CDT', '2CDT','Unimodal3D', '1cht', '2cht', '4cr', '4crev1','4crev2','5cvt','1csurr','4ce1cf',
#                 '4ce1cf','fg2c2d','gears2c2d','keystroke', 'Unimodal5D']
#     for i in testArray:
#         test_data = datagen.dataset(i)
#         if test_data.empty:
#             print(i + "is empty")
#         else: 
#             print(i + " dataset created")