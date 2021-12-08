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

class Datagen:
    def __init__(self) -> None:
        # Unimodal, Multimodal, 1cdt, 2cdt, Unimodal3D,  1cht, 2cht, 4cr, 4crev1,4crev2
        # 5cvt, 1csurr, 4ce1cf, fg2c2d, gears2c2d, keystroke, Unimodal5D, noaa, custom
        # custom_mm, spirals, corners, cic, hk, nss_ext, nss, nss1, ol, diff_clus, 
        # mov_dat, cyc, vio_dcr, const_nvio_dcr, const_vio_dcr, gwac, vt, custom_vt
        self.datatype = ''
        self.data = []
        self.dataset = []


    def gen_dataset(self, datatype):
        # change the directory to your particular files location
        os.chdir('/Users/martinlopez/extreme_verification_latency_lopez/extreme_verification_latency/data/files/')
        self.datatype = datatype
        if self.datatype == 'Unimodal':
            # Unimodal option
            UG_2C_2D =  pd.read_csv('UG_2C_2D.txt', delimiter="," , names=['feat1', 'feat2', 'feat3'])                                                                                        
            step = 1000
            data = pd.DataFrame()
            df = pd.DataFrame(UG_2C_2D)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

    
        if self.datatype == 'Multimodal':           
            # multimodal option
            MG_2C_2D = pd.read_csv("MG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 2000
            data = pd.DataFrame()
            df = pd.DataFrame(MG_2C_2D) 
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '1CDT':           
            X1CDT = pd.read_csv("1CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X1CDT)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '2CDT':           
            X2CDT = pd.read_csv("2CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X2CDT)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)
        
        if self.datatype == 'Unimodal3D':           
            UG_2C_3D = pd.read_csv("UG_2C_3D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3', 'feat4'])                                                                                    
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(UG_2C_3D)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)      
        
        if self.datatype == '1cht':           
            X1CHT = pd.read_csv("1CHT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X1CHT)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)
        
        if self.datatype == '2cht':           
            X2CHT = pd.read_csv("4CR.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                      
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X2CHT)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)       

        if self.datatype == '4cr':           
            X4CR = pd.read_csv("4CR.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 400
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CR)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '4crev1':           
            X4CRE_V1 = pd.read_csv("4CRE-V1.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CRE_V1)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '4crev2':           
            X4CRE_V2 = pd.read_csv("4CRE-V2.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CRE_V2)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '5cvt':           
            X5CVT = pd.read_csv("5CVT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 1000
            data = pd.DataFrame() 
            df = pd.DataFrame(X5CVT)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '1csurr':           
            X1Csurr = pd.read_csv("1Csurr.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            num = 600 * math.floor(len(X1Csurr)/600)
            X1Csurr = X1Csurr[0:num]                                                                              
            step = 600
            data = pd.DataFrame() 
            df = pd.DataFrame(X1Csurr)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == '4ce1cf':           
            X4CE1CF = pd.read_csv("4CE1CF.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            drift_no = 750
            num = drift_no * math.floor(len(X4CE1CF)/drift_no)
            X4CE1CF = X4CE1CF[0:num]                                                                           
            data = pd.DataFrame() 
            df = pd.DataFrame(X4CE1CF)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == 'fg2c2d':           
            FG_2C_2D = pd.read_csv("FG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(FG_2C_2D)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == 'gears2c2d':           
            GEARS_2C_2D = pd.read_csv("GEARS_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(GEARS_2C_2D)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)


        if self.datatype == 'keystroke':           
            keystroke = pd.read_csv("keystroke.txt", delimiter=",", names=['feat1', 'feat2', 'feat3','feat4', 
                                    'feat5', 'feat6', 'feat7', 'feat8', 'feat9', 'feat10'])                                                                                    
            step = 200
            data = pd.DataFrame() 
            df = pd.DataFrame(keystroke)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)
    
        if self.datatype == 'Unimodal5D':           
            UG_2C_5D = pd.read_csv("UG_2C_5D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3',
                                    'feat4', 'feat5'])                                                                                      
            step = 2000
            data = pd.DataFrame() 
            df = pd.DataFrame(UG_2C_5D)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        if self.datatype == 'UnitTest':           
            unitTestData = pd.read_csv("unit_test.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])       
            step = 10                                                                           
            data = pd.DataFrame()
            df = pd.DataFrame(unitTestData)
            self.data = df
            num_col = np.shape(self.data)[1]
    
            if self.label_check() is True:
                for i in range(0, len(df), step):
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col]
                        test_train = np.array(data.T) 
                    self.dataset.append(test_train)  # appends to test_train.
            else:
                for i in range(0, len(df), step):                               
                    for j in range(0,step):
                        data[j] = df.iloc[j,:num_col-1]                             
                        test_train = data.T
                        test_train["label"] = 1 
                        arr_test_train = np.array(test_train)
                    self.dataset.append(arr_test_train)

        return self.dataset

    def label_check(self):
        # length_data = len(self.data.columns)
        # print(length_data)
        exists_label = 1 in self.data.values   # self.data.iloc[:,-1:]

        if exists_label is True:
            return True
        else:
            return False

if __name__ == '__main__':
    # testArray = ['Unimodal', 'Multimodal', '1CDT', '2CDT','Unimodal3D', '1cht', '2cht', '4cr', '4crev1','4crev2','5cvt','1csurr','4ce1cf',
    #             '4ce1cf','fg2c2d','gears2c2d','keystroke', 'Unimodal5D']
    # for i in testArray:
    #     test_data = Datagen.dataset(i)
    #     if test_data.empty:
    #         print(i + "is empty")
    #     else: 
    #         print(i + " dataset created")
    testData = Datagen()
    
    test = testData.gen_dataset('UnitTest')
    print(test)
    # if test.empty:
    #     print("Unit Test dataset is empty")
    # else:
    #     print("Unit Test dataset created")