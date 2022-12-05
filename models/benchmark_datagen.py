#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          benchmark_datagen.py
Author:             Martin Manuel Lopez
Creation:           08/05/2021

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
from pathlib import Path


class Datagen:
    def __init__(self) -> None:
        # Unimodal, Multimodal, 1cdt, 2cdt, Unimodal3D,  1cht, 2cht, 4cr, 4crev1,4crev2
        # 5cvt, 1csurr, 4ce1cf, fg2c2d, gears2c2d, keystroke, Unimodal5D, noaa, custom
        # custom_mm, spirals, corners, cic, hk, nss_ext, nss, nss1, ol, diff_clus, 
        # mov_dat, cyc, vio_dcr, const_nvio_dcr, const_vio_dcr, gwac, vt, custom_vt
        self.datatype = ''
        self.data = []
        self.dataset = []
        
    def change_directory(self):
        path = str(Path.home())
        path = path + '/extreme_verification_latency/data/synthetic_data/'
        os.chdir(path)

    def gen_dataset(self, datatype):
        self.datatype = datatype
        self.change_directory()
        if self.datatype == 'UG_2C_2D':
            # Unimodal option
            UG_2C_2D =  pd.read_csv('UG_2C_2D.txt', delimiter="," , names=['feat1', 'feat2', 'feat3'])                                                                                        
            step = 1000
            data = []
            self.data = UG_2C_2D 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 


    
        if self.datatype == 'MG_2C_2D':           
            # multimodal option
            MG_2C_2D = pd.read_csv("MG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 2000
            data = []
            self.data = MG_2C_2D 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 

        if self.datatype == '1CDT':           
            X1CDT = pd.read_csv("1CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 160
            data = []
            self.data = X1CDT 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 


        if self.datatype == '2CDT':           
            X2CDT = pd.read_csv("2CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 160
            data = []
            self.data = X2CDT 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 
        
        if self.datatype == 'UG_2C_3D':           
            UG_2C_3D = pd.read_csv("UG_2C_3D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3', 'feat4'])                                                                                    
            step = 2000
            data = []
            self.data = UG_2C_3D 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 
        
        if self.datatype == '1CHT':           
            X1CHT = pd.read_csv("1CHT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 160
            data = []
            self.data = X1CHT 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data) 
        
        if self.datatype == '2CHT':           
            X2CHT = pd.read_csv("2CHT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                      
            step = 160
            data = []
            self.data = X2CHT 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)  

        if self.datatype == '4CR':           
            X4CR = pd.read_csv("4CR.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 400
            data = []
            self.data = X4CR 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == '4CREV1':           
            X4CRE_V1 = pd.read_csv("4CRE-V1.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 1000
            data = []
            self.data = X4CRE_V1 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == '4CREV2':           
            X4CRE_V2 = pd.read_csv("4CRE-V2.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 1000
            data = []
            self.data = X4CRE_V2 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == '5CVT':           
            X5CVT = pd.read_csv("5CVT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 1000
            data = []
            self.data = X5CVT 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)    
       

        if self.datatype == '1CSURR':           
            X1Csurr = pd.read_csv("1Csurr.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            num = 600 * math.floor(len(X1Csurr)/600)
            X1Csurr = X1Csurr[0:num]                                                                              
            step = 600
            data = []
            self.data = X1Csurr 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == '4CE1CF':           
            X4CE1CF = pd.read_csv("4CE1CF.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            drift_no = 750
            num = drift_no * math.floor(len(X4CE1CF)/drift_no)
            X4CE1CF = X4CE1CF[0:num]                                                                           
            self.data = X4CE1CF 
            data = []
            if self.label_check() is True:
                for i in self.batch(self.data , num):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, num):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == 'FG_2C_2D':           
            FG_2C_2D = pd.read_csv("FG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            step = 2000
            data = []
            self.data = FG_2C_2D 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)

        if self.datatype == 'GEARS_2C_2D':           
            GEARS_2C_2D = pd.read_csv("GEARS_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            step = 2000
            data = []
            self.data = GEARS_2C_2D 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)


        if self.datatype == 'keystroke':           
            keystroke = pd.read_csv("keystroke.txt", delimiter=",", names=['feat1', 'feat2', 'feat3','feat4', 
                                    'feat5', 'feat6', 'feat7', 'feat8', 'feat9', 'feat10', 'feat11'])                                                                                    
            step = 200
            data = []
            self.data = keystroke 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat11', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)
    
        if self.datatype == 'UG_2C_5D':           
            UG_2C_5D = pd.read_csv("UG_2C_5D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3',
                                    'feat4', 'feat5', 'feat6'])                                                                                      
            step = 2000
            data = []
            self.data = UG_2C_5D
            if self.label_check() is True:
                for i in self.batch(self.data , n=step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat6', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, n=step):
                    data.append(i)
                self.dataset.append(data)       

        if self.datatype == 'UnitTest':           
            unitTestData = pd.read_csv("unit_test.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])     
            step = 10                                                                          
            data = []
            self.data = unitTestData 
            if self.label_check() is True:
                for i in self.batch(self.data , step):
                    data.append(i) 
                self.dataset.append(data)
            else:
                self.data.drop('feat3', axis=1, inplace=True)
                self.data['label'] = 1 
                for i in self.batch(self.data, step):
                    data.append(i)
                self.dataset.append(data)



        return self.dataset
    
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield np.array(iterable[ndx:min(ndx + n, l)])

    def label_check(self):
        # length_data = len(self.data.columns)
        # print(length_data)
        exists_label = 1 in self.data.values   # self.data.iloc[:,-1:]

        if exists_label is True:
            return True
        else:
            return False

# if __name__ == '__main__':
#     testData = Datagen()
#     testArray = ['Unimodal', 'Multimodal', '1CDT', '2CDT','Unimodal3D', '1cht', '2cht', '4cr', '4crev1','4crev2','5cvt','1csurr','4ce1cf',
#                 '4ce1cf','fg2c2d','gears2c2d','keystroke', 'Unimodal5D', 'UnitTest']
#     for i in testArray:
#         test_dataset = testData.gen_dataset(i)
#         if len(test_dataset) == 0:
#             print(i, "is empty")
#         else: 
#             print(i, " dataset created with size " ,  np.shape(test_dataset))
    
#     # testData = Datagen()
#     test = testData.gen_dataset('Unimodal5D')
#     print(np.shape(test))
#     # if test.empty:
#     #     print("Unit Test dataset is empty")
#     # else:
#     #     print("Unit Test dataset created")