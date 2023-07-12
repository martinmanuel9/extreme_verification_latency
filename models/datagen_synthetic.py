#!/usr/bin/env python 

"""
Application:        Synthetic Data Generation for Verification Latency 
File name:          
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

import random
from typing import Match
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path


class Synthetic_Datagen:
    def __init__(self) -> None:
        # Unimodal, Multimodal, 1cdt, 2cdt, Unimodal3D,  1cht, 2cht, 4cr, 4crev1,4crev2
        # 5cvt, 1csurr, 4ce1cf, fg2c2d, gears2c2d, keystroke, Unimodal5D, noaa, custom
        # custom_mm, spirals, corners, cic, hk, nss_ext, nss, nss1, ol, diff_clus, 
        # mov_dat, cyc, vio_dcr, const_nvio_dcr, const_vio_dcr, gwac, vt, custom_vt
        self.datatype = ''
        self.data = []
        self.labels = []
        self.use = []
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
            UG_2C_2D = UG_2C_2D.to_numpy()
            self.dataset = UG_2C_2D
            step = 1000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = UG_2C_2D
            self.labels = UG_2C_2D[:,-1]
            self.use = UG_2C_2D[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

    
        if self.datatype == 'MG_2C_2D':           
            # multimodal option
            MG_2C_2D = pd.read_csv("MG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            MG_2C_2D = MG_2C_2D.to_numpy()
            self.dataset = MG_2C_2D                                                                                   
            step = 2000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = MG_2C_2D
            self.labels = MG_2C_2D[:,-1]
            self.use = MG_2C_2D[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '1CDT':           
            X1CDT = pd.read_csv("1CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])    
            X1CDT = X1CDT.to_numpy()    
            self.dataset = X1CDT                                                                             
            step = 160
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X1CDT
            self.labels = X1CDT[:,-1]
            self.use = X1CDT[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '2CDT':           
            X2CDT = pd.read_csv("2CDT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])   
            X2CDT = X2CDT.to_numpy()   
            self.dataset = X2CDT                                                                              
            step = 160
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X2CDT
            self.labels = X2CDT[:,-1]
            self.use = X2CDT[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data
        
        if self.datatype == 'UG_2C_3D':           
            UG_2C_3D = pd.read_csv("UG_2C_3D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3', 'feat4'])     
            UG_2C_3D = UG_2C_3D.to_numpy()   
            self.dataset = UG_2C_3D                                                                            
            step = 2000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = UG_2C_3D
            self.labels = UG_2C_3D[:,-1]
            self.use = UG_2C_3D[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data
        
        if self.datatype == '1CHT':           
            X1CHT = pd.read_csv("1CHT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            X1CHT = X1CHT.to_numpy()  
            self.dataset = X1CHT                                                                               
            step = 160
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X1CHT
            self.labels = X1CHT[:,-1]
            self.use = X1CHT[:step]
            self.use[:,-1][indx] = 1 

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data
        
        if self.datatype == '2CHT':           
            X2CHT = pd.read_csv("2CHT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                      
            X2CHT = X2CHT.to_numpy() 
            self.dataset = X2CHT                                                                                
            step = 160
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X2CHT
            self.labels = X1CHT[:,-1]
            self.use = X2CHT[:step]
            self.use[:,-1][indx] = 1   

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '4CR':           
            X4CR = pd.read_csv("4CR.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])  
            X4CR = X4CR.to_numpy()   
            self.dataset = X4CR                                                                               
            step = 400
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X4CR
            self.labels = X4CR[:,-1]
            self.use = X4CR[:step]
            self.use[:,-1][indx] = 1  

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '4CREV1':           
            X4CRE_V1 = pd.read_csv("4CRE-V1.txt", delimiter=",", names=['feat1', 'feat2', 'feat3']) 
            X4CRE_V1 = X4CRE_V1.to_numpy()   
            self.dataset = X4CRE_V1                                                                               
            step = 1000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X4CRE_V1
            self.labels = X4CRE_V1[:,-1]
            self.use = X4CRE_V1[:step]
            self.use[:,-1][indx] = 1 

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '4CREV2':           
            X4CRE_V2 = pd.read_csv("4CRE-V2.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                     
            X4CRE_V2 = X4CRE_V2.to_numpy()  
            self.dataset = X4CRE_V2                                                                                
            step = 1000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X4CRE_V2
            self.labels = X4CRE_V2[:,-1]
            self.use = X4CRE_V2[:step]
            self.use[:,-1][indx] = 1 

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '5CVT':           
            X5CVT = pd.read_csv("5CVT.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])                                                                                    
            X5CVT = X5CVT.to_numpy()  
            self.dataset = X5CVT                                                                                 
            step = 1000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X5CVT
            self.labels = X5CVT[:,-1]
            self.use = X5CVT[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '1CSURR':           
            X1Csurr = pd.read_csv("1Csurr.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            num = 600 * math.floor(len(X1Csurr)/600)
            X1Csurr = X1Csurr[0:num] 
            X1Csurr = X1Csurr.to_numpy() 
            self.dataset = X1Csurr                                                                            
            step = 600
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X1Csurr
            self.labels = X1Csurr[:,-1]
            self.use = X1Csurr[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == '4CE1CF':           
            X4CE1CF = pd.read_csv("4CE1CF.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])
            drift_no = 750
            step = drift_no * math.floor(len(X4CE1CF)/drift_no)
            X4CE1CF = X4CE1CF[0:step]                                                                           
            X4CE1CF = X4CE1CF.to_numpy()
            self.dataset = X4CE1CF 
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = X4CE1CF
            self.labels = X4CE1CF[:,-1]
            self.use = X4CE1CF[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == 'FG_2C_2D':           
            FG_2C_2D = pd.read_csv("FG_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])     
            FG_2C_2D = FG_2C_2D.to_numpy()  
            self.dataset = FG_2C_2D                                                                               
            step = 2000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = FG_2C_2D
            self.labels = FG_2C_2D[:,-1]
            self.use = FG_2C_2D[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == 'GEARS_2C_2D':           
            GEARS_2C_2D = pd.read_csv("GEARS_2C_2D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])  
            GEARS_2C_2D = GEARS_2C_2D.to_numpy()  
            self.dataset = GEARS_2C_2D                                                                                
            step = 2000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = GEARS_2C_2D
            self.labels = GEARS_2C_2D[:,-1]
            self.use = GEARS_2C_2D[:step]
            self.use[:,-1][indx] = 1

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == 'keystroke':           
            keystroke = pd.read_csv("keystroke.txt", delimiter=",", names=['feat1', 'feat2', 'feat3','feat4', 
                                    'feat5', 'feat6', 'feat7', 'feat8', 'feat9', 'feat10', 'feat11'])  
            keystroke = keystroke.to_numpy() 
            self.dataset = keystroke                                                                                 
            step = 200
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = keystroke
            self.labels = keystroke[:,-1]
            self.use = keystroke[:step]
            self.use[:,-1][indx] = 1
    
            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == 'UG_2C_5D':           
            UG_2C_5D = pd.read_csv("UG_2C_5D.txt", delimiter=",", names=['feat1', 'feat2', 'feat3',
                                    'feat4', 'feat5', 'feat6'])     
            UG_2C_5D = UG_2C_5D.to_numpy() 
            self.dataset = UG_2C_5D                                                                                 
            step = 2000
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = UG_2C_5D
            self.labels = UG_2C_5D[:,-1]
            self.use = UG_2C_5D[:step]
            self.use[:,-1][indx] = 1       

            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        if self.datatype == 'UnitTest':           
            unitTestData = pd.read_csv("unit_test.txt", delimiter=",", names=['feat1', 'feat2', 'feat3'])    
            unitTestData = unitTestData.to_numpy() 
            self.dataset = unitTestData
            step = 10                                                                          
            a = []
            indx = []
            for d in range(step-1):
                a.append(d)
            for v in range(int(0.5 * len(a))):
                rnd = random.choice(a)
                indx.append(rnd)
            
            self.data = UG_2C_5D
            self.labels = UG_2C_5D[:,-1]
            self.use = UG_2C_5D[:step]
            self.use[:,-1][indx] = 1  
        
            dataset = []
            data = []
            for i in self.batch(self.data, step):
                dataset.append(i)
            data.append(dataset)
            self.data = data

            labels = []
            lbl_data = []
            for i in self.batch(self.labels, step):
                labels.append(i)
            lbl_data.append(labels)
            self.labels = lbl_data

        return self.data, self.labels, self.use, self.dataset

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield np.array(iterable[ndx:min(ndx + n, l)])


# if __name__ == '__main__':
#     testData = COMPOSE_Datagen()
#     test_data, test_labels, test_use = testData.gen_dataset('UG_2C_2D')
#     print(test_data)
#     print("\n", test_labels, "\n")
#     print(test_use)

