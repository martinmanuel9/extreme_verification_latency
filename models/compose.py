#!/usr/bin/env python 

"""
Application:        COMPOSE Framework 
File name:          compose.py
Author:             Martin Manuel Lopez
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




class ComposeV1(): 
    def __init__(self, 
                 classifier, 
                 method): 
        """
        Initialization of COMPOSEV1
        """
        _timestep = 1                   # [INTEGER] The current timestep of the datase
        _synthetic = 0                  # [INTEGER] 1 Allows synthetic data during cse and {0} does not allow synthetic data
        _n_cores =  1                   # [INTEGER] Level of feedback displayed during run {default}
        _verbose = 1                    #    0  : No Information Displayed
                                        #   {1} : Command line progress updates
                                        #    2  : Plots when possible and Command line progress updates
        self.classifier = classifier

    def validateComposeAttributes(self, dataset, verbose, *args):
        """
        Check if the input parameters are not empty for compose
        This checks if the dataset is empty and checks what option of feedback you want
        """
        # need to limit arguements to 2 for dataset and verbose 
        max_args = 2
        try:
            len(*args) <= max_args
        except ValueError:
            print("Number of input parameters must be a min of two. Input valid dataset and valid option to display information")


    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier
    

class ComposeV2(): 
    """
    """
    def __init__(self, 
                 classifier, 
                 method): 
        self.classifier = classifier 
    
    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier


class FastCompose(): 
    """
    """
    def __init__(self, 
                 classifier, 
                 method): 
        self.classifier = classifier 

    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier