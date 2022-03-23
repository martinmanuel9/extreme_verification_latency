#!/usr/bin/env python 

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

import pandas as pd 
import argparse


def config_parser():
    '''build the parser for the main experiment script. 
    '''
    parser = argparse.ArgumentParser()
    # number of trials 
    parser.add_argument('-r',
                        '--runs',
                        type=int, 
                        default=5, 
                        help='number of trials to runs')
    # type of EVL 
    parser.add_argument('-c',
                        '--config',
                        type=str, 
                        required=True,  
                        help='experiment config file')
    return parser


def print_config(df:pd.DataFrame): 
    """print out the experiment configuration
    """
    print('>> Experiment Information >>')
    for key in df.keys(): 
        print(' - %s: %s ' % (key, str(df[key][0])))




