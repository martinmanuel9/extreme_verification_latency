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


import os 
import pandas as pd
import numpy as np 
from data import stream_file_loader, generate_stream
from utils import config_parser, print_config
from models import APT, ComposeV1, ComposeV2, FastCompose, LevelIW, MClassification, Scargc

import warnings
warnings.filterwarnings("ignore")


# run the main program 
if __name__ == '__main__': 
    parser = config_parser()
    args = parser.parse_args()

    # make the result directory if it does not exist 
    if not os.path.isdir('results/'): 
        os.mkdir('results/')

    # load the config file and make sure it exists. 
    try: 
        df_config = pd.read_csv(args.config)
    except: 
        raise(FileExistsError('Config file %s does not exist' % args.config))
    print_config(df_config)

    # read in the datastream
    Xinit, Yinit, Xt, Yt = stream_file_loader(experiment_name='1CDT', chunk_size=250)

    T = np.min([len(Xt), df_config['T'][0]])
    
    if df_config['model'][0] == 'apt': 
        mdl = APT(resample=True,
                  Xinit=Xinit, 
                  Yinit=Yinit, 
                  Kclusters=df_config['kcluster'][0], 
                  T=T)
    elif df_config['model'][0] == 'scargc': 
        mdl = Scargc(Xinit=Xinit, 
                     Yinit=Yinit, 
                     Kclusters=df_config['kcluster'][0], 
                     maxpool=25,
                     resample=True)
    else: 
        raise(ValueError('Unknown model: %s' % df_config['model'][0]))

    mdl.run(Xt, Yt)


