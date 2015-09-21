#-*- coding: utf8
from __future__ import division, print_function

from prme import dataio
from prme import learn

import argparse
import numpy as np
import pandas as pd
import os
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trace_fpath', help='The trace to learn topics from', \
            type=str)
    parser.add_argument('num_topics', help='The number of topics to learn', \
            type=int)
    parser.add_argument('model_fpath', \
            help='The name of the model file (a h5 file)', type=str)
    parser.add_argument('leaveout', \
            help='The number of transitions to leave for test', type=float, \
            default=0.3)

    parser.add_argument('--learning_rate', \
            help='The learning rate for the algorithm', \
            type=float, default=0.005)
    parser.add_argument('--regularization', help='The regularization', \
            type=float, default=0.03)
    parser.add_argument('--alpha', help='Value for the alpha parameter', \
            type=float, default=0.02)
    parser.add_argument('--tau', help='Value for the tau parameter', \
            type=float, default=3 * 60 * 50)
    

    args = parser.parse_args()
    started = time.mktime(time.localtime())

    df = pd.read_csv(trace_fpath, sep='\t', names=['dt', 'u', 's', 'd'], \
            dtype={'dt':float, 'u': str, 's':str, 'd':str})
    num_lines = len(df)
    to = int(num_lines - num_lines * args.leaveout)
    
    rate = 0.005
    for alpha in [0.2, 0.5, 0.8]:
        for tau in [0, 60, 5 * 60, 1 * 60 * 60, 12 * 60 * 60, 24 * 60 * 60]:


    rv = learn(args.trace_fpath, args.num_topics, rate, \
            best_alpha, args.alpha, args.tau, from_, to)
    ended = time.mktime(time.localtime())
    rv['training_time'] = np.array([ended - started])
    dataio.save_model(args.model_fpath, rv)
    print('Learning took', ended - started, 'seconds')

if __name__ == '__main__':
    main()
