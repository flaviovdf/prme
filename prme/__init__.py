#-*- coding: utf8
from __future__ import print_function, division

import dataio
import numpy as np
import os

from prme import sgd

def learn(trace_fpath, nk, rate, regularization, alpha, tau,
        from_=0, to=np.inf, validation=0.1):

    dts, Trace, seen, hyper2id, obj2id = \
            dataio.initialize_trace(trace_fpath, from_, to)
    no = len(obj2id)
    nh = len(hyper2id)
    
    validation_from = int(len(dts) - len(dts) * validation)
    print('Using first %d of %d as train, rest is validation' \
            % (validation_from, len(dts)))

    dts_train = dts[:validation_from]
    Trace_train = Trace[:validation_from]
    
    rnd_idx = np.arange(len(dts_train))
    np.random.shuffle(rnd_idx)

    dts_train = np.asanyarray(dts_train[rnd_idx], dtype='f8', order='C')
    Trace_train = np.asanyarray(Trace_train[rnd_idx], dtype='i4', order='C')

    dts_val = np.asanyarray(dts[validation_from:], dtype='f8', order='C')
    Trace_val = np.asanyarray(Trace[validation_from:], dtype='i4', order='C')

    XG_ok = np.random.normal(0, 0.01, (no, nk))
    XP_ok = np.random.normal(0, 0.01, (no, nk))
    XP_hk = np.random.normal(0, 0.01, (nh, nk))
    
    cost_train, cost_val = sgd(dts, Trace, XG_ok, XP_ok, XP_hk, seen, rate, \
            regularization, alpha, tau, dts_val, Trace_val)

    rv = {}
    rv['num_topics'] = np.asarray([nk])
    rv['trace_fpath'] = np.asarray([os.path.abspath(trace_fpath)])
    rv['rate'] = np.asarray([rate])
    rv['regularization'] = np.asarray([regularization])
    rv['alpha'] = np.asarray([alpha])
    rv['tau'] = np.asarray([alpha])
    rv['from_'] = np.asarray([from_])
    rv['cost_train'] = np.asarray([cost_train])
    rv['cost_val'] = np.asarray([cost_val])
    rv['to'] = np.asarray([to])
    rv['hyper2id'] = hyper2id
    rv['obj2id'] = obj2id
    rv['XG_ok'] = XG_ok
    rv['XP_ok'] = XP_ok
    rv['XP_hk'] = XP_hk
    return rv
 
