#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import division, print_function

from cython.parallel cimport prange

import numpy as np

def compute(double[::1] dts, int[:, ::1] HSDs, double[:, ::1] XP_hk, \
        double[:, ::1] XP_ok, double[:, ::1] XG_ok, double alpha, \
        double tau):
   
    cdef double[::1] aux = np.zeros(XP_ok.shape[0], dtype='d')
    cdef double[::1] rrs = np.zeros(HSDs.shape[0], dtype='d')
    cdef int i, h, s, d, candidate_d, k
    cdef double dt, alpha_to_use

    for i in xrange(HSDs.shape[0]):
        dt = dts[i]

        if dt > tau:
            alpha_to_use = 1.0
        else:
            alpha_to_use = alpha

        h = HSDs[i, 0]
        s = HSDs[i, 1]
        d = HSDs[i, 2]
        for candidate_d in prange(XP_ok.shape[0], schedule='static', nogil=True):
            aux[candidate_d] = 0.0

        for k in xrange(XP_ok.shape[1]):
            for candidate_d in prange(XP_ok.shape[0], schedule='static', nogil=True):
                aux[candidate_d] += alpha_to_use * \
                        (XP_hk[h, k] - XP_ok[candidate_d, k]) ** 2
            
            for candidate_d in prange(XP_ok.shape[0], schedule='static', nogil=True):
                aux[candidate_d] += (1 - alpha_to_use) * \
                        (XG_ok[s, k] - XG_ok[candidate_d, k]) ** 2
        
        for candidate_d in prange(XP_ok.shape[0], schedule='static', nogil=True):
            if aux[candidate_d] <= aux[d]:
                rrs[i] += 1
        rrs[i] = 1.0 / rrs[i]

    return np.array(rrs)
