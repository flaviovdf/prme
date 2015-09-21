#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import print_function, division

import numpy as np

from prme.myrandom.random cimport rand

cdef extern from 'math.h':
    inline double exp(double)
    inline double log(double)
    inline double abs(double)

cdef inline double sigma(double z):
    return 1.0 / (1 + exp(-z))

cdef inline double compute_dist(int h, int s, int d, double alpha, \
        double[:, ::1] XG_ok, double[:, ::1] XP_ok, double[:, ::1] XP_hk):

    cdef double dp_ho = 0.0
    cdef double ds_oo = 0.0
    cdef int k
    for k in range(XG_ok.shape[1]):
        dp_ho += alpha * ((XP_ok[d, k] - XP_hk[h, k]) ** 2)
        ds_oo += (1 - alpha) * ((XG_ok[d, k] - XG_ok[s, k]) ** 2)
    return dp_ho + ds_oo

cdef inline void update(int row, double[:, ::1] X, double[::1] update):
    cdef int k
    for k in range(X.shape[1]):
        X[row, k] += update[k]

cdef void do_iter(double[::1] dts, int[:, ::1] Trace, double[:, ::1] XG_ok, \
        double[:, ::1] XP_ok, double[:, ::1] XP_hk, dict seen, \
        double rate, double regularization, double alpha, double tau):

    cdef int i, k, h, s, d_old, d_new
    cdef double dt, z, sigma_z
    
    cdef double[::1] update_XP_h = np.zeros(XP_ok.shape[1], dtype='d') 
    cdef double[::1] update_XP_dnew = np.zeros(XP_ok.shape[1], dtype='d')
    cdef double[::1] update_XP_dold = np.zeros(XP_ok.shape[1], dtype='d')

    cdef double[::1] update_XG_s = np.zeros(XP_ok.shape[1], dtype='d') 
    cdef double[::1] update_XG_dnew = np.zeros(XP_ok.shape[1], dtype='d')
    cdef double[::1] update_XG_dold = np.zeros(XP_ok.shape[1], dtype='d')

    cdef double[::1] deriv_XP_h = np.zeros(XP_ok.shape[1], dtype='d') 
    cdef double[::1] deriv_XP_dnew = np.zeros(XP_ok.shape[1], dtype='d')
    cdef double[::1] deriv_XP_dold = np.zeros(XP_ok.shape[1], dtype='d')

    cdef double[::1] deriv_XG_s = np.zeros(XP_ok.shape[1], dtype='d') 
    cdef double[::1] deriv_XG_dnew = np.zeros(XP_ok.shape[1], dtype='d')
    cdef double[::1] deriv_XG_dold = np.zeros(XP_ok.shape[1], dtype='d')

    cdef set seen_hs
    cdef double alpha_to_use
    
    for i in xrange(Trace.shape[0]):
        dt = dts[i]
        h = Trace[i, 0]
        s = Trace[i, 1]
        d_old = Trace[i, 2]

        if dt >= tau:
            alpha_to_use = 1.0
            d_new = <int> (XP_ok.shape[0] * rand())
            while d_new == d_old:
                d_new = <int> (XP_ok.shape[0] * rand())
        else:
            alpha_to_use = alpha
            seen_hs = seen[h, s]
            d_new = <int> (XP_ok.shape[0] * rand())
            while d_new in seen_hs:
                d_new = <int> (XP_ok.shape[0] * rand())
            
        z = compute_dist(h, s, d_new, alpha_to_use, \
                XG_ok, XP_ok, XP_hk)
        z -= compute_dist(h, s, d_old, alpha_to_use, \
                XG_ok, XP_ok, XP_hk)
        sigma_z = sigma(z)

        #Compute derivatives dz/dTheta and zero auxiliary
        for k in range(XP_ok.shape[1]):
            update_XP_h[k] = 0.0
            update_XP_dnew[k] = 0.0
            update_XP_dold[k] = 0.0
            
            update_XG_s[k] = 0.0
            update_XG_dnew[k] = 0.0
            update_XG_dold[k] = 0.0
            
            #1. XP_h deriv
            deriv_XP_h[k] = XP_ok[d_old, k] - XP_ok[d_new, k]
            
            #2. XP_o(d_new) deriv
            deriv_XP_dnew[k] = XP_ok[d_new, k] - XP_hk[h, k]
            
            #3. XP_o(d_old) deriv
            deriv_XP_dold[k] = -(XP_ok[d_old, k] - XP_hk[h, k])

            #4. XG_o(s) deriv
            deriv_XG_s[k] = XG_ok[d_old, k] - XG_ok[d_new, k]

            #5. XG_o(d_new) deriv
            deriv_XG_dnew[k] = XG_ok[d_new, k] - XG_ok[s, k]
        
            #6. XG_o(d_old) deriv
            deriv_XG_dold[k] = -(XG_ok[d_old, k] - XG_ok[s, k])
        
        for k in range(XP_ok.shape[1]):
            deriv_XP_h[k] *= 2 * alpha_to_use
            deriv_XP_dnew[k] *= 2 * alpha_to_use
            deriv_XP_dold[k] *= 2 * alpha_to_use
        
            deriv_XG_s[k] *= 2 * (1 - alpha_to_use)
            deriv_XG_dnew[k] *= 2 * (1 - alpha_to_use)
            deriv_XG_dold[k] *= 2 * (1 - alpha_to_use)
        
        for k in range(XP_ok.shape[1]):
            update_XP_h[k] = rate * ((1 - sigma_z) * deriv_XP_h[k] - \
                    (2 * regularization * XP_hk[h, k]))
            
            update_XP_dnew[k] = rate * ((1 - sigma_z) * deriv_XP_dnew[k] - \
                    (2 * regularization * XP_ok[d_new, k]))
            
            update_XP_dold[k] = rate * ((1 - sigma_z) * deriv_XP_dold[k] - \
                    (2 * regularization * XP_ok[d_old, k]))
        
        if dt < tau:
            for k in range(XP_ok.shape[1]):
                update_XG_s[k] = rate * ((1 - sigma_z) * deriv_XG_s[k] - \
                        (2 * regularization * XG_ok[s, k]))
                
                update_XG_dnew[k] = rate * ((1 - sigma_z) * deriv_XP_dnew[k] - \
                        (2 * regularization * XG_ok[d_new, k]))
                
                update_XG_dold[k] = rate * ((1 - sigma_z) * deriv_XG_dold[k] - \
                        (2 * regularization * XG_ok[d_old, k]))
        
        update(h, XP_hk, update_XP_h)
        update(d_new, XP_ok, update_XP_dnew)
        update(d_old, XP_ok, update_XP_dold)
        
        update(s, XG_ok, update_XG_s)
        update(d_new, XG_ok, update_XG_dnew)
        update(d_old, XG_ok, update_XG_dold)
        
def compute_cost(double[::1] dts, int[:, ::1] Trace, double[:, ::1] XG_ok, \
        double[:, ::1] XP_ok, double[:, ::1] XP_hk, dict seen, double rate, \
        double regularization, double alpha, double tau, int num_examples=-1, \
        int num_candidates=-1):

    cdef int i, j, h, s, d_old, d_new
    cdef double dt, z
    
    cdef set seen_hs
    cdef double alpha_to_use
    cdef double cost = 0.0
    cdef double curr_cost = 0.0
    cdef dict precomputed = {}
    
    cdef int[::1] idx = np.arange(Trace.shape[0], dtype='i4')
    if num_examples > 0:
        np.random.shuffle(idx)
        idx = idx[:num_examples]
    
    cdef int[::1] candidates = np.arange(XG_ok.shape[0], dtype='i4')
    if num_candidates > 0:
        np.random.shuffle(candidates)
        candidates = candidates[:num_candidates]

    for i in xrange(idx.shape[0]):
        dt = dts[idx[i]]
        h = Trace[idx[i], 0]
        s = Trace[idx[i], 1]
        d_old = Trace[idx[i], 2]

        if (h, s, dt >= tau) in precomputed:
            cost += precomputed[h, s, dt >= tau]
            continue

        if dt >= tau:
            alpha_to_use = 1.0
        else:
            alpha_to_use = alpha
        
        curr_cost = 0.0
        for j in xrange(candidates.shape[0]):
            d_new = candidates[j]
            z = compute_dist(h, s, d_new, alpha_to_use, \
                    XG_ok, XP_ok, XP_hk)
            z -= compute_dist(h, s, d_old, alpha_to_use, \
                     XG_ok, XP_ok, XP_hk)
            curr_cost += log(sigma(z))

        precomputed[h, s, dt >= tau] = curr_cost
        cost += curr_cost
    return cost

def sgd(double[::1] dts, int[:, ::1] Trace, double[:, ::1] XG_ok, \
        double[:, ::1] XP_ok, double[:, ::1] XP_hk, dict seen,
        double rate, double regularization, double alpha, double tau,
        double[::1] dts_val, int[:, ::1] Trace_val):
    
    cost_train = 0.0
    cost_val = 0.0
    i = 0
    while i < 1000:
        do_iter(dts, Trace, XG_ok, XP_ok, XP_hk, seen, rate, \
                regularization, alpha, tau)
        i += 1

    cost_train = compute_cost(dts_val, Trace_val, XG_ok, XP_ok, XP_hk, \
            seen, rate, regularization, alpha, tau, 1000, 1000)
    cost_val = compute_cost(dts_val, Trace_val, XG_ok, XP_ok, XP_hk, \
            seen, rate, regularization, alpha, tau, 1000, 1000)
    return cost_train, cost_val
