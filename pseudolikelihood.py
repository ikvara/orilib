# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:22:36 2022

@author: iva
"""

import math
import numpy as np
import pickle as pckl
import multiprocessing
from mpmath import *
mp.dps = 10

from orifuncs import *
 
class nrm: 
    '''
    Newton-Raphson optimizing method

    To run the optimizing function 'fun' use the function self.run which returns the estimated theta and number of iterations
    '''
    def __init__(self, fun, theta, Procs, print_res=True):
        '''
        fun ... 
        theta ... initial value of the parameter
        Procs ... number of multiprocessors
        print_res ... Bool whether to print the running results, default=True
        '''
        self.f = fun 
        self.theta = theta 
        self.numProcessors = Procs
        self.print_res = print_res

        self.Jn = self.f.hess(self.theta, self.numProcessors)
        self.fn = self.f.grad(self.theta, self.numProcessors)
        if print_res:
            print('fn:', self.fn)
            print('ln PL:', self.f.logPL(self.theta))

    def step(self):
        self.theta -= self.fn/self.Jn
        if self.print_res:
            print('new thetas:', self.theta)

        self.fn = self.f.grad(self.theta, self.numProcessors)
        self.Jn = self.f.hess(self.theta, self.numProcessors)
        if self.print_res:
            print('fn new:', self.fn)

    def run(self, tol=6e-6, maxiter=np.inf):
        '''
        tol ... tolerance
        maxiter ... maximum of iterations
        '''
        it = 0
        self.step()
        if self.print_res:
            print('ln PL:', self.f.logPL(self.theta))

        while (abs(self.fn)>tol) and (it+1<=maxiter):
            it += 1
            self.step()
            if self.print_res:
                print('ln PL:', self.f.logPL(self.theta))

        return self.theta, it


# Auxiliary functions
def run_func_padd_vec(theta, fs_new, hs_new):
    val_num = 0
    val_den = 0

    for m in range(len(fs_new)):
        exp_val = np.exp(theta*hs_new[m])
        val_num += fs_new[m]*exp_val*hs_new[m]
        val_den += fs_new[m]*exp_val

    return -val_num/val_den

def run_func_padd_Hess(theta, fs_new, hs_new):
    val_num_1 = 0
    val_num_2 = 0
    val_den = 0

    for m in range(len(fs_new)):   
        exp_val = np.exp(theta*hs_new[m])
        val_num_1 += fs_new[m]*exp_val*hs_new[m]
        val_num_2 += fs_new[m]*exp_val*(hs_new[m]**2)
        val_den += fs_new[m]*exp_val
        
    H = (val_num_1/val_den)**2 - val_num_2/val_den

    return H  

def run_func_lnPL(theta, fs_orig, fs_new, h_orig, h_new):
    M = len(fs_new)

    C = 0
    for m in range(M):
        C += fs_new[m] * np.exp(theta * h_new[m])
    C *= math.pi**2 / (3*M)

    return -np.log(C) + np.log(fs_orig) + theta * h_orig


class pt:
    '''
    The class of pairwise interaction density
    '''
    def __init__(self, fs_orig, file_rep=None, den_ea=None, data_ors=None, neighs=None, step=0.1):
        '''
        fs_orig ... fitted semiparametric density evaluated on the data
        file_rep ... filename to load [fs_new, h_orig, h_new], 
                    fs_new ... fitted semiparametric density evaluated on a grid (see below the code)
                    h_orig ... inner products evaluated on the data (see below the code)
                    h_new ... inenr products evaluated on a grid (see below the code)
        
        If file_rep is not provided, then provide
        den_ea ... fitted density function without interactions (semiparametric density f_s)
        data_ors ... list of orientations
        neights ... list of neighbours [id1, id2, weight]
        step ... step of a grid of transformed fundamental zone 
        '''
        if file_rep is not None:
            file = open(file_rep,'rb')
            [self.fs_new, self.h_orig, self.h_new] = pckl.load(file)
            file.close()
        
        else:
            if data_ors is None:
                raise Exception("Provide a list of orientations!") 
            if neighs is None:
                raise Exception("Provide a neighbouring structure!") 

            N = len(data_ors)

            new_ors_tF = grid_tF(step_len=step)
            new_ors_F_M = [eu2mat(g[0], np.arccos(g[1]), g[2]) for g in new_ors_tF]
            new_ors_F_eu = [(g[0], np.arccos(g[1]), g[2]) for g in new_ors_tF]
            M = len(new_ors_F_M)

            inn_data = np.zeros((N, N))
            inn_new = np.zeros((M, N, N), dtype='f')
            
            for line in neighs:
                val = line[2]  * t_inner_prod(data_ors[line[0]], data_ors[line[1]]) # w_ij * inn(g_i, g_j)

                inn_data[line[0], line[1]] = val
                inn_data[line[1], line[0]] = val

                for m in range(M):
                    inn_new[m, line[0], line[1]] = line[2]  * t_inner_prod(new_ors_F_M[m], data_ors[line[1]])
                    inn_new[m, line[1], line[0]] = line[2]  * t_inner_prod(new_ors_F_M[m], data_ors[line[0]])

            self.fs_new = [den_ea(um) for um in new_ors_F_eu]
            self.h_orig = inn_data.sum(axis=1)
            self.h_new = inn_new.sum(axis=2)

        self.n = len(self.h_orig)
        self.fs_orig = fs_orig

    # evluate gradient in parameter theta
    def grad(self, theta, nrProc=1):
        pool = multiprocessing.Pool(nrProc)

        results_vec = []
        results_vec = pool.starmap_async(run_func_padd_vec, [(theta, self.fs_new, self.h_new[:,i]) for i in range(self.n)]).get()
        pool.close()

        vec = sum(self.h_orig)
        for sub_vec in results_vec:
            vec += sub_vec

        return vec

    # Hessian
    def hess(self, theta, nrProc=1):
        pool = multiprocessing.Pool(nrProc)

        results_H = []
        results_H = pool.starmap_async(run_func_padd_Hess, [(theta, self.fs_new, self.h_new[:,i]) for i in range(self.n)]).get()
        pool.close()

        H = 0
        for sub_H in results_H:
            H += sub_H

        return H

    # log pseudolikelihood
    def logPL(self, theta, nrProc=1):
        pool = multiprocessing.Pool(nrProc)

        results_PL = []
        results_PL = pool.starmap_async(run_func_lnPL, [(theta, self.fs_orig[i], self.fs_new, self.h_orig[i], self.h_new[:,i]) for i in range(self.n)]).get()
        pool.close()

        lnPL = 0
        for sub_lnPL in results_PL:
            lnPL += sub_lnPL

        return lnPL


    
