#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ot


def wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt):
    """
    Computes the 1-pt Wasserstein distance for a single point in the spatain domain
    """
    
    N = data1.shape[0]
    
    xs[:] = (data1[i,j,:]).reshape((N,1))
    xt[:] = (data2[i,j,:]).reshape((N,1))

    M = ot.dist(xs, xt, metric='euclidean')
    # G0 = ot.emd(a,b,M)
    # return np.sum(G0*M)
    return ot.emd2(a,b,M)

def vector_wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt):
    """
    Computes the 1-pt vector Wasserstein distance for a single point in the spatial domain
    Based on Kjetil Lye's work.
    """
    xs[:,:] = (data1[i,j,:,:])
    xt[:,:] = (data2[i,j,:,:])

    M = ot.dist(xs, xt, metric='euclidean')

    return ot.emd2(a,b,M)


def wasserstein_point2_fast(data1, data2, i, j, ip, jp, a, b, xs, xt):
    """
    Computes the 2-pt Wasserstein distance for a single point in the spatain domain
    """

    xs[:,0] = data1[i,j,:]
    xs[:,1] = data1[ip, jp, :]

    xt[:,0] = data2[i,j, :]
    xt[:,1] = data2[ip, jp, :]
    

    M = ot.dist(xs, xt, metric='euclidean')
    # G0 = ot.emd(a,b,M)
    # return np.sum(G0*M)
    return ot.emd2(a,b,M)

def vector_wasserstein_point2_fast(data1, data2, i, j, ip, jp, a, b, xs, xt):
    """
    Computes the 2-pt vector Wasserstein distance for a single point in the spatial domain
    Based on Kjetil Lye's work.
    """
    xs[:,0:2] = (data1[i,j,:,:])
    xs[:,2:4] = (data1[ip,jp,:,:])

    xt[:,0:2] = (data2[i,j,:,:])
    xt[:,2:4] = (data2[ip,jp,:,:])
    
    M = ot.dist(xs, xt, metric='euclidean')

    return ot.emd2(a,b,M)
    
    
def wasserstein1pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N,1))
    xt = np.zeros((N,1))
    distance = 0

    points = 0.1*np.array(range(0,10))
    for (n,x) in enumerate(points):
        for y in points:

            i = int(x*N)
            j = int(y*N)
                    
            distance += wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt)
            # print(n, i, j)

    return distance / len(points)**2

def vector_wasserstein1pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    d = data1.shape[3]
    
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N,d))
    xt = np.zeros((N,d))
    distance = 0

    points = 0.1*np.array(range(0,10))
    for (n,x) in enumerate(points):
        for y in points:

            i = int(x*N)
            j = int(y*N)
            
            distance += vector_wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt)
            # print(n, i, j)

    return distance / len(points)**2
   
    
def wasserstein2pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N,2))
    xt = np.zeros((N,2))
    distance = 0

    points = 0.1*np.array(range(0,10))
    for (n,x) in enumerate(points):
        for y in points:

            for xp in points:
                for yp in points:
                    i = int(x*N)
                    j = int(y*N)
                    ip = int(xp*N)
                    jp = int(yp*N)
                    
                    distance += wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt)
                    # print(n, i, j, ip, jp)

    return distance / len(points)**4
    
def vector_wasserstein2pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    N = data1.shape[0]
    d = data1.shape[3]
    
    a = np.ones(N)/N
    b = np.ones(N)/N
    xs = np.zeros((N,2*d))
    xt = np.zeros((N,2*d))
    distance = 0

    points = 0.1*np.array(range(0,10))
    for (n,x) in enumerate(points):
        for y in points:

            for xp in points:
                for yp in points:
                    i = int(x*N)
                    j = int(y*N)
                    ip = int(xp*N)
                    jp = int(yp*N)
                    
                    distance += vector_wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt)
                    # print(n, i, j, ip, jp)

    return distance / len(points)**4
