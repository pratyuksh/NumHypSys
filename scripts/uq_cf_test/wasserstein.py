#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ot


def wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt):
    """
    Computes the 1-pt Wasserstein distance for a single point in the spatain domain
    """
    
    N = data1.shape[2]
    
    xs[:] = (data1[i,j,:]).reshape((N,1))
    xt[:] = (data2[i,j,:]).reshape((N,1))

    M = ot.dist(xs, xt, metric='euclidean')
    
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
    
    Nx = data1.shape[0]
    Ny = data1.shape[1]
    M = data1.shape[2]
    a = np.ones(M)/M
    b = np.ones(M)/M
    xs = np.zeros((M,1))
    xt = np.zeros((M,1))
    distance = 0
    
    for i in range(Nx):
        for j in range(Ny):
            distance += wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt)
            # print(n, i, j)
    
    return distance / (Nx*Ny)

def vector_wasserstein1pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    Nx = data1.shape[0]
    Ny = data1.shape[1]
    M = data1.shape[2]
    d = data1.shape[3]
    
    a = np.ones(M)/M
    b = np.ones(M)/M
    xs = np.zeros((M,d))
    xt = np.zeros((M,d))
    distance = 0

    for i in range(Nx):
        for j in range(Ny):
            distance += vector_wasserstein_point1_fast(data1, data2, i, j, a, b, xs, xt)
            # print(n, i, j)

    return distance / (Nx*Ny)

def wasserstein2pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    Nx = data1.shape[0]
    Ny = data1.shape[1]
    M = data1.shape[2]
    a = np.ones(M)/M
    b = np.ones(M)/M
    xs = np.zeros((M,2))
    xt = np.zeros((M,2))
    distance = 0
    
    for i in range(Nx):
        for j in range(Ny):

            for ip in range(Nx):
                for jp in range(Ny):
                    
                    distance += wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt)
                    # print(n, i, j, ip, jp)

    return distance / (Nx*Ny)**2
    
def vector_wasserstein2pt_fast(data1, data2):
    """
    Approximate the L^1(W_1) distance (||W_1(nu1, nu2)||_{L^1})
    """
    Nx = data1.shape[0]
    Ny = data1.shape[1]
    M = data1.shape[2]
    d = data1.shape[3]
    
    a = np.ones(M)/M
    b = np.ones(M)/M
    xs = np.zeros((M,2*d))
    xt = np.zeros((M,2*d))
    distance = 0

    points = 0.1*np.array(range(0,10))
    for i in range(Nx):
        for j in range(Ny):

            for ip in range(Nx):
                for jp in range(Ny):
                    
                    distance += vector_wasserstein_point2_fast(data1, data2, i,j, ip, jp, a, b, xs, xt)
                    # print(n, i, j, ip, jp)

    return distance / (Nx*Ny)**2


# End of file
