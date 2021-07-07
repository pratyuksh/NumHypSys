#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json

import wasserstein as wm


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data_vx(filename):
    j = load_json(filename)
    return np.array(j["vx"])
    
def load_data_vy(filename):
    j = load_json(filename)
    return np.array(j["vy"])
    
def load_data(filename):
    j = load_json(filename)
    return np.array(j["vx"]), np.array(j["vy"])


def compute_v_wasserstein(dir_sol, lx, M, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, 2*M], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, 2*M], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M samples
    for n in range(0, M):
        f1 = dir_sol+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                val = np.sqrt(valx**2 + valy**2)
                data1[i, j, n] = val
    
    # upsampling of data1
    for n in range(0, M):
        nn = n
        f1 = dir_sol+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                val = np.sqrt(valx**2 + valy**2)
                data1[i, j, n+M] = val
    
    # data2, level lx+1, Nx \times Ny uniform grid, with 2*M samples
    for n in range(0, 2*M):
        f2 = dir_sol+'/pp_lx'+str(lx+1)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f2)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                val = np.sqrt(valx**2 + valy**2)
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt v: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt v: ", val) 
    
    return val


def compute_vector_wasserstein(dir_sol, lx, M, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, 2*M, 2], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, 2*M, 2], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M samples
    for n in range(0, M):
        f1 = dir_sol+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                data1[i, j, n, 0] = valx
                data1[i, j, n, 1] = valy
    
    # upsampling of data1
    for n in range(0, M):
        nn = n
        f1 = dir_sol+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                data1[i, j, n+M, 0] = valx
                data1[i, j, n+M, 1] = valy
    
    # data2, level lx, Nx \times Ny uniform grid, with 2*M samples
    for n in range(0, 2*M):
        f2 = dir_sol+'/pp_lx'+str(lx+1)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f2)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                data2[i, j, n, 0] = valx
                data2[i, j, n, 1] = valy
    
    if metric == 1:
        val = wm.vector_wasserstein1pt_fast(data1, data2)
        print("Vector Wasserstein distance 1pt: ", val)
    elif metric == 2:
        val = wm.vector_wasserstein2pt_fast(data1, data2)
        print("Vector Wasserstein distance 2pt: ", val) 
    
    return val



if __name__ == '__main__':
    
    Nx = 15
    Ny = 5
    
    lx_ = np.array([0,1,2,3])
    nsamples_ = np.array([60, 120, 240, 480])
    
    metric = 2
    
    # Re = 1600
    Re = 3200
    dir_sol = '../../output/uq_pincompNS_cf_test/Re'+str(Re)
    
    J = nsamples_.size-1
    wd = np.zeros((2, J))
    for k in range(0, J):
        wd[0,k] = compute_v_wasserstein(dir_sol, lx_[k], nsamples_[k], Nx, Ny, metric)
        wd[1,k] = compute_vector_wasserstein(dir_sol, lx_[k], nsamples_[k], Nx, Ny, metric)
    
    wd_data = {}
    wd_data['v'] = list(wd[0,:])
    wd_data['vector'] = list(wd[1,:])
    
    lxbuf = []
    Mbuf = []
    for k in range(0, J):
        lxbuf.append(int(lx_[k]))
        Mbuf.append(int(nsamples_[k]))
    wd_data['lx'] = list(lxbuf)
    wd_data['M'] = list(Mbuf)
    
    print(wd_data)
    filename = '../../output/uq_pincompNS_cf_test/Re'+str(Re)+'/convergence/wasserstein_cf_'+str(metric)+'pt.json'
    with open(filename, 'w') as outfile:
        json.dump(wd_data, outfile)
    
    
# End of file
