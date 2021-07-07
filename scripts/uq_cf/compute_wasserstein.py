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


def compute_vx_wasserstein(dir_sol, lx, M1, M2, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M1 samples
    for n in range(0, M1):
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vx = load_data_vx(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vx[i + j*Nx]
                data1[i, j, n] = val
    
    # upsampling of data1, assumes that M1 is a multiple of 10
    for n in range(0, M2-M1):
        nn = n
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx = load_data_vx(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vx[i + j*Nx]
                data1[i, j, n+M1] = val
    
    
    # data2, level lx, Nx \times Ny uniform grid, with M2 samples
    for n in range(0, M2):
        #nn = (M2-M1)+n
        nn = n
        f2 = dir_sol+'/samples'+str(M2)+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx = load_data_vx(f2)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vx[i + j*Nx]
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt vx: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt vx: ", val) 
    
    return val


def compute_vy_wasserstein(dir_sol, lx, M1, M2, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M1 samples
    for n in range(0, M1):
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vy = load_data_vy(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vy[i + j*Nx]
                data1[i, j, n] = val
    
    # upsampling of data1, assumes that M1 is a multiple of 10
    for n in range(0, M2-M1):
        nn = n
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vy = load_data_vy(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vy[i + j*Nx]
                data1[i, j, n+M1] = val
    
    # data2, level lx, Nx \times Ny uniform grid, with M2 samples
    for n in range(0, M2):
        f2 = dir_sol+'/samples'+str(M2)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vy = load_data_vy(f2)
        for i in range(0,Nx):
            for j in range(0,Ny):
                val = vy[i + j*Nx]
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt vy: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt vy: ", val) 
    
    return val
    

def compute_v_wasserstein(dir_sol, lx, M1, M2, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, M2], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M1 samples
    for n in range(0, M1):
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                val = np.sqrt(valx**2 + valy**2)
                data1[i, j, n] = val
    
    # upsampling of data1, assumes that M1 is a multiple of 10
    for n in range(0, M2-M1):
        nn = n
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                val = np.sqrt(valx**2 + valy**2)
                data1[i, j, n+M1] = val
    
    # data2, level lx, Nx \times Ny uniform grid, with M2 samples
    for n in range(0, M2):
        f2 = dir_sol+'/samples'+str(M2)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
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


def compute_vector_wasserstein(dir_sol, lx, M1, M2, Nx, Ny, metric):
    
    data1 = np.zeros([Nx, Ny, M2, 2], dtype=np.float64)
    data2 = np.zeros([Nx, Ny, M2, 2], dtype=np.float64)
    
    # data1, level lx, Nx \times Ny uniform grid, with M1 samples
    for n in range(0, M1):
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                data1[i, j, n, 0] = valx
                data1[i, j, n, 1] = valy
    
    # upsampling of data1
    for n in range(0, M2-M1):
        nn = n
        f1 = dir_sol+'/samples'+str(M1)+'/pp_lx'+str(lx)+'/velocity_s'+str(nn)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx):
            for j in range(0,Ny):
                valx = vx[i + j*Nx]
                valy = vy[i + j*Nx]
                data1[i, j, n+M1, 0] = valx
                data1[i, j, n+M1, 1] = valy
    
    # data2, level lx, Nx \times Ny uniform grid, with M2 samples
    for n in range(0, M2):
        f2 = dir_sol+'/samples'+str(M2)+'/pp_lx'+str(lx)+'/velocity_s'+str(n)+'.json'
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
    
    #nsamples_ = np.array([30, 60, 120, 240])
    nsamples_ = np.array([30, 60, 120, 240, 480])
    
    lx_ = 3
    metric = 2
    
    # Re = 1600
    Re = 3200
    dir_sol = '../../output/uq_pincompNS_cf/Re'+str(Re)
    
    J = nsamples_.size-1
    wd = np.zeros((4, J))
    for k in range(0, J):
        wd[0,k] = compute_vx_wasserstein(dir_sol, lx_, nsamples_[k], nsamples_[k+1], Nx, Ny, metric)
        wd[1,k] = compute_vy_wasserstein(dir_sol, lx_, nsamples_[k], nsamples_[k+1], Nx, Ny, metric)
        wd[2,k] = compute_v_wasserstein(dir_sol, lx_, nsamples_[k], nsamples_[k+1], Nx, Ny, metric)
        wd[3,k] = compute_vector_wasserstein(dir_sol, lx_, nsamples_[k], nsamples_[k+1], Nx, Ny, metric)
    
    wd_data = {}
    wd_data['vx'] = list(wd[0,:])
    wd_data['vy'] = list(wd[1,:])
    wd_data['v'] = list(wd[2,:])
    wd_data['vector'] = list(wd[3,:])
    
    Mbuf = []
    for k in range(0, J):
        Mbuf.append(int(nsamples_[k]))
    wd_data['M'] = list(Mbuf)
    
    print(wd_data)
    filename = '../../output/uq_pincompNS_cf/Re'+str(Re)+'/convergence/wasserstein_cf_lx'+str(lx_)+'_'+str(metric)+'pt.json'
    with open(filename, 'w') as outfile:
        json.dump(wd_data, outfile)
    
    
# End of file
