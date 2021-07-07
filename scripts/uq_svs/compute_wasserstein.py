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


def compute_vx_wasserstein(dir_sol, lx, Nx, metric):
    
    data1 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    data2 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    
    # data1, on lx-1, Nx/2 \times Nx/2 mesh
    # upsampling
    lx1 = lx-1
    Nx1 = int(Nx/2)
    for n in range(0, Nx1):
        f1 = dir_sol+'/pp_lx'+str(lx1)+'/velocity_s'+str(n)+'.json'
        vx = load_data_vx(f1)
        for i in range(0,Nx1):
            for j in range(0,Nx1):
                val = vx[i + j*Nx1]
                for ii in range(0,2):
                    for jj in range(0,2):
                        data1[ii+2*i,  jj+2*j,  2*n] = val
                        data1[ii+2*i,  jj+2*j,  2*n+1] = val
    # print(data1[0:4,0:4,0])
    
    # data2, on lx, Nx \times Nx mesh
    lx2 = lx
    Nx2 = Nx
    for n in range(0, Nx2):
        f2 = dir_sol+'/pp_lx'+str(lx2)+'/velocity_s'+str(n)+'.json'
        vx = load_data_vx(f2)
        for i in range(0,Nx2):
            for j in range(0,Nx2):
                val = vx[i + j*Nx2]
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt vx: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt vx: ", val) 
    
    return val

def compute_vy_wasserstein(dir_sol, lx, Nx, metric):
    
    data1 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    data2 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    
    # data1, on lx-1, Nx/2 \times Nx/2 mesh
    # upsampling
    lx1 = lx-1
    Nx1 = int(Nx/2)
    for n in range(0, Nx1):
        f1 = dir_sol+'/pp_lx'+str(lx1)+'/velocity_s'+str(n)+'.json'
        vy = load_data_vy(f1)
        for i in range(0,Nx1):
            for j in range(0,Nx1):
                val = vy[i + j*Nx1]
                for ii in range(0,2):
                    for jj in range(0,2):
                        data1[ii+2*i,  jj+2*j,  2*n] = val
                        data1[ii+2*i,  jj+2*j,  2*n+1] = val
    # print(data1[0:4,0:4,0])
    
    # data2, on lx, Nx \times Nx mesh
    lx2 = lx
    Nx2 = Nx
    for n in range(0, Nx2):
        f2 = dir_sol+'/pp_lx'+str(lx2)+'/velocity_s'+str(n)+'.json'
        vy = load_data_vy(f2)
        for i in range(0,Nx2):
            for j in range(0,Nx2):
                val = vy[i + j*Nx2]
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt vy: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt vy: ", val) 
    
    return val

def compute_v_wasserstein(dir_sol, lx, Nx, metric):
    
    data1 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    data2 = np.zeros([Nx, Nx, Nx], dtype=np.float64)
    
    # data1, on lx-1, Nx/2 \times Nx/2 mesh
    # upsampling
    lx1 = lx-1
    Nx1 = int(Nx/2)
    for n in range(0, Nx1):
        f1 = dir_sol+'/pp_lx'+str(lx1)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx1):
            for j in range(0,Nx1):
                valx = vx[i + j*Nx1]
                valy = vy[i + j*Nx1]
                val = np.sqrt(valx**2 + valy**2)
                for ii in range(0,2):
                    for jj in range(0,2):
                        data1[ii+2*i,  jj+2*j,  2*n] = val
                        data1[ii+2*i,  jj+2*j,  2*n+1] = val
    # print(data1[0:4,0:4,0])
    
    # data2, on lx, Nx \times Nx mesh
    lx2 = lx
    Nx2 = Nx
    for n in range(0, Nx2):
        f2 = dir_sol+'/pp_lx'+str(lx2)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f2)
        for i in range(0,Nx2):
            for j in range(0,Nx2):
                valx = vx[i + j*Nx2]
                valy = vy[i + j*Nx2]
                val = np.sqrt(valx**2 + valy**2)
                data2[i, j, n] = val
    
    if metric == 1:
        val = wm.wasserstein1pt_fast(data1, data2)
        print("Wasserstein distance 1pt v: ", val)
    elif metric == 2:
        val = wm.wasserstein2pt_fast(data1, data2)
        print("Wasserstein distance 2pt v: ", val)
    
    return val

def compute_vector_wasserstein(dir_sol, lx, Nx, metric):
    
    data1 = np.zeros([Nx, Nx, Nx, 2], dtype=np.float64)
    data2 = np.zeros([Nx, Nx, Nx, 2], dtype=np.float64)
    
    # data1, on lx-1, Nx/2 \times Nx/2 mesh
    # upsampling
    lx1 = lx-1
    Nx1 = int(Nx/2)
    for n in range(0, Nx1):
        f1 = dir_sol+'/pp_lx'+str(lx1)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f1)
        for i in range(0,Nx1):
            for j in range(0,Nx1):
                valx = vx[i + j*Nx1]
                valy = vy[i + j*Nx1]
                for ii in range(0,2):
                    for jj in range(0,2):
                        data1[ii+2*i, jj+2*j, 2*n, 0] = valx
                        data1[ii+2*i, jj+2*j, 2*n, 1] = valy
                        data1[ii+2*i, jj+2*j, 2*n+1, 0] = valx
                        data1[ii+2*i, jj+2*j, 2*n+1, 1] = valy
    
    # data2, on lx, Nx \times Nx mesh
    lx2 = lx
    Nx2 = Nx
    for n in range(0, Nx2):
        f2 = dir_sol+'/pp_lx'+str(lx2)+'/velocity_s'+str(n)+'.json'
        vx, vy = load_data(f2)
        for i in range(0,Nx2):
            for j in range(0,Nx2):
                valx = vx[i + j*Nx2]
                valy = vy[i + j*Nx2]
                data2[i, j, n, 0] = valx
                data2[i, j, n, 1] = valy
    
    val = 0
    if metric == 1:
        val = wm.vector_wasserstein1pt_fast(data1, data2)
        print("Vector Wasserstein distance 1pt: ", val)
    elif metric == 2:
        val = wm.vector_wasserstein2pt_fast(data1, data2)
        print("Vector Wasserstein distance 2pt: ", val)
    
    return val
    

if __name__ == '__main__':
    
    # lx_ = np.array([4, 5, 6])
    lx_ = np.array([4,5,6,7])
    Nx_ = 4*2**lx_
    
    dir_sol = '../../output/uq_pincompNS_svs/'
    metric = 2
    
    wd = np.zeros((4, lx_.size))
    for k in range(0, lx_.size):
        wd[0,k] = compute_vx_wasserstein(dir_sol, lx_[k], Nx_[k], metric)
        wd[1,k] = compute_vy_wasserstein(dir_sol, lx_[k], Nx_[k], metric)
        wd[2,k] = compute_v_wasserstein(dir_sol, lx_[k], Nx_[k], metric)
        wd[3,k] = compute_vector_wasserstein(dir_sol, lx_[k], Nx_[k], metric)
    
    wd_data = {}
    wd_data['vx'] = list(wd[0,:])
    wd_data['vy'] = list(wd[1,:])
    wd_data['v'] = list(wd[2,:])
    wd_data['vector'] = list(wd[3,:])
    
    Nxbuf = []
    for k in range(0, lx_.size):
        Nxbuf.append(int(Nx_[k]))
    wd_data['Nx'] = list(Nxbuf)
    
    print(wd_data)
    filename = dir_sol+'convergence/wasserstein_'+str(metric)+'pt.json'
    with open(filename, 'w') as outfile:
        json.dump(wd_data, outfile)
    
# End of file
