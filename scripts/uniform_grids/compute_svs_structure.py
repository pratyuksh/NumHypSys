import numpy as np
import json
import time

import structure_mean_svs as sm


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data(filename):
    j = load_json(filename)
    return np.array(j["vx"]), np.array(j["vy"])

def compute_structure_mean(lx, N, stencil, p):
    
    data = np.zeros([N, N, 2], dtype=np.float64)
    
    f = "../../output/uq_pincompNS_uq_svs/velocity_mean_lx"+str(lx)+".json"
    vx, vy = load_data(f)
    
    for i in range(0,N):
        for j in range(0,N):
            data[i, j, 0] = vx[i + j*N]
            data[i, j, 1] = vy[i + j*N]
    
    Smean = sm.structure_mean_fast(data, stencil, p)
    print("Structure mean:\n", Smean)
    
    return Smean


if __name__ == '__main__':
    
    lx_ = 7
    N_ = 4*2**lx_
    
    p = [1,2,3]
    
    hx = 1./N_
    #max_cutoff = 0.09375
    max_cutoff = 0.03125
    max_stencil = int(max_cutoff/hx)
    stencil = np.arange(1,max_stencil+1)
    offsets = np.asarray(stencil)*hx
    print(offsets)
    
    smean = np.zeros((len(p), len(stencil)))
    for (nq,q) in enumerate(p):
        start_time = time.time()
        smean[nq,:] = compute_structure_mean(lx_, N_, stencil, q)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed time: ', elapsed_time)
        print('\n')
        
    sm_data = {}
    sm_data['offsets'] =  list(offsets)
    sm_data['p1'] = list(smean[0,:])
    sm_data['p2'] = list(smean[1,:])
    sm_data['p3'] = list(smean[2,:])
    
    # print(sm_data)
    filename = '../../output/uq_incompNS/structure_mean_svs_lx'+str(lx_)+'.json'
    with open(filename, 'w') as outfile:
        json.dump(sm_data, outfile)
    
# End of file
