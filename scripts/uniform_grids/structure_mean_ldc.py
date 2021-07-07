import numpy as np
import numpy.linalg as la


def ldc_sol(data, N, i, j):
    
    U = np.zeros(2)
    
    if 0 < i < N and 0 < j < N:
        U = data[i,j,:]
    
    elif j > N-1: # top boundary
        U[0] = 1
        U[1] = 0
        
    return U


def structure_mean_point_fast(data, i, j, stencil, p):
    # Approximates mean field structure function for a single point in the spatial domain
    # Computes only the outer shell of the stencil
    N = data.shape[0]
    Smeanpt = 0
    
    istart = i - stencil
    iend = i + stencil
    
    jstart = j - stencil
    jend = j + stencil
    
    Uij = data[i, j, :]
    #print(Uij)
    for m in [istart, iend]:
        for n in range(jstart, jend+1):
            # print(m,n)
            Umn = ldc_sol(data, N, m, n)
            #Umn = data[m,n,:]
            Smeanpt += (np.abs(Uij[0] - Umn[0]))**p
            Smeanpt += (np.abs(Uij[1] - Umn[1]))**p
    
    for m in range(istart+1, iend):
        for n in [jstart, jend]:
            # print(m,n)
            Umn = ldc_sol(data, N, m, n)
            #Umn = data[m,n,:]
            Smeanpt += (np.abs(Uij[0] - Umn[0]))**p
            Smeanpt += (np.abs(Uij[1] - Umn[1]))**p
    
    # print('\n\n')
    return Smeanpt
    
def structure_mean_fast(data, stencil, p):
    #
    # Approximates mean field structure function
    #
    N = data.shape[0]
    max_stencil = stencil[-1]
    """
    # compute outer shells of stencil
    Smean_raw = np.zeros(stencil.size)
    for i in range(max_stencil, N-max_stencil):
        for j in range(max_stencil, N-max_stencil):
            for (ns,s) in enumerate(stencil):
                Smean_raw[ns] += structure_mean_point_fast(data, i, j, s, p)
    
    # post-process
    Smean = np.zeros(stencil.size)
    for ns in range(0, stencil.size):
        factor = 1./(2*stencil[ns]+1)**2/(N-2*max_stencil)**2
        Smean[ns] = (np.sum(Smean_raw[0:ns+1])*factor)**(1./p)
    """
    # compute outer shells of stencil
    Smean_raw = np.zeros(stencil.size)
    for i in range(0,N):
        for j in range(0,N):
            for (ns,s) in enumerate(stencil):
                Smean_raw[ns] += structure_mean_point_fast(data, i, j, s, p)
    
    # post-process
    Smean = np.zeros(stencil.size)
    for ns in range(0, stencil.size):
        factor = 1./(2*stencil[ns]+1)**2/N**2
        Smean[ns] = (np.sum(Smean_raw[0:ns+1])*factor)**(1./p)    
    
    return Smean
    
