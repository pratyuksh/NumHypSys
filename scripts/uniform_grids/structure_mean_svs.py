import numpy as np
import numpy.linalg as la


def periodic_sol(data, N, i, j):
    
    m = i
    n = j
    
    # x-periodic
    if i < 0:
        m = N+i
    elif i > N-1:
        m = i-N
        
    # y-periodic
    if j < 0:
        n = N+j
    elif j > N-1:
        n = j-N
        
    return data[m,n,:]


def structure_mean_point(data, i, j, stencil, p):
    #
    # Approximates mean field structure function for a single point in the spatial domain
    #
    N = data.shape[0]
    Smeanpt = 0
    
    istart = i - stencil
    iend = i + stencil
    
    jstart = j - stencil
    jend = j + stencil
    
    Uij = data[i, j, :]
    #print(Uij)
    for m in range(istart, iend+1):
        for n in range(jstart, jend+1):
            if m == i and n == j:
                continue
            Umn = periodic_sol(data, N, m, n)
            #Smeanpt += la.norm(Uij - Umn, p)**p
            Smeanpt += (np.abs(Uij[0] - Umn[0]))**p
            Smeanpt += (np.abs(Uij[1] - Umn[1]))**p
            
    # print('\n\n')
    return Smeanpt/(2*stencil+1)**2


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
            Umn = periodic_sol(data, N, m, n)
            Smeanpt += (np.abs(Uij[0] - Umn[0]))**p
            Smeanpt += (np.abs(Uij[1] - Umn[1]))**p
    
    for m in range(istart+1, iend):
        for n in [jstart, jend]:
            # print(m,n)
            Umn = periodic_sol(data, N, m, n)
            Smeanpt += (np.abs(Uij[0] - Umn[0]))**p
            Smeanpt += (np.abs(Uij[1] - Umn[1]))**p
    
    # print('\n\n')
    return Smeanpt

    
def structure_mean(data, stencil, p):
    """
    Approximates mean field structure function
    """
    N = data.shape[0]
    Smean = 0

    for i in range(0,N):
        for j in range(0,N):
            Smean += structure_mean_point(data, i, j, stencil, p)
            # print(i, j)
    
    return (Smean / N**2)**(1./p)
    
def structure_mean_fast(data, stencil, p):
    """
    Approximates mean field structure function
    """
    N = data.shape[0]
    
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
    
