import numpy as np
from numpy.linalg import solve

@profile
def renormalize(Z, Q):
    size = Q.shape[0]
    ident = np.eye(size, dtype=complex)
    temp = ident - Z
    renormalized = solve(temp, Q)
    return renormalized
#

def decimate(greenFunction, t00, t, td):
    # careful here, between .dot and matmul
    # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication#:~:text=matmul%20differs%20from%20dot%20in,if%20the%20matrices%20were%20elements.
    temp = np.matmul(greenFunction, t00)
    
    GR  = renormalize(temp, greenFunction)
    TR  = t
    TRD = td
    GR0 = 0.0
    
    n = 15
    for i in range(n):
        Z  = np.matmul(GR, TR)    # Z(N-1)   = GR(N-1)*TR(N-1)
        ZzD= np.matmul(GR, TRD)   # ZzD(N-1) = GR(N-1)*TRD(N-1)
        TR = np.matmul(np.matmul(TR, GR), TR)   #   TR(N) = TR(N-1)*GR(N-1)*TR(N-1)
        TRD= np.matmul(np.matmul(TRD, GR), TRD) # TRD(N) = TRD(N-1)*GR(N-1)*TRD(N-1)
        GR = renormalize( np.matmul(Z, ZzD) + np.matmul(ZzD,Z), GR)
    #
    return GR
#