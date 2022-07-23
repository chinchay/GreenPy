import numpy as np
from numpy.linalg import solve
from numpy import matmul

# @profile
def renormalize(Z, Q):
    """Calculates the "dressed-up" Green function 
    
    Specifically:
    `GreenFunction_new = Inverse( Identity - Z ) * Q`
    
    Args:
        - Z : numpy array, interaction information
        - Q : numpy array, Green function

    Returns:
        - renormalized: numpy array, the new Green function of the dressed-up system
    """
    size = Q.shape[0]
    ident = np.eye(size, dtype=complex)
    temp = ident - Z
    renormalized = solve(temp, Q)
    return renormalized
#

# @profile
def decimate(isolated_greenFunc, t00, t, td):
    """Applies the Dyson equation iteratively

    Args:
        - None

    Returns:
        - GR, numpy array, the renormalized Green function throug the use of the `renormalize()` function
    """ 
    # careful here, between .dot and matmul
    # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication#:~:text=matmul%20differs%20from%20dot%20in,if%20the%20matrices%20were%20elements.
    temp = matmul( isolated_greenFunc, t00 )

    GR  = renormalize(temp, isolated_greenFunc)
    TR  = t
    TRD = td
    
    iterations = 15
    for _ in range(iterations):
        Z   = matmul( GR, TR  )               # Z(N-1)   = GR(N-1)*TR(N-1)
        ZzD = matmul( GR, TRD )               # ZzD(N-1) = GR(N-1)*TRD(N-1)
        TR  = matmul( matmul(TR, GR), TR )    # TR(N)    = TR(N-1)*GR(N-1)*TR(N-1)
        TRD = matmul( matmul(TRD, GR), TRD )  # TRD(N)   = TRD(N-1)*GR(N-1)*TRD(N-1)
        GR  = renormalize( matmul(Z, ZzD) + matmul(ZzD,Z), GR )
    #
    return GR
#    

def get_isolated_Green_(e_minus_onsite, eta, contrib_list=0):
    invE     = 1 / (  e_minus_onsite + complex(0, eta) + contrib_list )
    greenFun = np.diag( invE )
    return greenFun

