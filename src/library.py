import numpy as np
# from numpy.linalg import solve
# from numpy import matmul

# eye8 = np.eye( 8, dtype=complex)

# @profile
# def renormalize(Z, Q):
#     """Calculates the "dressed-up" Green function 
    
#     Specifically:
#     `GreenFunction_new = Inverse( Identity - Z ) * Q`
    
#     Args:
#         - Z : numpy array, interaction information
#         - Q : numpy array, Green function

#     Returns:
#         - renormalized: numpy array, the new Green function of the dressed-up system
#     """
#     size = Q.shape[0]
#     # ident = np.eye(size, dtype=complex)
#     ident = eye8
#     temp = ident - Z
#     renormalized = solve(temp, Q)
#     return renormalized
# #

# @profile
# def decimate(isolated_greenFunc, t00, t, td):
#     """Applies the Dyson equation iteratively

#     Args:
#         - None

#     Returns:
#         - GR, numpy array, the renormalized Green function throug the use of the `renormalize()` function
#     """ 
#     # careful here, between .dot and matmul
#     # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication#:~:text=matmul%20differs%20from%20dot%20in,if%20the%20matrices%20were%20elements.
#     temp = matmul( isolated_greenFunc, t00 )

#     GR  = renormalize(temp, isolated_greenFunc)
#     TR  = t
#     TRD = td
    
#     iterations = 15
#     for _ in range(iterations):
#         Z   = matmul( GR, TR  )               # Z(N-1)   = GR(N-1)*TR(N-1)
#         ZzD = matmul( GR, TRD )               # ZzD(N-1) = GR(N-1)*TRD(N-1)
#         TR  = matmul( matmul(TR, GR), TR )    # TR(N)    = TR(N-1)*GR(N-1)*TR(N-1)
#         TRD = matmul( matmul(TRD, GR), TRD )  # TRD(N)   = TRD(N-1)*GR(N-1)*TRD(N-1)
#         GR  = renormalize( matmul(Z, ZzD) + matmul(ZzD,Z), GR )
#     #
#     return GR
# #

def get_energy_minus_onsite(energy, onsite_list):
    e_minus_onsite = energy - onsite_list  # they must be numpy arrays
    return e_minus_onsite

# @profile
def get_isolated_Green_(energy_all_contributions, eta):
    invE     = 1 / (  energy_all_contributions + complex(0, eta ) )
    greenFun = np.diag( invE )
    return greenFun

def get_t00_2ZGNR():
    t00 = [ [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
            ]
    return np.asarray(t00)

def get_t_2ZGNR():
    t   = [ [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
    ]
    return np.asarray(t)

def get_eye_up():
    eye_up = np.eye(2)
    eye_up[1, 1] = 0
    return eye_up

def get_eye_dw():
    eye_dw = np.eye(2)
    eye_dw[0, 0] = 0
    return eye_dw

def get_grid_imag_axis():
    dx       = 0.05
    x_list   = np.arange(dx, 1.0, dx) # avoid zero, so avoid dividing by zero in invE = 1 / E_
    eta_list = x_list / ( 1 - x_list)
    fac_list = 1 / np.power( 1 - x_list, 2)
    return eta_list, fac_list, dx

def get_pondered_sum(list, list_prev):
    alpha = 0.5
    return (alpha * list) + ((1 - alpha) * list_prev)

def is_above_error(list, list_prev, error):
    error_list = np.abs( (list - list_prev) / list_prev ) # so, they must be numpy arrays
    return np.any(error_list > error), max(error_list)

def get_half_traces(matrix, n):
    diag     = matrix.diagonal()
    trace_up  = sum( diag[:n] )
    trace_dw  = sum( diag[n:] )
    return trace_up, trace_dw