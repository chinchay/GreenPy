import numpy as np

def get_energy_minus_onsite(energy, onsite_list):
    """Calculate the relative energy at each atomic site

    Args:
        energy (float): Energy in eV or hopping units
        onsite_list (np array): on-site energies at each atomic site

    Returns:
        e_minus_onsite: relative energies
    """
    e_minus_onsite = energy - onsite_list  # they must be numpy arrays
    return e_minus_onsite

# @profile
def get_isolated_Green_(energy_all_contributions, eta):
    """Introduce the relative energies at each atomic site into a diagonal matrix

    Args:
        energy_all_contributions (np array): relative energies at each atomic site
        eta (float: imaginary part of the energy

    Returns:
        greenFun (np array): diagonal matrix
    """
    invE     = 1 / (  energy_all_contributions + complex(0, eta ) )
    greenFun = np.diag( invE )
    return greenFun

def get_t00_2ZGNR():
    """Build the hopping matrix for a cell

    Returns:
        t00: intra-cell hopping matrix
    """
    t00 = [ [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
            ]
    return np.asarray(t00)

def get_t_2ZGNR():
    """Build the interaction of the central cell with neighbor

    Returns:
        t: inter-cell hopping matrix
    """
    t   = [ [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
    ]
    return np.asarray(t)

def get_eye_up():
    """
    Build the follwing matrix:
    [ 1 0
      0 0 ]

    Returns:
        eye-UP
    """
    eye_up = np.eye(2)
    eye_up[1, 1] = 0
    return eye_up

def get_eye_dw():
    """
    Build the follwing matrix:
    [ 0 0
      0 1 ]

    Returns:
        eye_dw
    """
    eye_dw = np.eye(2)
    eye_dw[0, 0] = 0
    return eye_dw

def get_grid_imag_axis():
    """To be used in the integration the occupation in the complex plane

    Returns:
        eta_list (np array) : grid along the imaginary axis
        fac_list (np array) : part of the integrand function
        dx (float) : step size of the grid

        See reference:
        Rocha, C. Propriedades Físicas de Nanotubos de Carbono. 2005
        Universidade Federal Fluminense, Niterói, 2005
        http://oldsite.if.uff.br/index.php?option=com_content&view=article&id=348

    """
    dx       = 0.05
    x_list   = np.arange(dx, 1.0, dx) # avoid zero, so avoid dividing by zero in invE = 1 / E_
    eta_list = x_list / ( 1 - x_list)
    fac_list = 1 / np.power( 1 - x_list, 2)
    return eta_list, fac_list, dx

def get_pondered_sum(list, list_prev):
    """Improve convergence by updating the `list` (occupations) with
    a pondered sum of previous results

    Args:
        list (np array): last values obtained
        list_prev (np array): previous values obtained

    Returns:
        pondered_sum, depending on the alpha value, set here to alpha=0.5
    """
    alpha = 0.5
    return (alpha * list) + ((1 - alpha) * list_prev)

def is_above_error(list, list_prev, error):
    """Calculate a percentual change over a list, and determine 
    if any element change is above a threshold

    Args:
        list (np array): last values obtained
        list_prev (np array): previous values obtained
        error (float): threshold

    Returns:
        bool
    """
    error_list = np.abs( (list - list_prev) / list_prev ) # so, they must be numpy arrays
    return np.any(error_list > error), max(error_list)

def get_half_traces(matrix, n):
    """Calculate the trace of half a matrix

    Args:
        matrix (np array): used for green function matrices
        n (int): matrix size

    Returns:
        trace_up: upper half trace of the matrix
        trace_dw: down  half trace of the matrix
    """
    diag     = matrix.diagonal()
    trace_up  = sum( diag[:n] )
    trace_dw  = sum( diag[n:] )
    return trace_up, trace_dw

def get_ZGNR_interactions(nAtoms):
    """Returns the hopping interactions t00, t, td for a ZGNR with
    number of atoms as nAtoms

    Args:
        nAtoms (int): Number of atoms in the ZGNR. Multiple of 4 (each cell has 4 atoms)

    Returns:
        t00, t, td: interaction matrices
        onsite_list: on-site energies, zeros.
    """
    try:
        assert nAtoms % 4 == 0
    except:
        print("number of atoms should be multiple of 4, since each cell has 4 atoms")        
    #
    ones = np.asarray(np.ones(nAtoms - 1))
    t00  = np.diag(ones, k=1) + np.diag(ones, k=-1)
    
    t_onecell = [   [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                ]
    n_cells = nAtoms // 4
    t    = np.kron( np.eye(n_cells), t_onecell)
    td   = np.transpose(t)
    onsite_list = np.zeros(nAtoms)
    return t00, t, td, onsite_list
