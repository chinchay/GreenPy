import numpy as np
from numpy.linalg import solve
import time
import matplotlib.pyplot as plt
#%%

def renormalize(Z, Q, size):
    size = Q.shape[0]
    ident = np.eye(size, dtype=complex)
    temp = ident - Z
    renormalized = solve(temp, Q)
    return renormalized
#

nTimes = 10 #20
size_list = [ 500 ]
time_list = []
for size in size_list:
    print(size)
    eye    = np.eye(size, dtype=complex)
    energy = -2.0
    delta  = 0.01
    invE   = 1 / complex(energy, delta)
    g       = invE * eye
    t00    = eye.copy()
    t      = eye.copy()
    td     = eye.copy()

    r_solve = renormalize(g, g, size) # just a toy example
    
    start_time = time.time()
    for _ in range(nTimes):
        r_solve = renormalize(g, g, size) # just a toy example
    #
    end_time = time.time()
    delta = (end_time - start_time) / nTimes
    time_list.append( delta )
#

#%%

size_list     = [ 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000 ]
times_python  = [ 0.0033731937408447267, 0.0002031564712524414, 8.555650711059571e-05, 0.006784498691558838, 0.004736852645874023, 0.010187649726867675, 0.0276763916015625, 0.059314048290252684, 0.061353445053100586, 0.10828009843826295, 0.11283899545669555, 0.2726146101951599, 0.23491215705871582, 0.3515087962150574, 1.9757980108261108, 16.247647047042847]
times_fortran = [ 3.3, 6.7, 20.0, 110.0, 273.3, 776.7, 1000.8, 3000.1, 6000.0, 8000.9, 12000, 18000, 25000, 34000, 176900, 1881400 ]

times_python   = np.asarray(times_python)
times_fortran  = np.asarray(times_fortran)
times_fortran /= pow(10, 6) # convert from microseconds to seconds

plt.loglog(size_list, times_python, '.')
plt.loglog(size_list, times_fortran, '+')
plt.title("Performance evolution Python vs FORTRAN")
plt.xlabel("matrix size")
plt.ylabel("time (seconds)")
plt.show()
