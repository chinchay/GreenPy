# GreenPy

The package implements a class to handle different calculations to obtain the electronic density of states of a nanomaterial. The code is based on the iterative use of the Dyson equation through a process called "dressing up" or "decimation." It is assumed the system is periodic along the axial axis.

## Quick start

Clone the repo: `https://github.com/chinchay/GreenPy.git`

## What's included

Ready to use methods are available: `get_density_OneLinearChain()` and `get_density_smallestZGNR()` to calculate and display the electronic density of a linear chain of atoms and for the smallest zigzag graphene nanoribbon, respectively.


```python
import numpy as np
from GreenPy import Green
import matplotlib.pyplot as plt

energy_list  = np.arange(-3.15, 3.25, 0.05)
density_list = []
for energy in energy_list:
    dens = Green.get_density_smallestZGNR(energy)
    density_list.append( dens )
#
Green.plot(energy_list, density_list)
```

<img src="images/DOS.png" alt="drawing" width="450"/>


The `src` folder contains a performance comparison between Numpy and LAPACK functions. Unit test is also available.

The package allows to include electronic correlation. First, a self-consistent calculation is performed to find the occupation at each atom site:

```python
import numpy as np
from GreenPy import Green
import matplotlib.pyplot as plt
from GreenPy import library as lib

nAtoms = 4 # for a 2-ZGNR
t00, t, td, onsite_list = lib.get_ZGNR_interactions(nAtoms)
g = Green(t00=t00, t=t, td=td, onsite_list=onsite_list, consider_spin=True)

store_errors=True
g.find_occupations(store_errors=True)
plt.plot(g.hist_err, '.')
plt.show()
```

<img src="images/occupation_converge.png" alt="drawing" width="450"/>


Now, the DOS for electrons with spin up and down can be found after a decimation process, using the converged occupations stored in `g`:

```python
energy_list = np.arange(-3.15, 3.25, 0.05)
up_list, dw_list = [], []
for (i, energy) in enumerate(energy_list):
    up, dw = g.get_DOS(energy=energy)
    up_list.append(up)
    dw_list.append(dw)
#
g.plot(energy_list, up_list)
g.plot(energy_list, dw_list)
```

<img src="images/DOS_UP.png" alt="drawing" width="450"/>

DOS curves are degenerated, so results for spin-down electrons are the same.


## Under development

* A future release will connect the code to ASE objects and calculators
* Electric and magnetic field effects


## References
        
* [Rocha, C. __*Propriedades Físicas de Nanotubos de Carbono*__. Universidade Federal Fluminense, Brazil, 2005](http://oldsite.if.uff.br/index.php?option=com_content&view=article&id=348)
* E. Economou. __*Green’s Functions in Quantum Physics*__. Springer, 2006. 11, 12, 70
* M. Di Ventra. __*Electrical Transport in Nanoscale Systems*__, 1st edition, pp 150. Cambridge University Press, 2008. 73



## Copyright and license

Code released under the MIT License