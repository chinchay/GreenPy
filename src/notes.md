# Performance history

`Line_profiler` can analyze the time spent at each line. Information can be found [here](https://github.com/rkern/line_profiler#kernprof) and [here](https://towardsdatascience.com/how-to-assess-your-code-performance-in-python-346a17880c9f). It would be required to add the decorator `@profile` before the code definition. 

To install, type the following:

```ShellSession
$ pip install line_profiler
```

Now, `kernprof` command will be available. Type the following in the directory where the file resides (in this case, `cd src/`):

```ShellSession
$ kernprof -l utils.py
```

A file `*.lprof`will be created. To display results type

```ShellSession
$ python -m line_profiler utils.py.lprof
```

Results:
```
Timer unit: 1e-06 s

Total time: 0.024579 s
File: utils.py
Function: renormalize_inv at line 3

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    3                                           @profile
    4                                           def renormalize_inv(Z, Q):
    5        20         38.0      1.9      0.2      size = Q.shape[0]
    6        20        482.0     24.1      2.0      ident = np.eye(size, dtype=complex)
    7        20        461.0     23.1      1.9      temp = ident - Z
    8        20      20526.0   1026.3     83.5      inversa = np.linalg.inv( temp )
    9        20       3054.0    152.7     12.4      renormalized = np.matmul(inversa, Q)
    10        20         18.0      0.9      0.1      return renormalized

Total time: 0.0117 s
File: utils.py
Function: renormalize_solve at line 13

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    13                                           @profile
    14                                           def renormalize_solve(Z, Q):
    15        20         30.0      1.5      0.3      size = Q.shape[0]
    16        20        418.0     20.9      3.6      ident = np.eye(size, dtype=complex)
    17        20        347.0     17.4      3.0      temp = ident - Z
    18        20      10891.0    544.5     93.1      renormalized = np.linalg.solve(temp, Q)
    19        20         14.0      0.7      0.1      return renormalized
```


Here is the piece of code, after improving performance. Notice that now I use `solve()` instead of `linalg.solve()` to speed up the code further.

`utils.py`

```python
import numpy as np
from numpy.linalg import solve
from numpy import matmul

@profile
def renormalize(Z, Q):
    size = Q.shape[0]
    ident = np.eye(size, dtype=complex)
    temp = ident - Z
    renormalized = solve(temp, Q)
    return renormalized
#

size    = 20
energy  = -2.0
delta   = 0.01
invE    = 1 / complex(energy, delta)
ident   =  np.eye(size, dtype=complex)
g       = invE * ident.copy()
t00     = ident.copy()
t       = ident.copy()
td      = ident.copy()
r_solve = renormalize(g, g) # just a toy example
```




## `decimate()` profile

Still, `renormalize()` takes most of the time in the decimation process, about 10X of any other operation.

```
$ kernprof -l utils.py
Wrote profile results to utils.py.lprof
$ python -m line_profiler utils.py.lprof
Timer unit: 1e-06 s

Total time: 0.000633 s
File: utils.py
Function: decimate at line 14

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    14                                           @profile
    15                                           def decimate(greenFunction, t00, t, td):
    18         1         59.0     59.0      9.3      temp = matmul(greenFunction, t00)
    20         1        289.0    289.0     45.7      GR  = renormalize(temp, greenFunction)
    21         1          2.0      2.0      0.3      TR  = t
    22         1          1.0      1.0      0.2      TRD = td
    23
    24         1          1.0      1.0      0.2      n = 1 #15
    25         2          4.0      2.0      0.6      for i in range(n):
    26         1         23.0     23.0      3.6          Z   = matmul( GR, TR  )               # Z(N-1)   = GR(N-1)*TR(N-1)
    27         1         17.0     17.0      2.7          ZzD = matmul( GR, TRD )               # ZzD(N-1) = GR(N-1)*TRD(N-1)
    28         1         28.0     28.0      4.4          TR  = matmul( matmul(TR, GR), TR )    # TR(N)    = TR(N-1)*GR(N-1)*TR(N-1)
    29         1         18.0     18.0      2.8          TRD = matmul( matmul(TRD, GR), TRD )  # TRD(N)   = TRD(N-1)*GR(N-1)*TRD(N-1)
    30         1        190.0    190.0     30.0          GR  = renormalize( matmul(Z, ZzD) + matmul(ZzD,Z), GR )
    31                                               #
    32         1          1.0      1.0      0.2      return GR
```
