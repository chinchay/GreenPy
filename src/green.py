import numpy as np
from numpy.linalg import solve
from numpy import matmul

class Green():
    def __init__(
        self,
        t00    = np.eye(2, dtype=complex),
        t      = np.eye(2, dtype=complex),
        td     = np.eye(2, dtype=complex),
        energy = -2.0,
        delta  = 0.01
        ):
        # initialization
        self.t00       = t00
        self.t         = t
        self.td        = td
        self.size      = t00.shape[0]
        self.energy    = energy
        self.delta     = delta
        invE           = 1 / complex(self.energy, self.delta)
        self.greenFunc = invE * np.eye(self.size, dtype=complex)
    #
    
    def __repr__(self) -> str:
        return f"Green object with energy={round(self.energy, 3)}, delta={round(self.delta, 5)}"
        
    # @profile
    def renormalize(self, Z, Q):
        size = Q.shape[0]
        ident = np.eye(size, dtype=complex)
        temp = ident - Z
        renormalized = solve(temp, Q)
        return renormalized
    #

    # @profile
    def decimate(self):        
        # careful here, between .dot and matmul
        # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication#:~:text=matmul%20differs%20from%20dot%20in,if%20the%20matrices%20were%20elements.
        temp = matmul( self.greenFunc, self.t00 )
        
        GR  = self.renormalize(temp, self.greenFunc)
        TR  = self.t
        TRD = self.td
        
        iterations = 15
        for _ in range(iterations):
            Z   = matmul( GR, TR  )               # Z(N-1)   = GR(N-1)*TR(N-1)
            ZzD = matmul( GR, TRD )               # ZzD(N-1) = GR(N-1)*TRD(N-1)
            TR  = matmul( matmul(TR, GR), TR )    # TR(N)    = TR(N-1)*GR(N-1)*TR(N-1)
            TRD = matmul( matmul(TRD, GR), TRD )  # TRD(N)   = TRD(N-1)*GR(N-1)*TRD(N-1)
            GR  = self.renormalize( matmul(Z, ZzD) + matmul(ZzD,Z), GR )
        #
        return GR
    #