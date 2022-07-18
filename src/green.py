import numpy as np
from numpy.linalg import solve
from numpy import matmul
# import fortranLib

eye2   = np.eye(2, dtype=complex)
zeros2 = np.zeros(2, dtype=complex)

class Green():
    def __init__( self, t00=eye2, t=eye2, td=eye2, energy=-2.0, onsite_list=zeros2, eta=0.01 ):
        # initialization
        self.t00       = t00
        self.t         = t
        self.td        = td
        self.size      = t00.shape[0]
        self.eta       = eta
        self.energy    = energy
        self.E         = energy - onsite_list # they must be numpy arrays
        invE           = 1 / ( self.E + complex(0, eta) )
        self.greenFunc = invE * np.eye(self.size, dtype=complex)
    #
    
    def __repr__(self) -> str:
        return f"Green object with energy={round(self.energy, 3)}, eta={round(self.eta, 5)}"
        
    # @profile
    def renormalize(self, Z, Q):
        size = Q.shape[0]
        ident = np.eye(size, dtype=complex)
        temp = ident - Z
        renormalized = solve(temp, Q)
        # renormalized = myflib.solve(temp, Q, size)
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

    def trace(self):
        return -np.trace( self.decimate().imag ) / (self.size * 3.14159)

    @staticmethod
    def get_density_OneLinearChain(energy):
        eye1 = np.eye(1, dtype=complex)
        t00  = np.asarray( [0] )
        t    = eye1
        td   = np.transpose(t)
        #   
        onsite_list = np.zeros(1)
        eta   = 0.001
        g       = Green(t00, t, td, energy=energy, onsite_list=onsite_list, eta=eta)
        density = g.trace()
        return density

    @staticmethod
    def get_density_smallestZGNR(energy):
        eye4 = np.eye(4, dtype=complex)
        t00 = [ [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0]
            ]
        t00 = np.asarray(t00)

        t   = [ [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
        ]
        t   = np.asarray(t)

        td   = np.transpose(t)
        #
        onsite_list = np.zeros(4)
        eta   = 0.001
        g       = Green(t00, t, td, energy=energy, onsite_list=onsite_list, eta=eta)
        density = g.trace()
        return density
    
    @staticmethod
    def plot(energy_list, density_list):
        import matplotlib.pyplot as plt
        min_energy = min(energy_list)
        max_energy = max(energy_list)
        plt.plot(energy_list, density_list)
        plt.ylim((0, 1.0))
        # Fill under the curve
        # https://stackoverflow.com/questions/10046262/how-to-shade-region-under-the-curve-in-matplotlib
        plt.fill_between(
                x     = energy_list, 
                y1    = density_list, 
                where = (min_energy <= energy_list)
                        & (energy_list <= max_energy),
                color = "b",
                alpha = 0.2
            )
        plt.show()
 