import numpy as np
from numpy.linalg import solve
from numpy import matmul

eye2   = np.eye(2, dtype=complex)
zeros2 = np.zeros(2, dtype=complex)
pi     = 3.14159


class Green():
    """Class to handle Green function, at a given energy E for a periodic device
        - The system is considered as LEFT + CENTER + RIGHT parts
        - Initialize the isolated Green function for the center device in the energy complex plane as `1 / E + i * eta` where `eta << 1`
        - Interactions are given by the hopping matrices `t00`(inside the supercell), `t`(inter cell left), and `td`(inter cell right)
        - Decimation or "dressing up" of the interacting Green function is calculated through iteratively use of the Dyson equation
    """

    def __init__( self, t00=eye2, t=eye2, td=eye2, energy=-2.0, onsite_list=zeros2, eta=0.01 ):
        """Initialize the isolated Green function

        Args:
            - t00 : np.array type, cell self-interation
            - t   : np.array type, CENTER-RIGHT interaction
            - td  : np.array type, CENTER-LEFT interaction
            - energy : float, 
            - onsite_list : np.array type, on-site energies of each site on the cell
            - eta : float, `<< 1`
        """
        self.t00       = t00
        self.t         = t
        self.td        = td
        self.size      = t00.shape[0]
        self.eta       = eta
        self.energy    = energy
        self.E         = energy - onsite_list # they must be numpy arrays
        invE           = 1 / ( self.E + complex(0, eta) )
        self.greenFunc = invE * np.eye(self.size, dtype=complex)
        ones           = np.ones(self.size)
        self.up_prev   = ones.copy()
        self.dw_prev   = ones.copy()
        self.up        = ones.copy()
        self.dw        = ones.copy()
        self.Fermi     = 1.0
        self.Fermi_prev= 1.0
    #
    
    def __repr__(self) -> str:
        """Provides information about the complex energy selected """
        return f"Green object with energy={round(self.energy, 3)}, eta={round(self.eta, 5)}"
        
    # @profile
    def renormalize(self, Z, Q):
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
    def decimate(self):
        """
        Applies the Dyson equation iteratively on the Green function. Uses the `renormalize()` function.
        It updates the Green function
        """
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
        self.greenFunc = GR.copy()
    #



    def get_density(self, consider_spin=False):
        """Calculates the density of states by calculating the trace of the Green function

        Args:
            consider_spin (bool, optional): Consider spin degree of freedom in calculations. Defaults to False.

        Returns:
            density: Density of states. If consider_spin is True, density_up and density_dw are returned
        """

        denominator = self.size * pi

        if not consider_spin == None:
            self.decimate() # Updates the Green function
            return -np.trace( self.greenFunc.imag ) / denominator
        else:
            self.solve_self_consistent() # Updates the Green function
            n     = self.size
            nHalf = n / 2
            density_up, density_dw = 0, 0
            for i in range(nHalf):
                density_up -= self.greenFunc[i].imag
                density_dw -= self.greenFunc[i + nHalf].imag
            #
            density_up /= denominator
            density_dw /= denominator
            return density_up, density_dw
        #
    #

    @staticmethod
    def get_density_OneLinearChain(energy):
        """Get the electronic density of linear chain of sites
        
        Args:
            - energy: float
        
        Returns:
            - density: float
        """
        eye1 = np.eye(1, dtype=complex)
        t00  = np.asarray( [0] )
        t    = eye1
        td   = np.transpose(t)
        #   
        onsite_list = np.zeros(1)
        eta   = 0.001
        g       = Green(t00, t, td, energy=energy, onsite_list=onsite_list, eta=eta)
        density = g.get_density()
        return density

    @staticmethod
    def get_density_smallestZGNR(energy):
        """Get the electronic density of a 2-ZGNR
        
        Args:
            - energy: float
        
        Returns:
            - density: float
        """
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
        density = g.get_density()
        return density
    
    @staticmethod
    def plot(energy_list, density_list):
        """Script to display the electronic density of a system
        
        Args:
            - energy_list: numpy array, 1-dimensional
            - density_list: numpy array, 1-dimensional, same size as energy_list

        Returns: 
            - None
        """
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
        plt.title("Electronic density of states")
        plt.xlabel("Energy (hopping units)")
        plt.ylabel("Density of states (a.u.)")
        # plt.savefig("DOS.pdf", format="pdf", bbox_inches="tight")
        plt.show()
 
    def get_ansatz(self):
        """Get initial configuration for the occupation at each atom site
        
        It updates:
            self.up_prev (array float): occupations with spin upward
            self.dw_prev (array float): occupations with spin downward
        
        Returns:
            None
        """
        n       = self.size
        nMinus1 = n - 1

        self.up_prev[0],       self.dw_prev[nMinus1] -= 0.5
        self.up_prev[nMinus1], self.dw_prev[0]       += 0.5
    #
    
    def get_occupation(self):
        # self.up, self.dw = self.integrate()
        pass

    @staticmethod
    def is_under_error(list, list_prev, error):
        error_list = (list - list_prev) / list_prev # so, they must be numpy arrays
        return np.all(error_list < error)

    def check_convergence(self):
        error = 0.1
        up_under_error = self.is_under_error(self.up, self.up_prev, error)
        dw_under_error = self.is_under_error(self.dw, self.dw_prev, error)
        return up_under_error and dw_under_error

    @staticmethod
    def get_pondered_sum(list, list_prev):
        alpha = 0.5
        return (alpha * list) + ((1 - alpha) * list_prev)
    
    def update(self):
        """Update occupation and Fermi energy using a pondered sum from previous results
        """
        self.up    = self.get_pondered_sum(self.up, self.up_prev)
        self.dw    = self.get_pondered_sum(self.dw, self.dw_prev)
        self.Fermi = self.get_pondered_sum(self.Fermi, self.Fermi_prev)

    def solve_self_consistent(self):

        self.get_ansatz()
        for i in range(2):
            self.get_occupation()
            if self.check_convergence():
                self.decimate()
                self.update()    
            #
        #
    #