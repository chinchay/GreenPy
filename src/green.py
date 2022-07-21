import numpy as np
import library as lib

eye2   = np.eye(2, dtype=complex)
zeros2 = np.zeros(2, dtype=complex)
pi     = 3.14159
invPi  = 1 / pi

class Green():
    """Class to handle Green function, at a given energy E for a periodic device
        - The system is considered as LEFT + CENTER + RIGHT parts
        - Initialize the isolated Green function for the center device in the energy complex plane as `1 / E + i * eta` where `eta << 1`
        - Interactions are given by the hopping matrices `t00`(inside the supercell), `t`(inter cell left), and `td`(inter cell right)
        - Decimation or "dressing up" of the interacting Green function is calculated through iteratively use of the Dyson equation
    """

    def __init__( self, t00=eye2, t=eye2, td=eye2, energy=-2.0, onsite_list=zeros2, eta=0.01, consider_spin=False, store_errors=False):
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
        self.size      = len(onsite_list)
        self.eta       = eta
        self.energy    = energy
        self.onsite_list = onsite_list
        self.E         = energy - self.onsite_list # they must be numpy arrays
        invE           = 1 / ( self.E + complex(0, eta) )
        self.greenFunc = invE * np.eye(self.size, dtype=complex)

        self.consider_spin = consider_spin
        self.ones      = np.ones(self.size)
        self.eye       = np.eye(self.size)
        self.up_prev   = self.ones.copy()
        self.dw_prev   = self.ones.copy()
        self.up        = self.ones.copy()
        self.dw        = self.ones.copy()
        self.Fermi     = 0.0
        self.Fermi_prev= 0.0

        if consider_spin:
            self.t00       = np.kron( np.eye(2), self.t00 )
            self.t         = np.kron( np.eye(2), self.t   )
            self.td        = np.kron( np.eye(2), self.td  )
            self.init_greenFunc_spin()
            self.store_errors = store_errors
            if self.store_errors:
                self.hist_err  = []
            #
        #
    #

    def init_greenFunc_spin(self):
        self.U         = 0.0
        dE_up          = (self.U * self.dw_prev) - 0.5
        dE_dw          = (self.U * self.up_prev) - 0.5
        invE_up        = 1 / ( self.E + complex(0, self.eta) + dE_up )
        invE_dw        = 1 / ( self.E + complex(0, self.eta) + dE_dw )
        g_up           = invE_up * self.eye
        g_dw           = invE_dw * self.eye

        matrix_up, matrix_dw = np.eye(2), np.eye(2)
        matrix_up[1, 1] = 0
        matrix_dw[0, 0] = 0
        G_up = np.kron( matrix_up, g_up )
        G_dw = np.kron( matrix_dw, g_dw )
        self.greenFunc = G_up + G_dw
    #



    
    def __repr__(self) -> str:
        """Provides information about the complex energy selected """
        return f"Green object with energy={round(self.energy, 3)}, eta={round(self.eta, 5)}"

    def update(self, g):
        self.greenFunc = g.copy()

    def get_total_density(self):
        """Calculates the density of states by calculating the trace of the Green function"""
        g_decimated = lib.decimate(self.greenFunc, self.t00, self.t, self.td)
        self.update(g_decimated)
        return -np.trace( g_decimated.imag ) / (self.size * pi)

    def get_density_per_spin(self):
        g_consistent = self.solve_self_consistent()
        self.update(g_consistent)

        n = self.size
        dens_up, dens_dw = 0, 0
        for i in range(n):
            dens_up -= self.greenFunc[i, i].imag
            dens_dw -= self.greenFunc[i + n, i + n].imag
        #
        denominator = self.size * pi
        dens_up /= denominator
        dens_dw /= denominator
        return dens_up, dens_dw
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
        density = g.get_total_density()
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
        density = g.get_total_density()
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
    #

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

        self.up_prev[0]       -= 0.5
        self.up_prev[nMinus1] += 0.5
        
        self.dw_prev[0]       += 0.5
        self.dw_prev[nMinus1] -= 0.5
    #

    def get_occupation(self):
        eps = 0.1
        ef  = self.Fermi
        dx = 0.2
        x_list = np.arange(0, 1.0, dx)
        len_x = len(x_list)
        nAtoms = self.size

        n2 = nAtoms * 2
        g2 = np.zeros( (len_x, n2), dtype=complex )
        E_ = self.Fermi - self.onsite_list  # they must be numpy arrays

        prod_list = np.ones(len_x)
        for (i, x) in enumerate(x_list):
            prod_list[i] = 1 / (1 - pow(x, 2))
        #
        
        dE_up          = (self.U * self.dw_prev) - 0.5
        dE_dw          = (self.U * self.up_prev) - 0.5
        for (i, x) in enumerate(x_list):
            invE_up  = 1 / (  E_ + complex(0, x) + dE_up )
            invE_dw  = 1 / (  E_ + complex(0, x) + dE_dw )
            g_up     = invE_up * self.eye
            g_dw     = invE_dw * self.eye
            
            matrix_up, matrix_dw = np.eye(2), np.eye(2)
            matrix_up[1, 1] = 0
            matrix_dw[0, 0] = 0
            G_up = np.kron( matrix_up, g_up )
            G_dw = np.kron( matrix_dw, g_dw )
            
            g_   = G_up + G_dw
            
            g_   = lib.decimate(g_, self.t00, self.t, self.td)
            gii  = g_.diagonal()
            g2[i, :] = gii[:] * prod_list[i]
        #
        
        for j in range(nAtoms):
            sum_up    = np.trapz( y=g2[:, j].real, x=None, dx=dx, axis=-1 )
            sum_dw    = np.trapz( y=g2[:, j + nAtoms].real, x=None, dx=dx, axis=-1 )
            self.up[j] = 0.5 + ( invPi * sum_up )
            self.dw[j] = 0.5 + ( invPi * sum_dw )
        #
    #

    @staticmethod
    def is_under_error(list, list_prev, error):
        error_list = (list - list_prev) / list_prev # so, they must be numpy arrays
        return np.all(error_list < error), max(error_list)

    def converged(self):
        error = 0.1
        up_under_error, max_err_up = self.is_under_error(self.up, self.up_prev, error)
        dw_under_error, max_err_dw = self.is_under_error(self.dw, self.dw_prev, error)
        if self.store_errors:
            self.hist_err.append( abs(max(max_err_up, max_err_dw)) )
        #
        return up_under_error and dw_under_error

    @staticmethod
    def get_pondered_sum(list, list_prev):
        alpha = 0.5
        return (alpha * list) + ((1 - alpha) * list_prev)

    def update_spin_info(self):
        """Update occupation and Fermi energy using a pondered sum from previous results
        """
        self.up_prev      = self.get_pondered_sum(self.up, self.up_prev)
        self.dw_prev      = self.get_pondered_sum(self.dw, self.dw_prev)
        self.Fermi_prev   = self.get_pondered_sum(self.Fermi, self.Fermi_prev)

    def solve_self_consistent(self):
        self.get_ansatz()
        for i in range(5):
            self.get_occupation() # update self.up, self.dw
            if self.converged():  # compare self.up, self.dw with self.up_prev and self.dw_prev
                self.update_spin_info() # update self.up_prev, self.dw_prev
            #
        #
        self.init_greenFunc_spin() # updates self.greenFunc with the new self.up_prev, self.dw_prev
        g = lib.decimate(self.greenFunc, self.t00, self.t, self.td)
        # print(np.round(g, 2))

        # if self.store_errors:
        #     print("g:", len(self.hist_err))
        # #
        return g
    #
#