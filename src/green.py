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

    def __init__( self, t00=eye2, t=eye2, td=eye2, energy=-2.0, onsite_list=zeros2, eta=0.01, consider_spin=False ):
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
        self.E         = energy - onsite_list # they must be numpy arrays
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
            self.U         = 1.0
            dE_up          = (self.U * self.dw) - 0.5
            dE_dw          = (self.U * self.up) - 0.5
            invE_up        = 1 / ( self.E + complex(0, eta) + dE_up )
            invE_dw        = 1 / ( self.E + complex(0, eta) + dE_dw )
            g_up           = invE_up * self.eye
            g_dw           = invE_dw * self.eye

            self.t00       = np.kron( np.eye(2), t00 )
            self.t         = np.kron( np.eye(2), t   )
            self.td        = np.kron( np.eye(2), td  )

            matrix_up, matrix_dw = self.eye.copy(), self.eye.copy()
            matrix_up[1, 1] = 0
            matrix_dw[0, 0] = 0
            G_up = np.kron( matrix_up, g_up )
            G_dw = np.kron( matrix_dw, g_dw )
            self.greenFunc = G_up + G_dw
        #
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

        n_half = self.size / 2
        dens_up, dens_dw = 0, 0
        for i in range(n_half):
            dens_up -= self.greenFunc[i].imag
            dens_dw -= self.greenFunc[i + n_half].imag
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
 