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

        e_minus_onsite = lib.get_energy_minus_onsite(self.energy, self.onsite_list)
        self.greenFunc = lib.get_isolated_Green_(e_minus_onsite, self.eta)

        self.consider_spin = consider_spin
        self.ones      = np.ones(self.size)
        self.eye       = np.eye(self.size)
        self.up_prev   = self.ones.copy() / 2
        self.dw_prev   = self.ones.copy() / 2
        self.up        = self.ones.copy() / 2
        self.dw        = self.ones.copy() / 2
        self.Fermi     = 0.0
        self.Fermi_prev= 0.0

        if consider_spin:
            self.t00       = np.kron( np.eye(2), self.t00 )
            self.t         = np.kron( np.eye(2), self.t   )
            self.td        = np.kron( np.eye(2), self.td  )
            # self.init_greenFunc_spin()
            self.store_errors = store_errors
            if self.store_errors:
                self.hist_err  = []
            #
            self.eta_list, self.fac_list, self.dx = lib.get_grid_imag_axis()
        #
    #

    def init_greenFunc_spin(self):
        self.U         = -1.0
        hub_up, hub_dw = self.get_Hubbard_terms(self.U, self.up_prev, self.dw_prev)
        e_minus_onsite = lib.get_energy_minus_onsite(self.energy, self.onsite_list)
        self.greenFunc = self.get_isolated_Green(e_minus_onsite, self.eta, hub_up, hub_dw)
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
        g_decimated = lib.decimate(self.greenFunc, self.t00, self.t, self.td)
        self.update(g_decimated)
        dens_up, dens_dw = lib.get_half_traces(g_decimated, self.size)

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
        t00  = np.asarray( [0] )
        t    = np.eye(1, dtype=complex)
        td   = np.transpose(t)
        #   
        onsite_list = np.zeros(1)
        eta     = 0.001
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
        t00  = lib.get_t00_2ZGNR()
        t    = lib.get_t_2ZGNR()
        td   = np.transpose(t)
        #
        onsite_list = np.zeros( t00.shape[0] )
        eta     = 0.001
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

        self.up_prev[0]       -= 0.2
        self.up_prev[nMinus1] += 0.2
        
        self.dw_prev[0]       += 0.2
        self.dw_prev[nMinus1] -= 0.2
    #

    @staticmethod
    def get_Hubbard_terms(U, up, dw):
        hub_up = U * (dw - 0.5)
        hub_dw = U * (up - 0.5)
        return hub_up, hub_dw

    @staticmethod
    def get_isolated_Green(e_minus_onsite, eta, hubbard_up, hubbard_dw):
        g_up = lib.get_isolated_Green_(e_minus_onsite, eta, hubbard_up)
        g_dw = lib.get_isolated_Green_(e_minus_onsite, eta, hubbard_dw)
        
        G_up = np.kron( lib.get_eye_up(), g_up )
        G_dw = np.kron( lib.get_eye_dw(), g_dw )
        
        g_   = G_up + G_dw
        return g_

    def get_g2(self, e_minus_onsite, eta_list, hub_up, hub_dw, fac_list):
        len_x = len(eta_list)
        n2    = len(e_minus_onsite) * 2
        g2    = np.zeros( (len_x, n2), dtype=complex )
        for (i, eta) in enumerate(eta_list):
            g_ = self.get_isolated_Green(e_minus_onsite, eta, hub_up, hub_dw)
            
            g_   = lib.decimate(g_, self.t00, self.t, self.td)
            gii  = g_.diagonal()
            g2[i, :] = gii[:] * fac_list[i]
        #
        return g2        

    def integrate_complex_plane(self, g2, dx):
        nAtoms   = self.size
        for j in range(nAtoms):
            sum_up    = np.trapz( y=g2[:, j].real, x=None, dx=dx, axis=-1 )
            sum_dw    = np.trapz( y=g2[:, j + nAtoms].real, x=None, dx=dx, axis=-1 )
            self.up[j] = 0.5 + ( invPi * sum_up )
            self.dw[j] = 0.5 + ( invPi * sum_dw )
        #
    #

    def get_occupation(self):
        hub_up, hub_dw = self.get_Hubbard_terms(self.U, self.up_prev, self.dw_prev)
        e_minus_onsite = lib.get_energy_minus_onsite(self.Fermi, self.onsite_list)    
        g2 = self.get_g2(e_minus_onsite, self.eta_list, hub_up, hub_dw, self.fac_list)
        self.integrate_complex_plane(g2, self.dx) # updates self.up, self.dw
    #

    def unconverged(self):
        error = 0.1
        up_unconverged, max_err_up = lib.is_above_error(self.up, self.up_prev, error)
        dw_unconverged, max_err_dw = lib.is_above_error(self.dw, self.dw_prev, error)
        if self.store_errors:
            self.hist_err.append( abs(max(max_err_up, max_err_dw)) )
        #
        return up_unconverged or dw_unconverged

    def update_spin_info(self):
        """Update occupation and Fermi energy using a pondered sum from previous results
        """
        self.up_prev      = lib.get_pondered_sum(self.up, self.up_prev)
        self.dw_prev      = lib.get_pondered_sum(self.dw, self.dw_prev)
        self.Fermi_prev   = lib.get_pondered_sum(self.Fermi, self.Fermi_prev)

    def solve_self_consistent(self):
        self.get_ansatz()          # update self.up_prev, self.dw_prev
        self.init_greenFunc_spin() # update self.greenFunc with the new self.up_prev, self.dw_prev
        self.get_occupation()      # update self.up, self.dw
        for i in range(15):
            if self.unconverged():    # compare self.up, self.dw with self.up_prev and self.dw_prev
                self.update_spin_info() # update self.up_prev, self.dw_prev using pondered sum
                self.get_occupation()   # update self.up, self.dw
            else:
                break
        #
        
        self.init_greenFunc_spin() # update self.greenFunc with the new self.up_prev, self.dw_prev
        g = lib.decimate(self.greenFunc, self.t00, self.t, self.td)
        return g
    #