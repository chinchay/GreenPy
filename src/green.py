import numpy as np
from . import library as lib
from numpy.linalg import solve
from numpy import matmul

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

    def __init__(self, t00=eye2, t=eye2, td=eye2, onsite_list=zeros2, eta=0.01, consider_spin=False):
        """Initialize the isolated Green function

        Args:
            - t00 : np.array type, cell self-interation
            - t   : np.array type, CENTER-RIGHT interaction
            - td  : np.array type, CENTER-LEFT interaction
            - onsite_list : np.array type, on-site energies of each site on the cell
            - eta : float, `<< 1`
            - consider_spin: bool type, for spin-degree of freedom
        """
        self.U         = 0
        self.t00       = t00
        self.t         = t
        self.td        = td
        self.size      = len(onsite_list)
        self.eta       = eta        
        self.onsite_list = onsite_list

        zeros          = np.zeros(self.size)
        self.initialize_occupations(zeros)
        self.eye       = np.eye(self.size, dtype=complex)

        self.consider_spin = consider_spin
        if consider_spin:
            self.U         = -1.0
            self.t00       = np.kron( np.eye(2), self.t00 )
            self.t         = np.kron( np.eye(2), self.t   )
            self.td        = np.kron( np.eye(2), self.td  )

            self.Fermi     = 0.0
            self.Fermi_prev= 0.0
            
            self.eta_list, self.fac_list, self.dx = lib.get_grid_imag_axis()

            n2 = self.size * 2
            self.greenFunc = np.zeros( (n2, n2), dtype=complex )

            halfs          = np.ones(self.size) / 2
            self.initialize_occupations(halfs)
            
            # self.zeros     = np.zeros( (n2, n2), dtype=complex)
            self.eye       = np.eye(n2, dtype=complex)
        #
    #

    def initialize_occupations(self, arr):
        """Copy `arr` to self.up_prev, self.dw_pre, self.up, and self.dw

        Args:
            arr (np.array)
        """
        self.up_prev = arr.copy()
        self.dw_prev = arr.copy()
        self.up      = arr.copy()
        self.dw      = arr.copy()
    #

    def init_greenFunc(self, energy, store_errors):
        """Initializes variables dependent on `energy`

        Args:
            energy (float): Energy in hopping or eV units
            store_errors (bool): Allows to save convergence evolution
        """
        self.energy       = energy
        self.store_errors = store_errors
        self.hist_err     = [] if store_errors else None
        all_energies_up, all_energies_dw  = self.get_all_energy_contributions(energy)
        self.greenFunc = self.get_isolated_Green(all_energies_up, all_energies_dw, self.eta)
    #

    def get_all_energy_contributions(self, energy):
        """Consider on-site and electronic repulsion energies into each atom site

        Args:
            energy (float): Energy in hopping or eV units

        Returns:
            all_energies_up : relative energies for electrons with spin up
            all_energies_dw : None if consider_spin=False
        """
        e_minus_onsite   = lib.get_energy_minus_onsite(energy, self.onsite_list)
        hub_up, hub_dw   = self.get_Hubbard_terms(self.U, self.up_prev, self.dw_prev)
        all_energies_up  = e_minus_onsite + hub_up
        all_energies_dw  = e_minus_onsite + hub_dw
        return all_energies_up, all_energies_dw
    #

    def __repr__(self) -> str:
        """Provides information about the complex energy selected """
        return f"Green object with energy={round(self.energy, 3)}, eta={round(self.eta, 5)}"
    #

    def update(self, g):
        """Updates self.greenFunc with g

        Args:
            g (numpy array)
        """
        self.greenFunc = g.copy()
    #

    def get_dens(self):
        """Calculate density of states (DOS) by 
        getting the trace of the Green Function matrix

        Returns:
            dens_up: DOS at each atom site for electrons with spin up
            dens_dw: None if consider_spin=False
        """
        denominator = self.size * pi
        if self.consider_spin:
            trace_up, trace_dw = lib.get_half_traces(self.greenFunc.imag, self.size)
            dens_up = -trace_up / denominator
            dens_dw = -trace_dw / denominator
        else:
            dens_up = -np.trace( self.greenFunc.imag ) / denominator
            dens_dw = None
        #
        return dens_up, dens_dw
    #

    def get_DOS(self, energy, store_errors=False):
        """Calculates the density of states by calculating the trace of the Green function"""
        self.init_greenFunc(energy, store_errors)
        g_decimated = self.decimate(self.greenFunc, self.t00, self.t, self.td)
        self.update(g_decimated)
        return self.get_dens()
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
        g       = Green(t00, t, td, onsite_list=onsite_list, eta=eta)
        density, _ = g.get_DOS(energy=energy)
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
        g       = Green(t00, t, td, onsite_list=onsite_list, eta=eta)
        density, _ = g.get_DOS(energy=energy)
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
        """Calculate the electronic correlation using a mean-field approximation

        Args:
            U (float): Hubbard penalty, in eV or hopping units
            up (array float): occupations with spin upward
            dw (array float): occupations with spin downward

        Returns:
            ub_up: energy contribution for electronics with spin up
            ub_dw: energy contribution for electronics with spin dw
        """
        hub_up = U * (dw - 0.5)
        hub_dw = U * (up - 0.5)
        return hub_up, hub_dw

    # @profile
    def get_isolated_Green(self, all_energies_up, all_energies_dw, eta):
        """Build the isolated Green function for the system

        Args:
            all_energies_up (np array): relative energies at each atomic site for spin-up electrons
            all_energies_dw (np array): relative energies at each atomic site for spin-dw electrons
            eta (float): imaginary part of the energy in the complex plane (E = energy + i * eta)

        Returns:
            isolated_green_funtion (np array): complex diagonal matrix
        """
        if self.consider_spin:
            g_up = lib.get_isolated_Green_(all_energies_up, eta)
            g_dw = lib.get_isolated_Green_(all_energies_dw, eta)
            
            n = self.size
            # self.greenFunc[:, :] = self.zeros[:, :]
            self.greenFunc = np.zeros((self.size * 2, self.size * 2), dtype=complex)
            self.greenFunc[:n, :n] = g_up[:,:]
            self.greenFunc[n:, n:] = g_dw[:,:]
            return self.greenFunc
        else:
            return lib.get_isolated_Green_(all_energies_up, eta)
        #

    # @profile
    def get_integrand(self, all_energies_up, all_energies_dw, eta_list, fac_list):
        """The integral of the DOS along the energy axis gives the 
        occupation. To avoid singularities, the integral is performed
        on the complex plane. This function builds the integrand function.
        
        See reference:
        Rocha, C. Propriedades Físicas de Nanotubos de Carbono. 2005
        Universidade Federal Fluminense, Niterói, 2005
        http://oldsite.if.uff.br/index.php?option=com_content&view=article&id=348

        Args:
            all_energies_up (np array): relative energies for spin-up electrons
            all_energies_dw (np array): relative energies for spin-dw electrons
            eta_list (np array): values of the imaginary part
            fac_list (np array): grid on the imaginary axis

        Returns:
            integ (np array) : integrand function
        """
        # Assuming self.consider_spin == True
        len_x = len(eta_list)
        n2    = self.size * 2
        integ = np.zeros( (len_x, n2), dtype=complex )
        for (i, eta) in enumerate(eta_list):
            g_ = self.get_isolated_Green(all_energies_up, all_energies_dw, eta)
            
            g_   = self.decimate(g_, self.t00, self.t, self.td)
            gii  = g_.diagonal()
            integ[i, :] = gii[:] * fac_list[i]
        #
        return integ

    def integrate_complex_plane(self, integrand, dx):
        """Sum up the integrand function to obtain the density of states

        Args:
            integrand (np array): see get_integrand() function docstring
            dx (float): grid separation on the imaginary axis
        """
        nAtoms   = self.size
        for j in range(nAtoms):
            sum_up    = np.trapz( y=integrand[:, j].real, x=None, dx=dx, axis=-1 )
            sum_dw    = np.trapz( y=integrand[:, j + nAtoms].real, x=None, dx=dx, axis=-1 )
            self.up[j] = 0.5 + ( invPi * sum_up )
            self.dw[j] = 0.5 + ( invPi * sum_dw )
        #
    #

    # @profile
    def get_occupation(self):
        """updates the occupation arrays by getting the Hubbard contributions,
        and integrating the decimated Green function over the density of states
        """
        all_energies_up, all_energies_dw = self.get_all_energy_contributions(self.Fermi)        
        integrand = self.get_integrand(all_energies_up, all_energies_dw, self.eta_list, self.fac_list)
        self.integrate_complex_plane(integrand, self.dx) # updates self.up, self.dw
    #

    def unconverged(self):
        """Determines if occupations have converged after an iteration

        Returns:
            bool
        """
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

    # @profile
    def find_occupations(self, store_errors=True):
        """Performs a self-consistent process to obtain occupations

        Args:
            store_errors (bool, optional): Whether or not to save evolution convergence. Defaults to True.
        """
        self.get_ansatz()          # update self.up_prev, self.dw_prev
        energy = self.Fermi_prev
        self.init_greenFunc(energy, store_errors) # update self.greenFunc with the new self.up_prev, self.dw_prev
        count = 0
        # self.unconverged() compares self.up, self.dw with self.up_prev and self.dw_prev
        while self.unconverged() and (count < 15):
            count += 1
            if count != 1:
                self.update_spin_info() # update self.up_prev, self.dw_prev using pondered sum
            #
            self.get_occupation()   # update self.up, self.dw
        #
        # occupations are already updated in self.up and self.dw
#

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
        # size = Q.shape[0]
        # ident = np.eye(size, dtype=complex)
        # ident = eye8
        temp = self.eye - Z
        renormalized = solve(temp, Q)
        return renormalized
    #

    # @profile
    def decimate(self, isolated_greenFunc, t00, t, td, iterations=10):
        """Applies the Dyson equation iteratively

        Args:
            - None

        Returns:
            - GR, numpy array, the renormalized Green function throug the use of the `renormalize()` function
        """ 
        # careful here, between .dot and matmul
        # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication#:~:text=matmul%20differs%20from%20dot%20in,if%20the%20matrices%20were%20elements.
        temp = matmul( isolated_greenFunc, t00 )

        GR  = self.renormalize(temp, isolated_greenFunc)
        TR  = t
        TRD = td
        
        # iterations = 10 #15
        for _ in range(iterations):
            Z   = matmul( GR, TR  )               # Z(N-1)   = GR(N-1)*TR(N-1)
            ZzD = matmul( GR, TRD )               # ZzD(N-1) = GR(N-1)*TRD(N-1)
            TR  = matmul( matmul(TR, GR), TR )    # TR(N)    = TR(N-1)*GR(N-1)*TR(N-1)
            TRD = matmul( matmul(TRD, GR), TRD )  # TRD(N)   = TRD(N-1)*GR(N-1)*TRD(N-1)
            GR  = self.renormalize( matmul(Z, ZzD) + matmul(ZzD,Z), GR )
        #
        return GR
    #