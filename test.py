import unittest
import numpy as np
from GreenPy import Green
from GreenPy import library as lib

class TestGreenClass(unittest.TestCase):
    def setUp(self) -> None:
        eye1 = np.eye(1, dtype=complex)
        t00  = np.asarray( [0] )
        t    = eye1
        td   = np.transpose(t)
        #   
        onsite_list = np.zeros(1)
        energy = -1.0
        eta    = 0.001
        
        g = Green(t00=t00, t=t, td=td, onsite_list=onsite_list, eta=eta, consider_spin=False)
        g.init_greenFunc(energy=energy, store_errors=False)
        self.decimated = g.decimate(g.greenFunc, t00, t, td, iterations=15)
    #
    
    def test_initialize(self):
        # self.assertEqual( np.allclose( self.g_decimated.real, self.a_real), True, "wrong" )
        # self.assertEqual( np.allclose( self.g_decimated.imag, self.a_imag), True, "wrong" )
        self.assertAlmostEqual( self.decimated[0][0].real, -0.0001924499400464738, None, "wrong", delta=0.0001)
        self.assertAlmostEqual( self.decimated[0][0].imag, -0.5773500767396716, None, "wrong", delta=0.0001)

        energy = -2.65
        dens = Green.get_density_smallestZGNR(energy)
        self.assertAlmostEqual( dens, 0.001592691480946306, None, "error found when using statis method", delta=0.0001)

        nAtoms = 4
        t00, t, td, onsite_list = lib.get_ZGNR_interactions(nAtoms)
        g      = Green(t00=t00, t=t, td=td, onsite_list=onsite_list, consider_spin=True)
        self.assertEqual( sum(g.up_prev + g.dw_prev), 4.0, "wrong" )
        self.assertEqual( sum(g.up + g.dw), 4.0, "wrong" )



        # self.assertAlmostEqual( np.allclose(  )  )
        # # get_density_twoLines()

    def test_get_all_energy_contributions(self):
        nAtoms = 4
        t00, t, td, onsite_list = lib.get_ZGNR_interactions(nAtoms)
        g      = Green(t00=t00, t=t, td=td, onsite_list=onsite_list, consider_spin=True)
        energy = 0.0
        all_energies_up, all_energies_dw = g.get_all_energy_contributions(energy)
        self.assertEqual( sum(all_energies_up + all_energies_dw), 0.0, "wrong" )

#
    
if __name__ == "__main__":
    unittest.main()
#