import unittest
import numpy as np
from src.green import Green

class TestHamiltonianClass(unittest.TestCase):
    def setUp(self) -> None:
        eye1 = np.eye(1, dtype=complex)
        t00  = np.asarray( [0] )
        t    = eye1
        td   = np.transpose(t)
        #   
        onsite_list = np.zeros(1)
        energy = -1.0
        eta    = 0.001
        g      = Green(t00, t, td, energy=energy, onsite_list=onsite_list, eta=eta)
        self.decimated = g.decimate()
    #
    
    def test_initialize(self):
        # self.assertEqual( np.allclose( self.g_decimated.real, self.a_real), True, "wrong" )
        # self.assertEqual( np.allclose( self.g_decimated.imag, self.a_imag), True, "wrong" )
        self.assertAlmostEqual( self.decimated[0][0].real, -0.0001924499400464738, None, "wrong", delta=0.0001)
        self.assertAlmostEqual( self.decimated[0][0].imag, -0.5773500767396716, None, "wrong", delta=0.0001)

        # get_density_twoLines()

#
    
if __name__ == "__main__":
    unittest.main()
#