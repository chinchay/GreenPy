from ctypes import util
from email import utils
import unittest
import numpy as np
from src import utils

class TestHamiltonianClass(unittest.TestCase):
    def setUp(self) -> None:
        energy   = -1.0
        delta    = 0.01
        invE     = 1 / complex(energy, delta)
        ident    =  np.eye(2, dtype=complex)
        self.g   = invE * ident.copy()
        self.t00 = ident.copy()
        self.t   = ident.copy()
        self.td  = ident.copy()

        self.g_decimated = utils.decimate(self.g, self.t00, self.t, self.td)
        # print( np.round( self.g_decimated, 3 ) )    
    
    
    def test_initialize(self):
        self.assertEqual( np.round( self.g_decimated, 3 )[0, 0].real, -3.531, "wrong" )
        self.assertEqual( np.round( self.g_decimated, 3 )[0, 0].imag, -3.54, "wrong" )
        

#
    
if __name__ == "__main__":
    unittest.main()
#