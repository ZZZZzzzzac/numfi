import unittest
import numpy as np
from numfi.numfi_copy import numfi,numqi,_is_floating

class numTest(unittest.TestCase):
    def assertArrayEqual(self, first, second, msg=None):
        self.assertTrue(np.all(np.asarray(first) == np.asarray(second)), msg=msg)
    
    def assertArrayAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        atol = 10**(-places) if delta is None else delta
        self.assertTrue(np.allclose(np.asarray(first), np.asarray(second), rtol=0, atol=atol), msg=msg)
    
    def assertProperty(self, x, property_dict):        
        for p_name,p_value in property_dict.items():
            p_attr = getattr(x, p_name)
            if _is_floating(p_value):
                self.assertArrayAlmostEqual(p_attr, p_value, msg=f"{p_name}: {p_attr} != {p_value}")
            else:
                self.assertArrayEqual(p_attr, p_value, msg=f"{p_name}: {p_attr} != {p_value}")

    def test_constructor(self):        
        x = 3.141592653589793   
        m_x = {
            "Signed" : 1,
            "WordLength" : 16,
            "FractionLength" : 15,
            "RoundingMethod" : "Nearest",
            "OverflowAction" : "Saturate",
            "Slope" : 2**(-15),
            "Bias" : 0
        }  

        #construct empty
        f_x = numfi([])
        q_x = numqi([])
        self.assertProperty(f_x, m_x)
        self.assertProperty(q_x, m_x)
        f_x = numfi()
        q_x = numqi()
        self.assertProperty(f_x, m_x)
        self.assertProperty(q_x, m_x)

        #construct by view casting
        y = np.array([x,0,-x])
        f_y = y.view(numfi)
        q_y = y.view(numqi)
        self.assertProperty(f_y, m_x)
        self.assertProperty(q_y, m_x)

        m_x["FractionLength"] = 13 # due to auto precision
        m_x["Slope"] = 1.220703125000000e-04
        # construct scalar
        # naming convention:    # m_ for matlab's fi()
        f_x = numfi(x)          # f_ for numfi
        q_x = numqi(x)          # q_ for numqi
        self.assertProperty(f_x, m_x)
        self.assertProperty(q_x, m_x)
        
        # construct list 
        f_x = numfi([x,x])
        q_x = numqi([x,x])
        self.assertProperty(f_x, m_x)
        self.assertProperty(q_x, m_x)

        #construct numpy array
        f_x = numfi(np.asarray([0,x]))
        q_x = numqi(np.asarray([0,x]))
        self.assertProperty(f_x, m_x)
        self.assertProperty(q_x, m_x)



        #construct by slicing
        f_z = f_x[1:]
        q_z = q_x[1:]
        self.assertProperty(f_z, m_x)
        self.assertProperty(q_z, m_x)

        #construct by indexing
        f_w = f_x[0]
        q_w = q_x[0]
        self.assertProperty(f_w, m_x)
        self.assertProperty(q_w, m_x)


if __name__ == "__main__":
    unittest.main()



