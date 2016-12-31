import numpy as np
import Constants as const
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Square dimensions
theta_max = 10*const.deg
phi_max = 10*const.deg

pixels = np.zeros([100, 100], dtype=float)

class CMB(object):
    def __init__(self, resolution):
        ''' Generate and save all the gaussian distributed polarization variables.
        
        :param resolution: Resolution of the intended image, to set max on the computed wavenumbers
        
        >>> cmb = CMB(4)
        '''
        self.n_max = int(const.pi/resolution)
        
        # Elements arranged in single array ordered first by n then by k
        self.E_nk = np.zeros([(self.n_max + 1)^2], dtype=float)
        self.B_nk = np.zeros([(self.n_max + 1)^2], dtype=float)
