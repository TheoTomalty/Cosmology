import numpy as np
import math
import cmath
import Constants as const
import Wake as wake
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#TODO: Use better approximation for the spectrum
#TODO: Implement approximation to spin-weighted spherical harmonics

uK = const.uK

class Spectrum(object):
    def __init__(self, nodes=()):
        ''' Simple class to set up a peacewise liner spectrum in log-log space.
        
        # Define the spectrum
        >>> spec = Spectrum()
        
        # Add the nodes
        >>> spec.append(10, 3)
        >>> spec.append(100, 5)
        >>> spec.append(50, 4)
        >>> print spec.nodes
        [(10.0, 3.0), (50.0, 4.0), (100.0, 5.0)]
        
        # Evaluate Spectrum
        >>> eval = [spec.eval(x) for x in range(20, 80, 10)]
        >>> print eval
        [3.3957009783920302, 3.650944325203774, 3.8435950448841982, 4.0, 4.241804602764226, 4.457616594046425]
        '''
        self.nodes = []
        
        for node in nodes:
            self.append(*node)
        
        self.sort()
        
    def sort(self):
        # Sort by x_value
        self.nodes = sorted(self.nodes, key=lambda vertex: vertex[0])
    
    def append(self, x, y):
        # Add a point to the spectrum
        self.nodes.append((float(x), float(y)))
        self.sort()
        
    def eval(self, x):
        x = float(x)
        assert len(self.nodes) > 1, "Not enough nodes in spectrum"
        assert self.nodes[0][0] <= x < self.nodes[-1][0], "Out of bounds"
        
        for node in self.nodes:
            if x > node[0]:
                continue
            
            end = node
            start = self.nodes[self.nodes.index(node) - 1]
            
            # Linear in log-log between nodes
            alpha = (
                math.log(float(start[1]) / float(end[1])) /
                math.log(float(start[0]) / float(end[0]))
            )
            return start[1] * (x/start[0])**alpha

class CMB(object):
    def __init__(self, resolution):
        ''' Generate and save all the gaussian distributed polarization variables.
        
        :param resolution: Resolution of the intended image, to set max on the computed wavenumbers
        
        #>>> cmb = CMB(0.01)
        #>>> cmb.generate()
        '''
        self.l_max = int(const.pi/resolution)
        
        # Elements arranged in single array ordered first by l then by m
        self.E = np.zeros([10], dtype=float) #np.zeros([(self.l_max + 1)**2], dtype=float)
        self.B = np.zeros([10], dtype=float) #np.zeros([(self.l_max + 1)**2], dtype=float)

        self.C_EE = Spectrum(
            [(2, 0.2*uK), (10, 0.05*uK), (600, 4*uK), (1100, 4*uK), (3000, 1*uK), (6000, 0.05*uK)]
        )
        self.C_BB = Spectrum(
            [(2, 0.035*uK), (8, 0.02*uK), (800, 0.5*uK), (1500, 0.5*uK), (3000, 0.2*uK), (6000, 0.01*uK)]
        )
    
    def index(self, l, m):
        ''' Converts from the natural indices to the index used in the numpy array.
        
              : m = ...
         l = 0:        0
         l = 1:     -1 0 1
         l = 2:  -2 -1 0 1 2
        '''
        
        assert l >= 0 and abs(m) <= l, "Badly defined l,m indices"
        # index of middle of row: total elements in tree MINUS number elements after zero (incl. zero itself)
        zero_index = (l + 1)**2 - (l + 1)
        return zero_index + m
    
    def indices(self, index):
        ''' Converts from the numpy indices to the natural indices of E and B coefficients. '''
        raise NotImplementedError
    
    def generate(self):
        # Generate random numbers
        E_featureless = np.random.normal(0, 1, len(self.E))
        B_featureless = np.random.normal(0, 1, len(self.B))
        
        # Generate E and B coefficients from spectrum and random numbers
        for l in range(2, self.l_max + 1):
            for k in range(-l, l + 1):
                index = self.index(l, k)
                self.E[index] = E_featureless[index]*self.C_EE.eval(l)
                self.B[index] = B_featureless[index]*self.C_BB.eval(l)

class Frame(object):
    def __init__(self, phi, theta, size, num_pixels):
        ''' Define the frame for CMB and string visualization.
        
        >>> frame = Frame(const.pi/2, 0., 5*const.deg, 50)
        
        :param phi: Azimuthal coordinate of centre of frame
        :param theta: Zenith coordinate of centre of frame
        :param size: Angular width of the frame
        :param pixels: Number of pixels in the frame
        '''
        
        self.phi = phi
        self.theta = theta
        self.size = size
        self.num_pixels = num_pixels
        
        self.cmb = CMB(size/num_pixels)
        self.sigma = np.array(
            [[0 if k == l == 0 else self.cmb.C_EE.eval(const.pi*self.num_pixels/np.sqrt(k**2 + l**2)/self.size) for k in range(self.num_pixels)] for l in range(self.num_pixels)]
        )
        self.modes = np.multiply(
            np.reshape(
                np.sum(
                    [np.random.normal(0, 1, num_pixels**2),
                    np.multiply(1j, np.random.normal(0, 1, num_pixels**2))],
                    axis=0
                ),
                [num_pixels, num_pixels]
            ),
            self.sigma
        )
        self.pixels = np.fft.fft2(self.modes).real
        #self.pixels = np.array(
        #    [[self.pixel_val(i, j).real for j in range(self.num_pixels)] for i in range(self.num_pixels)]
        #)
        
    
    def pixel_pos(self, i, j):
        ''' Converts from pixel indices to angular coordinates

        :param i: Pixel index along height of image
        :param j: Pixel index along base of image
        :return: Angular coordinates
        '''
        pixel_width = self.size / self.num_pixels
        begin = np.array([
            self.phi - self.size/2 + pixel_width/2,
            self.theta - self.size/2 + pixel_width/2
        ])
        pixel = np.array([
            pixel_width, 
            pixel_width
        ])
        index = np.array([i, j])
        
        return np.sum([begin, np.multiply(pixel, index)], axis=0)
    
    def add_strings(self, num_strings):
        orientations = 2 * np.random.randint(2, size=num_strings) - 1
        phis = const.pi * np.random.random(num_strings)
        dists = (math.sqrt(2) + 1)/2 * self.size * (np.random.random(num_strings) - 0.5)
        
        
        for orientation, phi, dist in zip(orientations, phis, dists):
            string = wake.SimpleWake(phi, dist, 80*uK, orientation)
            
            for i in range(self.num_pixels):
                for j in range(self.num_pixels):
                    relative_pos = self.pixel_pos(i, j) - np.array([self.phi, self.theta])
                    rho = string.linear_coords(relative_pos[0], relative_pos[1])[1]
                    
                    self.pixels[i][j] += string.width(rho)
    
    #def pixel_val(self, i, j):
    #    sum = 0.
    #    for k in range(self.num_pixels):
    #        for l in range(self.num_pixels):
    #            sum += self.modes[l][k] * cmath.exp(1j*2*const.pi/self.num_pixels * (i*k + j*l))
    #    
    #    return sum / self.num_pixels**2
    
    def draw(self):
        plt.imshow(self.pixels,interpolation='none')
        print self.pixels
        plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
