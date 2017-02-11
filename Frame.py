from __future__ import division
import numpy as np
import math
import cmath
import Constants as const
import Wake as wake
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#TODO: Use better approximation for the spectrum
#TODO: Implement approximation to spin-weighted spherical harmonics
#TODO: Add instrumental noise

uK = const.uK

def square_complex(size):
    return np.array([[0 + 0j]*size]*size)

def square_mode(i, j, total):
    magnitude = max(i, j)
    return magnitude, 2*magnitude + 1

def angled_mode(i, j, total):
    radius = np.sqrt(i**2 + j**2)
    
    if radius < total:
        return radius, math.pi/2 * radius
    
    intersect_angle = np.arccos(total/radius)
    angular_span = 2*(math.pi/4 - intersect_angle)
    
    if angular_span > 0:
        return radius, angular_span * radius
    
    return radius, math.pi/2*radius

class Spectrum(object):
    def __init__(self, nodes=()):
        ''' Simple class to set up a piecewise liner spectrum in log-log space.
        
        # Define the spectrum
        >>> spec = Spectrum()
        
        # Add the nodes
        >>> spec.append(10, 3)
        >>> spec.append(100, 5)
        >>> spec.append(50, 4)
        
        # Evaluate Spectrum
        >>> eval = [spec.eval(x) for x in range(20, 80, 10)]
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
        
    def delta_T(self, l):
        ''' Evaluate the power per log interval of wave number $\Delta_T^2 = \frac{l(l+1)C_l}{2\pi}$
        
        :param l: Spherical harmonic number
        :return: Power of modes per log interval in wave number
        '''
        l = float(l)
        assert len(self.nodes) > 0, "No nodes in spectrum"
        
        if l < self.nodes[0][0]:
            return self.nodes[0][1]
        
        for node in self.nodes:
            if l > node[0]:
                continue
            
            end = node
            start = self.nodes[self.nodes.index(node) - 1]
            
            # Linear in log-log between nodes
            alpha = (
                math.log(float(start[1]) / float(end[1])) /
                math.log(float(start[0]) / float(end[0]))
            )
            return start[1] * (l/start[0])**alpha
        
        return self.nodes[-1][1]

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
        self.num_regions = num_pixels/15
        
        self.pixels = np.zeros([self.num_pixels, self.num_pixels])
        self.regions = np.zeros([self.num_regions, self.num_regions])
    
    @property
    def pixel_width(self):
        return self.size/self.num_pixels
    
    def pixel_index(self, x, y):
        j = int(round((x - self.phi + self.size/2 - self.pixel_width/2)/self.pixel_width))
        i = int(round((y - self.theta + self.size/2 - self.pixel_width/2)/self.pixel_width))
        
        return i, j
    
    def pixel_pos(self, i, j):
        ''' Converts from pixel indices to angular coordinates

        :param i: Pixel index along height of image
        :param j: Pixel index along base of image
        :return: Angular coordinates
        '''
        begin = np.array([
            self.phi - self.size/2 + self.pixel_width/2,
            self.theta - self.size/2 + self.pixel_width/2
        ])
        pixel = np.array([
            self.pixel_width, 
            self.pixel_width
        ])
        index = np.array([i, j])
        
        return np.sum([begin, np.multiply(pixel, index)], axis=0)
    
    def add_noise(self, spectrum):
        assert isinstance(spectrum, Spectrum)
        
        modes = square_complex(self.num_pixels)
        
        for i in range(self.num_pixels):
            for j in range(self.num_pixels):
                mode, num_modes = angled_mode(i, j, self.num_pixels)
                if mode == 0:
                    continue
                
                # Wavelength of the mode in units of spherical-coordinate radians (see numpy fft for reference)
                #wavelength = self.size / mode
                l = (2*math.pi * mode) / self.size
                dl = (2*math.pi * 0.5) / self.size
                
                # The amplitude of the i-j mode goes like power per unit solid angle
                d_log = np.log((l + dl)/(l - dl))
                power = spectrum.delta_T(l)**2 * d_log/ (4*math.pi * num_modes)
                
                # The power in a given mode is evenly distributed between the real and imaginary parts, hence 1/2 factor
                random_complex = np.random.randn() + np.random.randn()*1j
                modes[i][j] += np.sqrt(power/2) * random_complex
        
        self.pixels += np.fft.fft2(modes).real + np.fft.fft2(modes).imag
    
    def add_strings(self, num_strings, scale=1):
        if not num_strings:
            pass
        
        orientations = 2 * np.random.randint(2, size=num_strings) - 1
        angles = const.pi * np.random.random(num_strings)
        widths = np.random.normal(scale*0.1*uK, 0.03*uK, num_strings)
        phis = self.size * (np.random.random(num_strings) - 0.5) + self.phi
        thetas = self.size * (np.random.random(num_strings) - 0.5) + self.theta
        
        for angle, phi, theta, width, orientation in zip(angles, phis, thetas, widths, orientations):
            self.add_string(angle, phi, theta, width, orientation)
    
    def add_string(self, angle, phi, theta, width, orientation):
        string = wake.SimpleWake(angle, phi, theta, width, 2*const.deg, orientation)
        
        x, y, ranges = string.pixelation(self.pixel_width)
        i_begin, j_begin = self.pixel_index(x, y)
        
        for i, bounds in zip(range(1, len(ranges)+ 1), ranges):
            for j in range(bounds[0], bounds[1], 1):
                if 0 <= i+i_begin < self.num_pixels and 0 <= j+j_begin < self.num_pixels:
                    self.pixels[i+i_begin][j+j_begin] += string.width_at_pixel(x, y, i, j, self.pixel_width)
        
        wake_index = self.pixel_index(phi, theta)
        wake_centre = self.pixel_pos(*wake_index)
        indices = string.edge_scan(wake_centre[0] - phi, wake_centre[1] - theta, self.pixel_width)
        for index in indices:
            if 0 <= wake_index[1]+index[1] < self.num_pixels and 0 <= wake_index[0]+index[0] < self.num_pixels:
                self.pixels[wake_index[1]+index[1]][wake_index[0]+index[0]] += 10*uK
        
    
    
    def draw(self):
        plt.imshow(self.pixels/uK, interpolation='nearest')
        cbar = plt.colorbar()
        plt.xlabel("Pixel Number")
        plt.ylabel("Pixel Number")
        cbar.ax.set_ylabel('Temperature ($\mu$K)', labelpad=14)
        plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
