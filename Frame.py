from __future__ import division
import numpy as np
import math
import cmath
import Constants as const
from Grid import Grid
import Wake as wake
import matplotlib.pyplot as plt
from LogLog import LogLog

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
        self.num_pixels = num_pixels + 20
        self.size = size * (self.num_pixels/num_pixels)
        self.num_regions = num_pixels/const.region_pixel_width
        self.region_width = size/self.num_regions
        
        self.pixels = np.zeros([self.num_pixels, self.num_pixels])
        self.regions = np.zeros([self.num_regions, self.num_regions])
    
    @property
    def pixel_width(self):
        return self.size/self.num_pixels
    
    def pixel_index(self, x, y, region=False):
        width = self.pixel_width if not region else self.region_width
        size = self.size if not region else self.region_width * self.num_regions
        j = int(round((x - self.phi + size/2 - width/2)/width))
        i = int(round((y - self.theta + size/2 - width/2)/width))
        
        return i, j
    
    def pixel_pos(self, i, j, region=False):
        ''' Converts from pixel indices to angular coordinates

        :param i: Pixel index along height of image
        :param j: Pixel index along base of image
        :return: Angular coordinates
        '''
        width = self.pixel_width if not region else self.region_width
        size = self.size if not region else self.region_width * self.num_regions
        
        begin = np.array([
            self.phi - size/2 + width/2,
            self.theta - size/2 + width/2
        ])
        pixel = np.array([
            width, 
            width
        ])
        index = np.array([i, j])
        
        return np.sum([begin, np.multiply(pixel, index)], axis=0)
    
    def add_noise(self, spectrum):
        assert isinstance(spectrum, LogLog)
        
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
                power = spectrum.eval(l)**2 * d_log/ (4*math.pi * num_modes)
                
                # The power in a given mode is evenly distributed between the real and imaginary parts, hence 1/2 factor
                random_complex = np.random.randn() + np.random.randn()*1j
                modes[i][j] += np.sqrt(power/2) * random_complex
        
        self.pixels += np.fft.fft2(modes).real + np.fft.fft2(modes).imag
    
    def add_strings(self, scale=1):
        buffer = 1*const.deg
        num_strings = int(round((self.size/const.deg)**2))
        
        angles = 2*const.pi * np.random.random(num_strings)
        widths = np.random.normal(scale*0.1*uK, 0.03*uK, num_strings)
        
        phis = (self.size + buffer) * (np.random.random(num_strings) - 0.5) + self.phi
        thetas = (self.size + buffer) * (np.random.random(num_strings) - 0.5) + self.theta
        
        for angle, phi, theta, width in zip(angles, phis, thetas, widths):
            self.add_string(angle, phi, theta, width)
    
    def add_string(self, angle, phi, theta, width):
        string = wake.Wake(angle, phi, theta, width, 2*const.deg)
        
        i_centre, j_centre = self.pixel_index(phi, theta)
        grid_centre = self.pixel_pos(i_centre, j_centre)
        x_0, y_0 = grid_centre[0] - phi, grid_centre[1] - theta
        
        ranges = string.pixelation(x_0, y_0, self.pixel_width)
        for j, i_min, i_max in ranges:
            for i in range(i_min, i_max):
                if 0 <= i+i_centre < self.num_pixels and 0 <= j+j_centre < self.num_pixels:
                    self.pixels[j+j_centre][i+i_centre] += string.width_at_pixel(x_0, y_0, i, j, self.pixel_width)
        
        #indices = string.edge_scan(x_0, y_0, self.pixel_width)
        #for i, j in indices:
        #    if 0 <= i+i_centre < self.num_pixels and 0 <= j+j_centre < self.num_pixels:
        #        self.pixels[j+j_centre][i+i_centre] += 0
        
        # Print Region information
        i_reg, j_reg = self.pixel_index(phi, theta, region=True)
        region_grid = self.pixel_pos(i_reg, j_reg, region=True)
        x_1, y_1 = region_grid[0] - phi, region_grid[1] - theta
        
        grid = Grid(self.region_width)
        grid.set_origin(x_1, y_1)
        
        for i, j, in grid.edge_scan(*string.front_edge):
            if 0 <= i+i_reg < self.num_regions and 0 <= j+j_reg < self.num_regions:
                self.regions[j+j_reg][i+i_reg] += 1
        
    
    def draw(self):
        f = plt.figure(1)
        plt.imshow(self.regions, interpolation='nearest')
        f.show()
        
        g = plt.figure(2)
        plt.imshow(
            np.array([row[10:-10] for row in self.pixels[10:-10]])/uK,
            interpolation='nearest'
        )
        cbar = plt.colorbar()
        plt.xlabel("Pixel Number")
        plt.ylabel("Pixel Number")
        cbar.ax.set_ylabel('Temperature ($\mu$K)', labelpad=14)
        g.show()
        raw_input()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
