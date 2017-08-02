from __future__ import division
import numpy as np
import Constants as const
from Grid import Grid
import math

def plus_minus(x, y):
    return x + y, x - y

class Wake(object):
    def __init__(self, phi, theta, orientation, intensity, length):
        self.orientation = orientation
        self.phi = phi
        self.theta = theta
        self.length = length
        self.intensity = intensity

    @property
    def span(self):
        return self.length/2
    
    @property
    def smooth(self):
        return self.length/4
    
    @property
    def slope(self):
        return np.tan(self.orientation)
    
    @property
    def inverse_slope(self):
        return -1/self.slope
    
    @property
    def line_hat(self):
        return np.array([np.cos(self.orientation), np.sin(self.orientation)])
    
    @property
    def rho_hat(self):
        return np.array([-np.sin(self.orientation), np.cos(self.orientation)])
    
    
    def linear_coords(self, x):
        # [rho, line] conversion
        rho = np.dot(x, self.rho_hat)
        line = np.dot(x, self.line_hat)
        
        return rho, line
    
    def width(self, rho, line):
        line = abs(line)
        if line < self.length/2:
            scaling = 1
        elif self.length/2 <= line < self.length/2 + self.smooth:
            scaling = (self.smooth + self.length/2 - line)/self.smooth
        else:
            scaling = 0
        
        if abs(rho) > self.span/2:
            return 0
        
        #return scaling*(0.5 if rho>0 else -0.5) * self.intensity
        return scaling*(rho + self.span/2)/self.span * self.intensity
    
    def width_at_pixel(self, x_0, y_0, i, j, step):
        grid = Grid(step)
        grid.set_origin(x_0, y_0)
        
        pos = grid.pos(i, j)
        
        rho, line = self.linear_coords(pos)
        return self.width(rho, line)
    
    @property
    def front_edge(self):
        line_bound = (self.length/2 + self.smooth) * self.line_hat
        rho_bound = (self.span/2) * self.rho_hat
        
        return plus_minus(rho_bound, line_bound)
    
    @property
    def back_edge(self):
        line_bound = (self.length/2 + self.smooth) * self.line_hat
        rho_bound = (self.span/2) * self.rho_hat
        
        return plus_minus(-rho_bound, line_bound)
    
    def pixelation(self, x_0, y_0, step):
        grid = Grid(step)
        grid.set_origin(x_0, y_0)
        
        x_1, x_2 = self.front_edge
        x_3, x_4 = self.back_edge
        
        return grid.pixelate_area(x_1, x_2, x_3, x_4)
    
    def edge_scan(self, x_0, y_0, step):
        grid = Grid(step)
        grid.set_origin(x_0, y_0)
        
        return grid.edge_scan(*self.front_edge)

def gamma(v):
    # Reletivistic gamma factor for a velocity v
    return 1/np.sqrt(1 - (v)**2)

def time(z):
    # The time (measured from Big Bang) corresponding to the redshift z
    return const.t_0/(1.0 + z)**(3/2)

def redshift(t):
    # The redshift associated with the given time t
    return (const.t_0/t)**(2/3) - 1

def a(t):
    # Scale factor of the universe at the given time t
    # Set to unity at present time
    return (t/const.t_0)**(2/3)

def hubble_length(t):
    # Hubble length at the given time t using scale factor definition
    return 3*t/2

def comoving(x, t):
    # Comoving distance associated with physical distance x at time t
    return x/a(t)

def physical(x_c, t):
    # Physical distance associated with comoving distance x_c at time t
    return x_c*a(t)

class WakePlacer(object):
    def __init__(self, mode, window_size, phi, theta):
        self.mode = mode
        self.window_size = window_size
        self.phi = phi
        self.theta = theta
        
        self.hubble_decomposition = 0.01
        self.intensity_cutoff = 0.001*const.uK
        self.size_cutoff = 5*const.deg
        
        self.v_gamma_s = 0.15
        self.G_mu = 8e-8
    
    @property
    def step_factor(self):
        return (5/2)**(self.hubble_decomposition)
    
    @property
    def solid_angle(self):
        return  self.window_size**2
    
    def N(self, t_i, t):
        t_i_scaling = const.t_0 / t_i
        t_begin = (2/(1+self.step_factor))*t
        t_scaling = lambda t1: (1 - np.sqrt(a(t1)))**3
        
        return (8/3 * (self.hubble_decomposition if self.mode == "pol" else 1) * const.string_formation_density)\
               * self.solid_angle * t_i_scaling * (t_scaling(t_begin) - t_scaling(self.step_factor*t_begin))
    
    def T(self, t_i, t):
        t_i_scaling = ((redshift(t_i) + 1)/1e3)**(1/2)
        t_scaling = ((redshift(t) + 1)/1e3)**2
        
        if self.mode == "pol":
            return const.cmb_quadrupole * const.ionization_fraction(redshift(t)) * self.G_mu * self.v_gamma_s * const.baryon_density_parameter * t_i_scaling * t_scaling * 1e7
        elif self.mode == "temp":
            return 8 * np.pi * self.G_mu * self.v_gamma_s * const.cmb_temp
    
    def W(self, t_i, t):
        return (t_i/const.t_0)**(1/3)/(2 * (1 - (t/const.t_0)**(1/3)))
    
    def add_signals(self, array, t_i, t):
        N = self.N(t_i, t)
        fixed_n = int(math.floor(N))
        dynamic_n = (1 if np.random.random() < N - fixed_n else 0)
        
        for _ in range(fixed_n + dynamic_n):
            #intensity = np.random.normal(0, self.P(t_i, t))
            intensity = np.random.random() * self.T(t_i, t)
            length = self.W(t_i, t)
            
            orientation = 2*const.pi * np.random.random()
            phi = self.window_size * (np.random.random() - 0.5) + self.phi
            theta = self.window_size * (np.random.random() - 0.5) + self.theta
            
            array.append((phi, theta, orientation, intensity, length))
    
    def genetate_wakes(self):
        t_next = lambda t_now: self.step_factor * t_now
        t = lambda t_now: (t_now + t_next(t_now))/ 2
        
        signal_list = []
        t_n = time(const.z_last_scatter)
        while t_next(t_n) < const.t_0:
            if self.mode == "temp":
                args = (t(t_n), t(t_n))
                if self.W(*args) < self.size_cutoff:
                    self.add_signals(signal_list, *args)
                
            elif self.mode == "pol":
                t_i = time(const.z_matter_radiation)
                while t_next(t_i) < t(t_n):
                    args = (t_i, t(t_n))
                    if self.T(*args) > self.intensity_cutoff:
                        self.add_signals(signal_list, *args)
                    else:
                        break
                    t_i = t_next(t_i)
            t_n = t_next(t_n)
        
        return signal_list

#if __name__ == "__main__":
    #signal_list = WakePlacer(50*const.deg, 0, 90*const.deg).genetate_wakes()
    #print len(signal_list)
    #print redshift(5/2*time(const.z_matter_radiation))
