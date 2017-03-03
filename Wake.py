from __future__ import division
import numpy as np
import Constants as const
from Grid import Grid

def plus_minus(x, y):
    return x + y, x - y

class Wake(object):
    def __init__(self, angle, phi, theta, total_width, length):
        self.angle = angle
        self.phi = phi
        self.theta = theta
        self.length = length
        self.total_width = total_width
        self.span = 1*const.deg
        self.smooth = 0.5*const.deg
    
    @property
    def slope(self):
        return np.tan(self.angle)
    
    @property
    def inverse_slope(self):
        return -1/self.slope
    
    @property
    def line_hat(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])
    
    @property
    def rho_hat(self):
        return np.array([-np.sin(self.angle), np.cos(self.angle)])
    
    
    def linear_coords(self, x):
        # [rho, line] conversion
        rho = np.dot(x, self.rho_hat)
        line = np.dot(x, self.line_hat)
        
        return rho, line
    
    def width(self, rho, line):
        line = abs(line)
        if line < self.length/2 - self.smooth:
            scaling = 1
        elif line < self.length/2:
            scaling = (self.length/2 - line)/self.smooth
        else:
            scaling = 0
        
        if abs(rho) > self.span/2:
            return 0
        
        return scaling*(rho + self.span/2)/self.span * self.total_width
    
    def width_at_pixel(self, x_0, y_0, i, j, step):
        grid = Grid(step)
        grid.set_origin(x_0, y_0)
        
        pos = grid.pos(i, j)
        
        rho, line = self.linear_coords(pos)
        return self.width(rho, line)
    
    @property
    def front_edge(self):
        line_bound = (self.length/2) * self.line_hat
        rho_bound = (self.span/2) * self.rho_hat
        
        return plus_minus(rho_bound, line_bound)
    
    @property
    def back_edge(self):
        line_bound = (self.length/2) * self.line_hat
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
    def __init__(self, window_size, theta):
        self.window_size = window_size
        self.theta = theta
        
        self.hubble_decomposition = 0.01
        
        self.v_gamma_s = 0.15
        self.G_mu = 1e-7
    
    @property
    def step_factor(self):
        return (5/2)**(self.hubble_decomposition)
    
    @property
    def solid_angle(self):
        return  self.window_size**2
    
    def N(self, t_i, t):
        t_i_scaling = const.t_0 / t_i
        t_scaling = lambda t1: (1 - np.sqrt(a(t1)))**3
        
        return (8/3 * self.hubble_decomposition * const.string_formation_density)\
               * self.solid_angle * t_i_scaling * (t_scaling(t) - t_scaling(self.step_factor*t))
    
    def P(self, t_i, t):
        t_i_scaling = ((redshift(t_i) + 1)/1e3)**3
        t_scaling = ((redshift(t) + 1)/1e3)**2
        
        return const.cmb_quadrupole * const.ionization_fraction(redshift(t)) * self.G_mu * self.v_gamma_s * const.baryon_density_parameter * t_i_scaling * t_scaling * 1e7
    
    def hubble_angle(self, t_i, t):
        return (t_i/const.t_0)**(1/3)/(2 * (1 - (t/const.t_0)**(1/3)))
    
    def genetate_strings(self):
        t_next = lambda t_now: self.step_factor * t_now
        t = lambda t_now: (t_now + t_next(t_now))/ 2
        
        tot_n = 0
        t_n = time(const.z_last_scatter)
        while t_next(t_n) < const.t_0:
            t_i = time(const.z_matter_radiation)
            while t_next(t_i) < t(t_n):
                args = (t_i, t(t_n))
                if self.P(*args)/const.uK > 0.1:
                    tot_n += self.N(*args)
                    #print "t_i = %f, t = %f" %args
                    #print self.N(*args), self.P(*args)/const.uK, self.hubble_angle(*args)/const.deg, "\n"
                else:
                    break
                t_i = t_next(t_i)
            t_n = t_next(t_n)
        
        print tot_n

if __name__ == "__main__":
    WakePlacer(50*const.deg, 90*const.deg).genetate_strings()
    #print redshift(5/2*time(const.z_matter_radiation))
