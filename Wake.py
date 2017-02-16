from __future__ import division
import numpy as np
import math
import Constants as const
from Grid import Grid

def plus_minus(x, y):
    return x + y, x - y

class SimpleWake(object):
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

class String(object):
    def __init__(self, tension=3e-7):
        self.tension = tension
        self.theta = 7*const.deg
        self.phi = 0*const.deg
        self.z = 1000.0
        self.v_s = 0.8 # light-years per year, perpendicular to us
    
    @property
    def t_i(self):
        return time(self.z)
    
    @property
    def r_c(self):
        return comoving(1/2*hubble_length(self.t_i), self.t_i)

class Wake(String):
    def __init__(self, tension=3e-7):
        String.__init__(self, tension)
    
    @property
    def t_cross(self):
        return ((3*const.t_0 + self.r_c)/(3*const.t_0**(2/3)))**3
    
    @property
    def dims(self):
        return const.c_1*hubble_length(self.t_cross), self.t_i*self.v_s*gamma(self.v_s)
    
    @property
    def psi_0(self):
        return 24*const.pi/5*self.tension*self.v_s*gamma(self.v_s)*const.t_0/(self.z + 1)**(0.5)
    
    def height(self, t):
        return self.psi_0*(self.z + 1)/(redshift(t) + 1)**2
        
    def width(self, theta, phi):
        return

