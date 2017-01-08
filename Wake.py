from __future__ import division
import numpy as np
import math
import Constants as const

class SimpleWake(object):
    def __init__(self, phi, dist, tot_width, orientation):
        self.phi = phi
        self.dist = dist
        self.tot_width = tot_width
        self.span = 1.5*const.deg
        
        assert abs(orientation) == 1, "The attribute orientation must be +/- 1"
        self.orientation = orientation
    
    def linear_coords(self, x, y):
        # [rho, line] conversion
        y_vec = np.array([-math.sin(self.phi), math.cos(self.phi)])
        x_vec = np.array([math.cos(self.phi), math.sin(self.phi)])
        
        return y*y_vec + x*x_vec
    
    def width(self, rho):
        # Measure the off dimension from the string itself rather than centre of image
        rho = rho - self.dist
        
        if self.orientation * rho < 0:
            return 0
        
        linear = self.span - self.orientation * rho
        if linear < 0:
            return 0
        
        return self.tot_width * (linear / self.span)
    
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

