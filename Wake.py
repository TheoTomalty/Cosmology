import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

c = 3.00e8 # m/s
hbar = 6.582119514e-16 # eV*s
pi = 3.141592
deg = 2*pi/360.0 # rad

c_1 = 1
t_0 = 13.772e9 # years

def gamma(v):
    return 1/np.sqrt(1 - (v)**2)

def time(z):
    return t_0/(1.0 + z)**(3/2)

def redshift(t):
    return (t_0/t)**(2/3) - 1

def a(t):
    return (t/t_0)**(2/3)

def hubble_length(t):
    return 3*t/2

def comoving(x, t):
    return x/a(t)

def physical(x_c, t):
    return x_c*a(t)

class String(object):
    def __init__(self, tension=3e-7):
        self.tension = tension
        self.theta = 7*deg
        self.phi = 0*deg
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
        return ((3*t_0 + self.r_c)/(3*t_0**(2/3)))**3
    
    @property
    def dims(self):
        return c_1*hubble_length(self.t_cross), self.t_i*self.v_s*gamma(self.v_s)
    
    @property
    def psi_0(self):
        return 24*pi/5*self.tension*self.v_s*gamma(self.v_s)*t_0/(self.z + 1)**(0.5)
    
    def height(self, t):
        return self.psi_0*(self.z + 1)/(redshift(t) + 1)**2
        
    def width(self, theta, phi):
        return

# Square dimensions
theta_max = 10*deg
phi_max = 10*deg

pixels = np.zeros([100, 100], dtype=float)

