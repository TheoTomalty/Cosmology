import sys
import os
import numpy as np
import LogLog as loglog

try:
    import camb
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    import camb
from camb import model

# Natural constants
c = 3.00e8 # m/s
hbar = 6.582119514e-16 # eV*s
pi = 3.141592
deg = 2*pi/360.0 # rad
arcmin = deg/60 #rad
uK = 1.0e-6 # Kelvin

# Cosmology parameters
c_1 = 1
t_0 = 13.772e9 # years
z_last_scatter = 1100
z_matter_radiation = 3600
baryon_density_parameter = 0.0486
cmb_quadrupole = 30*uK
cmb_temp = 2.725
string_formation_density = 3 # per hubble volume per hubble time


# Experimental parameters
resolution = 1*arcmin
image_pixel_width = 75
region_pixel_width = 15

# Constant Functions
pars = camb.CAMBparams()
pars.set_cosmology(H0=71.8, ombh2=0.0227, omch2=0.112, mnu=0.07, omk=0, YHe=0.24, tau=0.093)
pars.set_dark_energy()  # re-set defaults
pars.InitPower.set_params(ns=0.965, As=2e-9)

pars.set_matter_power(redshifts=[0., 0.17, 3.1])
pars.NonLinear = model.NonLinear_none
data = camb.get_results(pars)

lmax = 9000
pars.set_for_lmax(9000, lens_potential_accuracy=10)
cls = data.get_cmb_power_spectra(pars, lmax)
cls_tot = data.get_total_cls(lmax)

initializer1 = []
for l in range(10, 6000, 10):
    initializer1.append((l, np.sqrt(cls_tot[l, 0]/uK**2)*cmb_temp))
initializer2 = []
for l in range(10, 6550, 10):
    initializer2.append((l, np.sqrt((cls_tot[l, 2] + cls_tot[l, 1])/uK**2)*cmb_temp))

C_TT = loglog.LogLog(initializer1, spacing=10)
C_EB = loglog.LogLog(initializer2, spacing=10)

_ion_frac = loglog.LogLog(
    [
        (100, 3e-4*(4.0/3.0)**(0.2)), 
        (200, 3e-4*(4.0/3.0)**(0.9)), 
        (300, 4e-4*(5.0/4.0)**(0.9)), 
        (400, 6e-4), 
        (500, 8e-4), 
        (600, 1e-3*(2.0/1.0)**(0.2)), 
        (700, 1e-3*(2.0/1.0)**(0.8)), 
        (800, 3e-3*(4.0/3.0)**(0.95)), 
        (900, 1e-2*(2.0/1.0)**(0.4)), 
        (1000, 5e-2*(6.0/5.0)**(0.4)),
        (1100, 1e-1*(2.0/1.0)**(0.5))
    ]
)
ionization_fraction = lambda z: min(1, _ion_frac.eval(z))
