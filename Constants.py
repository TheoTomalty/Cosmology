import LogLog as plot

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
string_formation_density = 1 # per hubble volume per hubble time


# Experimental parameters
resolution = 1*arcmin
image_pixel_width = 75
region_pixel_width = 15

# Constant Functions
C_EE = plot.LogLog(
    [(2, 0.2*uK), (10, 0.05*uK), (600, 4*uK), (1200, 4*uK), (3000, 1*uK)]
)
C_TT = plot.LogLog(
    [(2, 30*uK), (30, 30*uK), (100, 70*uK), (700, 40*uK), (3000, 5*uK)]
)

_ion_frac = plot.LogLog(
    [(100, 3e-4), (200, 4e-4), (300, 5e-4), (400, 6e-4), (500, 8e-4), (600, 1.2e-3), (700, 1.9e-3), (800, 4e-3), (900, 1.4e-2), (1000, 5.7e-2)]
)
ionization_fraction = lambda z: min(1, _ion_frac.eval(z))
