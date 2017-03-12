import os
import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt

try:
    import camb
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    import camb
from camb import model, correlations


pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.07, omk=0)
pars.set_dark_energy()  # re-set defaults
pars.InitPower.set_params(ns=0.965, As=2e-9)

pars.set_matter_power(redshifts=[0., 0.17, 3.1])
pars.NonLinear = model.NonLinear_none
data = camb.get_results(pars)

lmax = 8000
pars.set_for_lmax(lmax)
cls = data.get_cmb_power_spectra(pars)
cls_tot = data.get_total_cls(8000)
#cls_scal = data.get_unlensed_scalar_cls(2500)
#cls_tensor = data.get_tensor_cls(2000)
#cls_lensed = data.get_lensed_scalar_cls(3000)
#cls_phi = data.get_lens_potential_cls(2000)

# check lensed CL against python; will only agree well for high lmax as python has no extrapolation template
#cls_lensed2 = correlations.lensed_cls(cls['unlensed_scalar'], cls['lens_potential'][:, 0], delta_cls=False)
plt.loglog(np.sqrt(cls_tot[:, 0])*2.7/1e-6)
plt.savefig('foo.png')
