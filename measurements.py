import h5py
import numpy as np
from math import log
from copy import deepcopy
from scipy.interpolate import interp2d
from scipy.stats import skewnorm

# read o2scl hist2d object
def read_hist2d(f, dataset, path, xdir, ydir):
    if dataset == False:
        dset = f
    else:
        dset = f[dataset]
    data = dset[path].value

    data = data.T #need to transpose

    xval = dset[xdir].value
    yval = dset[ydir].value

    return data, xval, yval

# define different measurements here


# non correlated 2D Gaussian 
def gaussian_MR(mass, rad, conf):
    rad_gauss  = -0.5*( (rad  - conf["rad_mean"]) /conf["rad_std" ])**2.0
    mass_gauss = -0.5*( (mass - conf["mass_mean"])/conf["mass_std"])**2.0

    return rad_gauss + mass_gauss

# Values from Nattila et al 2017 for 4U 1702-429, arXiv:1709.09120
NSK17 = { "rad_mean": 12.4,
           "rad_std": 0.4,
         "mass_mean": 1.8,
          "mass_std": 0.2,
        }


# log likelihood function for MR measurements
def measurement_MR(mass, rad, density):
    fun = density(rad, mass)
    return  log( fun )


# interpolating density function
def interp_MR(string):
    if string == "SHB18_6304_He":
        fname = 'mrdata/shb18/6304_He_nopl_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_6397_He":
        fname = 'mrdata/shb18/6397_He_syst_wilm3.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_M28_He":
        fname = 'mrdata/shb18/M28_He_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_M30_H":
        fname = 'mrdata/shb18/M30_H_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_X7_H":
        fname = 'mrdata/shb18/X7_H_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_X5_H":
        fname = 'mrdata/shb18/X5_H_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHB18_wCen_H":
        fname = 'mrdata/shb18/wCen_H_syst_wilm.o2'
        dataset = "rescaled"
        path = 'data/like'
    elif string == "SHS18_M13_H":
        fname = 'mrdata/shs18/M13_H_rs.o2'
        dataset = "rescaled_0"
        path = 'data/like'
    elif string == "NKS15_1724":
        fname = 'mrdata/nks15/1724b.o2'
        dataset = "mcarlo"
        path = 'data/weights'
    elif string == "NKS15_1810":
        fname = 'mrdata/nks15/1810b.o2'
        dataset = "mcarlo"
        path = 'data/weights'

    f = h5py.File(fname,'r')
    data, rval, mval = read_hist2d(f, dataset, path, 'xval', 'yval')

    return interp2d(rval, mval, data, kind='cubic')

# Measurements from Steiner et al 2018, arXiv:1709.05013
SHB18_6304_He = deepcopy(interp_MR("SHB18_6304_He"))    # NGC 6304, helium
SHB18_6397_He = deepcopy(interp_MR("SHB18_6397_He"))    # NGC 6397, helium
SHB18_M28_He = deepcopy(interp_MR("SHB18_M28_He"))      # M28, helium
SHB18_M30_H = deepcopy(interp_MR("SHB18_M30_H"))        # M30, hydrogen
SHB18_X7_H = deepcopy(interp_MR("SHB18_X7_H"))          # 47 Tuc X7, hydrogen
SHB18_X5_H = deepcopy(interp_MR("SHB18_X5_H"))          # 47 Tuc X5, hydrogen
SHB18_wCen_H = deepcopy(interp_MR("SHB18_wCen_H"))      # wCen, hydrogen

# Measurement from Shaw et al 2018, arXiv:1803.00029
SHS18_M13_H = deepcopy(interp_MR("SHS18_M13_H"))        # M13, hydrogen

# Measurement from Nattila et al 2016, arXiv:1509.06561
NKS15_1724 = deepcopy(interp_MR("NKS15_1724"))        # 4U 1724-307
NKS15_1810 = deepcopy(interp_MR("NKS15_1810"))        # SAX J1810.8-260


# log likelihood function for M measurements
def measurement_M(mass, par):
    loc = par['loc']
    shape = par['shape']
    scale = par['scale']

    return skewnorm.logpdf(mass, shape, loc, scale)

# Values from Antoniadis et al 2013 for J0348+0432, arXiv:1304.6875
J0348 = {    "loc": 1.9680226480964658,
           "shape": 2.0180547504576896,
           "scale": 0.06613638253163863,
        }

# Values from Cromartie et al 2019 for J0740+6620, arXiv:1904.06759
J0740 = {    "loc": 2.0494842569774145,
           "shape": 1.7069401394167225,
           "scale": 0.13147902131189182,
        }

# log likelihood function for TD measurements
def measurement_TD(TD1, TD2, density):
    res = density(TD1, TD2)

    return log( res )

# interpolating density function
def interp_TD(string):
    if string == "GW170817":
        fname = 'LV/LV_prior.h5'
        path = 'data'

    f = h5py.File(fname,'r')
    data, TD1val, TD2val = read_hist2d(f, False, path, 'x', 'y')

    return interp2d(TD1val, TD2val, data, kind='cubic')

GW170817 = deepcopy(interp_TD("GW170817"))
