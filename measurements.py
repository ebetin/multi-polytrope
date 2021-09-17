import h5py
import numpy as np
from math import log
from copy import deepcopy
from scipy.interpolate import interp2d
from scipy.stats import skewnorm

# define different measurements here


# non correlated 2D Gaussian 
def gaussian_MR(mass, rad, conf):
    rad_gauss  = -0.5*( (rad  - conf["rad_mean"]) /conf["rad_std" ])**2.0
    mass_gauss = -0.5*( (mass - conf["mass_mean"])/conf["mass_std"])**2.0

    return rad_gauss + mass_gauss

# Values from Nattila et al 2017 for 4U 1702-429, arXiv:1709.09120
NSK17_old = { "rad_mean": 12.4,
           "rad_std": 0.4,
         "mass_mean": 1.8,
          "mass_std": 0.2,
        }

# log likelihood function for MR measurements
def measurement_MR(mass, rad, density):
    fun = density(rad, mass)

    if fun>0:
        return log( fun )
    else:
        return -np.inf

# read o2scl hist2d object
def read_hist2d_GW(f, dataset, path, xdir, ydir):
    if dataset == False:
        dset = f
    else:
        dset = f[dataset]
    data = dset[path][()]
    data = data.T

    xval = dset[xdir][()]
    yval = dset[ydir][()]

    return data, xval, yval

# read o2scl hist2d object
def read_hist2d_nicer(f):
    dset = f["hist2d"]
    data = dset['weights'][()]
    data = data.T

    xval = dset['x_bins'][()]
    yval = dset['y_bins'][()]

    return data, xval, yval

# read o2scl hist2d object
def read_hist2d(f):
    dset = f["hist2_table"]
    data = dset['data/avgs'][()]
    data = data.T

    xval = dset['xval'][()]
    yval = dset['yval'][()]

    return data, xval, yval

# read o2scl hist2d object
def read_hist2d_old(f):
    dset = f["mcarlo"]
    data = dset['data/weights'][()]
    data = data.T

    xval = dset['xval'][()]
    yval = dset['yval'][()]

    return data, xval, yval

# read o2scl rescaled object
def read_rescaled(f):
    dset = f["rescaled"]
    data = dset['data/like'][()]
    data = data.T

    xval = dset['xval'][()]
    yval = dset['yval'][()]

    return data, xval, yval

# read o2scl rescaled object
def read_rescaled_old(f):
    dset = f["rescaled_0"]
    data = dset['data/like'][()]
    data = data.T

    xval = dset['xval'][()]
    yval = dset['yval'][()]

    return data, xval, yval

# interpolating density function
def interp_MR(string):
    if string == "SHB18_6304_He":
        fname = 'mrdata/shb18/6304_He_nopl_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_6397_He":
        fname = 'mrdata/shb18/6397_He_syst_wilm3.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_M28_He":
        fname = 'mrdata/shb18/M28_He_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_M30_H":
        fname = 'mrdata/shb18/M30_H_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_X7_H":
        fname = 'mrdata/shb18/X7_H_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_X5_H":
        fname = 'mrdata/shb18/X5_H_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHB18_wCen_H":
        fname = 'mrdata/shb18/wCen_H_syst_wilm.o2'
        dataset = "rescaled"
        #path = 'data/like'
    elif string == "SHS18_M13_H":
        fname = 'mrdata/shs18/M13_H_rs.o2'
        dataset = "rescaled_0"
        #path = 'data/like'
    elif string == "NKS15_1724":
        fname = 'mrdata/nks15/1724b.o2'
        dataset = "mcarlo"
        #path = 'data/weights'
    elif string == "NKS15_1810":
        fname = 'mrdata/nks15/1810b.o2'
        dataset = "mcarlo"
        #path = 'data/weights'
    elif string == "NICER_0030":
        fname = 'mrdata/nicer2020/0030_st_pst.o2'
        dataset = "weights"
    elif string == "NSK17":
        fname = 'mrdata/nat17/1702_D_X_int.o2'
        dataset = "hist2_table"

    f = h5py.File(fname,'r')

    if dataset == "mcarlo":
        data, rval, mval = read_hist2d_old(f)
    elif dataset == "hist2_table":
        data, rval, mval = read_hist2d(f)
    elif dataset == "weights":
        data, xval, yval = read_hist2d_nicer(f)
        rval = [ (xval[i]+xval[i+1])*0.5 for i in range(len(xval)-1) ]
        mval = [ (yval[i]+yval[i+1])*0.5 for i in range(len(yval)-1) ]
    elif dataset == "rescaled":
        data, rval, mval = read_rescaled(f)
    elif dataset == "rescaled_0":
        data, rval, mval = read_rescaled_old(f)

    data = data.clip(min=0)
    datasum = sum(sum(data))
    data = [[x / datasum for x in subs] for subs in data]

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

# NICER, J0030+0451, arXiv:TODO
NICER_0030 = deepcopy(interp_MR("NICER_0030"))

# Values from Nattila et al 2017 for 4U 1702-429, arXiv:1709.09120
NSK17 = deepcopy(interp_MR("NSK17"))

# log likelihood function for M measurements
def measurement_M(mass, par):
    loc = par['loc']
    shape = par['shape']
    scale = par['scale']

    return skewnorm.logpdf(mass, shape, loc, scale)

# Values from Antoniadis et al. 2013 for J0348+0432, arXiv:1304.6875
# Skrewed distribution assumed
#J0348 = {    "loc": 1.9680226480964658,
#           "shape": 2.0180547504576896,
#           "scale": 0.06613638253163863,
#        }
# Normal distribution asssumed
J0348 = {    "loc": 2.01,
           "shape": 0,
           "scale": 0.04,
        }



# Values from Cromartie et al. 2020 for J0740+6620, arXiv:1904.06759
#J0740 = {    "loc": 2.0494842569774145,
#           "shape": 1.7069401394167225,
#           "scale": 0.13147902131189182,
#        }

# Values from Fonseca et al. 2021 for J0740+6620 arXiv:2104.00880
# Normal distribution asssumed
J0740 = {    "loc": 2.08,
           "shape": 0,
           "scale": 0.07,
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
    data, TD1val, TD2val = read_hist2d_GW(f, False, path, 'x', 'y')

    return interp2d(TD1val, TD2val, data, kind='cubic')

GW170817 = deepcopy(interp_TD("GW170817"))

