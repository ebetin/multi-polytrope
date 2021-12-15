import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
#sys.path.insert(1, '/home/eannala/work/koodii/multi-polytrope')
import units as cgs

import argparse
import os

import h5py

from param_indices import blob_indices, param_names

from math import floor, ceil

from scipy.signal import savgol_filter

from hpd import hpd_grid

from scipy import interpolate

##################################################
#flags
flagTriangle = True
flagTriangle_nature = True
flagTriangleCGP = False
flagMR = True
flagEP = True
flagEP_tmp = False
flagNP = False
flagNE = False
flagNG = True
flagNC = True
flagNPP = False
flagMG = False
flagNM = False
flagNR = False
flagNL = False
flagML = False
histo  = False
flagNTa = True
flag_mr_dara = False
flagTriangle_td = False

flag_intervals = False


flag_mmax = False

##################################
# Parses
def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', 
            dest='file',
            default="",
            type=str,
            help='File name (e.g. M1_S3_PT0-s587198-w5-g200-n20000)')

    parser.add_argument('-b', '--burnin',
            dest='burnin',
            default=0,
            type=int,
            help='Length of the burn-in period (default: 0)')

    parser.add_argument('-t', '--thin',
            dest='thin',
            default=1,
            type=int,
            help='Thinning frequency (default: 1)')

    args = parser.parse_args()

    return args

##################################################
# read chain
args = parse_cli()

# e.g. 'M1_S4_PT0-s7346812-w2-g20-n10-HLPS+-uniform'
prefix = args.file
#filename = '../chains/csc/'+prefix+'run.h5'
filename = '../chains/'+prefix+'-run.h5'
#filename = '/media/eannala/My Data/csc/chains/'+prefix+'-run.h5' # TODO

reader = emcee.backends.HDFBackend(filename)

##################################################
# # of segments
eos_Nsegment_pos1 = prefix.find("S")
eos_Nsegment_pos2 = prefix.find("_", eos_Nsegment_pos1)
eos_Nsegment      = int(prefix[eos_Nsegment_pos1+1:eos_Nsegment_pos2])

# ngrid1
Ngrid_pos1 = prefix.find("g")
Ngrid_pos2 = prefix.find("-", Ngrid_pos1)
Ngrid      = int(prefix[Ngrid_pos1+1:Ngrid_pos2])

# phase transtion?
phaseTransition_pos1 = prefix.find("PT")
phaseTransition_pos2 = prefix.find("-", phaseTransition_pos1)
phaseTransition      = int(prefix[phaseTransition_pos1+2:phaseTransition_pos2])

# eos_model (0=poly, 1=cs2)
eos_model_pos1 = prefix.find("M")
eos_model_pos2 = prefix.find("_", eos_model_pos1)
eos_model = int(prefix[eos_model_pos1+1:eos_model_pos2])

# ceft_model (HLPS, HLPS3, HLPS+)
ceft_model_pos1 = prefix.find("H")
ceft_model_pos2 = prefix.find("-", ceft_model_pos1)
ceft_model = str(prefix[ceft_model_pos1:ceft_model_pos2])

# flag_TOV
flag_TOV_pos1 = prefix.find("TOV_")
flag_TOV_pos2 = prefix.find("-", flag_TOV_pos1)
flag_TOV = True if str(prefix[flag_TOV_pos1+4:flag_TOV_pos2])=='True' else False
flag_TOV = True

# flag_GW
flag_GW_pos1 = prefix.find("GW")
flag_GW_pos2 = prefix.find("-", flag_GW_pos1)
flag_GW = True if str(prefix[flag_GW_pos1:flag_GW_pos2])=='True' else False
flag_GW = True

# flag_Mobs
flag_Mobs_pos1 = prefix.find("Mobs")
flag_Mobs_pos2 = prefix.find("-", flag_Mobs_pos1)
#flag_Mobs = True if str(prefix[flag_Mobs_pos1:flag_Mobs_pos2])== 'True' else False
flag_Mobs = True

# flag_MRobs
flag_MRobs_pos1 = prefix.find("MRobs")
flag_MRobs = True if str(prefix[flag_MRobs_pos1:])== 'True' else False
#flag_MRobs_pos2 = prefix.find("-", MRobs_model_pos1) # TODO
#flag_MRobs = bool(str(prefix[flag_MRobs_pos1:flag_MRobs_pos2]))
flag_MRobs = True

flag_TD = False

##################################################
# get param_indices & blob params
with h5py.File(filename, 'r') as hf:
    global data_pos_mass, data_eps_grid, data_nsat_long, nsat_short_grid

    # First layer
    a_group_keys = list(hf.keys())

    # Second layer
    pos_a = a_group_keys.index('mcmc')
    data1 = hf[a_group_keys[pos_a]]
    b_group_keys = list(data1.keys())

    # mass_grid
    pos_mass = b_group_keys.index('mass_grid')
    data_mass_grid = list(data1[b_group_keys[pos_mass]])

    # eps_grid
    pos_eps = b_group_keys.index('eps_grid')
    data_eps_grid = list(data1[b_group_keys[pos_eps]])

    # nsat_long_grid
    pos_nsat_long = b_group_keys.index('nsat_long_grid')
    data_nsat_long_grid = list(data1[b_group_keys[pos_nsat_long]])

    # nsat_short_grid
    pos_nsat_short = b_group_keys.index('nsat_short_grid')
    data_nsat_short_grid = list(data1[b_group_keys[pos_nsat_short]])

param_indices = {
        'mass_grid' : data_mass_grid,
        'eps_grid': data_eps_grid,
        'nsat_long_grid': data_nsat_long_grid,
        'nsat_short_grid': data_nsat_short_grid,
               }

parameters2, param_indices = blob_indices(param_indices, eosmodel = eos_model, flag_TOV = flag_TOV, flag_GW = flag_GW, flag_Mobs = flag_Mobs, flag_MRobs = flag_MRobs, flag_TD = flag_TD)

##################################################
# Burn-in period
burnin = args.burnin

# Thinning
thin = args.thin

# Samples
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
#log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
blob_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
print(len(blob_samples),blob_samples.shape,blob_samples[0])
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
#print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat blob shape: {0}".format(blob_samples.shape))

all_samples = samples

##################################################
# create labels
labels = param_names(eos_model, ceft_model, eos_Nsegment, pt = phaseTransition, latex = True, flag_const_limits = False, flag_TOV = flag_TOV, flag_GW = flag_GW, flag_Mobs = flag_Mobs, flag_MRobs = flag_MRobs) # TODO flag_const_limits

##################################################
# triangle/corner plot

def round_up(x, d):
    x *= 10**d
    x = ceil(x)
    x /= 10**d
    return x

def round_down(x, d):
    x *= 10**d
    x = floor(x)
    x /= 10**d
    return x

# PLUS c2max
print(len(blob_samples[0]))
print(blob_samples[0][param_indices["c2max"]], "LOL")
if eos_model == 1:
    blobs_c2max = [item[param_indices["c2max"]] for item in blob_samples]
    blobs_muparam = [item[param_indices['mu_param']] for item in blob_samples]
    blobs_c2param = [item[param_indices['c2_param']] for item in blob_samples]
    samples_extra = [ [*item[:], blobs_muparam[i], blobs_c2param[i], blobs_c2max[i]] for i, item in enumerate(samples)]
    labels.append(r"$\mu_2$")
    labels.append(r"$c^2_2$")
    labels.append(r"$c^2_{max}$")
    samples_extra_max = np.max(samples_extra, axis=0)
    samples_extra_min = np.min(samples_extra, axis=0)
    if eos_Nsegment == 3:
        range_tmp = [(round_down(samples_extra_min[0], 1), round_up(samples_extra_max[0], 1)), (round_down(samples_extra_min[1], 1), round_up(samples_extra_max[1], 1)), (round_down(samples_extra_min[2], 1), round_up(samples_extra_max[2], 1)), (round_down(samples_extra_min[3], 2), round_up(samples_extra_max[3], 2)), (round_down(samples_extra_min[4], 1), round_up(samples_extra_max[4], 1)), (round_down(samples_extra_min[5], 1), round_up(samples_extra_max[5], 1)), (round_down(samples_extra_min[6], 1), round_up(samples_extra_max[6], 1)), (0, 1), (round_down(samples_extra_min[8], 1), round_up(samples_extra_max[8], 1)), (0, 1), (0.2, 1)]
elif eos_model == 0:
    blobs_c2max = [item[param_indices["c2max"]] for item in blob_samples]
    blobs_gamma1 = [item[param_indices['gamma1']] for item in blob_samples]
    blobs_gamma2 = [item[param_indices['gamma2']] for item in blob_samples]
    if flag_TOV:
        blobs_mmax_rho = [item[param_indices['mmax_rho']] for item in blob_samples]
        blobs_mmax = [item[param_indices['mmax']] for item in blob_samples]
        samples_extra = [ [*item[:], blobs_gamma1[i], blobs_gamma2[i], blobs_c2max[i], blobs_mmax_rho[i], blobs_mmax[i]] for i, item in enumerate(samples)]
    else:
        samples_extra = [ [*item[:], blobs_gamma1[i], blobs_gamma2[i], blobs_c2max[i]] for i, item in enumerate(samples)]
    labels.append(r"$\gamma_1$")
    labels.append(r"$\gamma_2$")
    labels.append(r"$c^2_{max}$")
    if flag_TOV:
        labels.append(r"$n(M_{max})$")
        labels.append(r"$M_{max}$")
    samples_extra_max = np.max(samples_extra, axis=0)
    samples_extra_min = np.min(samples_extra, axis=0)
    '''
    if eos_Nsegment == 2:
        range_tmp = [(round_down(samples_extra_min[0], 1), round_up(samples_extra_max[0], 1)), (round_down(samples_extra_min[1], 1), round_up(samples_extra_max[1], 1)), (round_down(samples_extra_min[2], 1), round_up(samples_extra_max[2], 1)), (round_down(samples_extra_min[3], 2), round_up(samples_extra_max[3], 2)), (round_down(samples_extra_min[4], 1), round_up(samples_extra_max[4], 1)), (0, 10.), (1, round_up(samples_extra_max[6], 1)), (1, round_up(samples_extra_max[7], 1)), (0, round_up(samples_extra_max[8], 1)), (0.2, 1), (round_down(samples_extra_min[10], 1), round_up(samples_extra_max[10], 1))]
    elif eos_Nsegment == 3:
        range_tmp = [(round_down(samples_extra_min[0], 1), round_up(samples_extra_max[0], 1)), (round_down(samples_extra_min[1], 1), round_up(samples_extra_max[1], 1)), (round_down(samples_extra_min[2], 1), round_up(samples_extra_max[2], 1)), (round_down(samples_extra_min[3], 2), round_up(samples_extra_max[3], 2)), (round_down(samples_extra_min[4], 1), round_up(samples_extra_max[4], 1)), (0, 10.), (0, round_up(samples_extra_max[6], 1)), (1, round_up(samples_extra_max[7], 1)), (1, round_up(samples_extra_max[8], 1)), (0, round_up(samples_extra_max[9], 1)), (0, round_up(samples_extra_max[10], 1)), (0.2, 1)]
    '''

from scipy import stats
bin_width = 2. * stats.iqr(samples_extra, axis=0, rng=(25,75), scale='raw') / len(samples_extra)**(1./3.)
bin_size = [int(round((samples_extra_max[i] - samples_extra_min[i]) / bin_width[i])) for i in range(len(bin_width))]
print(bin_size)

if flagTriangle:
    fig = corner.corner(#all_samples,
            samples_extra,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            #levels=(1-np.exp(-0.5),),
            #title_fmt='.2e',
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels,
            #range=[(.8, 1.1), (.2, .6), (1.4, 2.4), (.03, .13), (.75, 1.1), (0., 10.), (1.21875, 40.)]
            #range=[(.8, 1.1), (.2, .6), (1.4, 2.4), (.03, .13), (.75, 1.1), (0., 10.), (.9, 2.74), (0, 1), (.9, 2.74), (0, .4), (0.25, 1)]
            #range=range_tmp
            )
    
    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

if flagTriangle_nature:
    list_cs14 = []
    list_gamma14 = []
    list_pfd14 = []
    list_cs20 = []
    list_gamma20 = []
    list_pfd20 = []
    list_rho20 = []
    for item in blob_samples:
        rho_list = [item[param_indices["nsat_mass_"+str(i)]] for i, _ in enumerate(param_indices["nsat_short_grid"])]
        cs2_list = [item[param_indices["nsat_c2_"+str(i)]] for i, _ in enumerate(param_indices["nsat_long_grid"])]
        gamma_list = [item[param_indices["nsat_gamma_"+str(i)]] for i, _ in enumerate(param_indices["nsat_long_grid"])]
        pfd_list = [item[param_indices["nsat_press_"+str(i)]] for i, _ in enumerate(param_indices["nsat_long_grid"])]
        func_rho = interpolate.interp1d(rho_list, param_indices["nsat_short_grid"])
        func_cs2 = interpolate.interp1d(param_indices["nsat_long_grid"], cs2_list)
        func_gamma = interpolate.interp1d(param_indices["nsat_long_grid"], gamma_list)
        func_pfd = interpolate.interp1d(param_indices["nsat_long_grid"], pfd_list)

        rho14 = func_rho(1.4)
        rho20 = item[param_indices["mmax_rho"]]
        if item[param_indices["mmax"]] > 2:
            rho20 = func_rho(2.0)
        list_rho20.append(rho20)
        #continue    

        list_cs14.append(func_cs2(rho14))
        list_gamma14.append(func_gamma(rho14))
        list_pfd14.append(func_pfd(rho14))
        list_cs20.append(func_cs2(rho20))
        list_gamma20.append(func_gamma(rho20))
        list_pfd20.append(func_pfd(rho20))


    #import matplotlib.pyplot as plt
    #plt.hist(list_rho20, density=True, bins=50)
    #plt.savefig("csc/rho20_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")
    
    '''
    samples_nature=[ [list_pfd14[i], list_cs14[i], list_gamma14[i], list_pfd20[i], list_cs20[i], list_gamma20[i], item[param_indices["mmax_ppFD"]], item[param_indices["mmax_c2"]], item[param_indices["mmax_gamma"]], item[param_indices["mmax"]], item[param_indices["c2max"]]] for i, item in enumerate(blob_samples)]
    labels_nature = [r"$p/p_{FD}(1.4M_{\odot})$", r"$c_s^2(1.4M_{\odot})$", r"$\Gamma(1.4M_{\odot})$", r"$p/p_{FD}(2M_{\odot})$", r"$c_s^2(2M_{\odot})$", r"$\Gamma(2M_{\odot})$", r"$p/p_{FD}(M_{max})$", r"$c_s^2(M_{max})$", r"$\Gamma(M_{max})$", r"$M_{max}$", r"$c^2_{s,max}$"]

    fig = corner.corner(
            samples_nature,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels_nature,
            #truths=[1., 1./3., 1.]
            )

    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_nature.pdf")
    '''

    samples_nature=[ [list_pfd14[i], list_cs14[i], list_gamma14[i]] for i, item in enumerate(blob_samples)]
    labels_nature = [r"$p/p_{FD}(1.4M_{\odot})$", r"$c_s^2(1.4M_{\odot})$", r"$\Gamma(1.4M_{\odot})$"]

    fig = corner.corner(
            samples_nature,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels_nature,
            truths=[.4, .4, 1.75]
            )

    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_nature_14.pdf")

    samples_nature=[ [list_pfd20[i], list_cs20[i], list_gamma20[i]] for i, item in enumerate(blob_samples)]
    labels_nature = [r"$p/p_{FD}(2M_{\odot})$", r"$c_s^2(2M_{\odot})$", r"$\Gamma(2M_{\odot})$"]

    fig = corner.corner(
            samples_nature,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels_nature,
            truths=[.4, .4, 1.75]
            )

    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_nature_20.pdf")

    samples_nature=[ [item[param_indices["mmax_ppFD"]], item[param_indices["mmax_c2"]], item[param_indices["mmax_gamma"]]] for i, item in enumerate(blob_samples)]
    labels_nature = [r"$p/p_{FD}(M_{max})$", r"$c_s^2(M_{max})$", r"$\Gamma(M_{max})$"]

    fig = corner.corner(
            samples_nature,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels_nature,
            truths=[.4, .4, 1.75]
            )

    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_nature_MAX.pdf")

print("MIN Cs2", np.amin([item[param_indices["c2max"]] for item in blob_samples]))

##################################################
# ensemble histograms

# trick to make nice colorbars
# see http://joseph-long.com/writing/colorbars/
def colorbar(mappable, 
        loc="right", 
        orientation="vertical", 
        size="5%", 
        pad=0.05, 
        ticklocation='auto'):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, orientation=orientation, ticklocation=ticklocation)


#--------------------------------------------------
def hpd(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]

    '''
    v = 2
    min_int_width2 = 100
    print(np.min(int_width), n, n_samples)
    input()
    for l1 in range(1, n_samples-2):
        for v in range(2, n-n_samples+2):
            for i1 in range(0, n-n_samples-v+2):
                i2 = i1 + l1 + v
                l2 = n_samples - l1 - 2
                int_width2_1 = d[i1+l1] - d[i1]
                int_width2_2 = d[i2+l2] - d[i2]
                int_width2 = int_width2_1 + int_width2_2
                if int_width2 < min_int_width2 and d[i2]-d[i1+l1] > 0.01 and int_width2_1 > 0.01 and int_width2_2 > 0.01 and int_width2 < np.min(int_width):
                    min_int_width2 = int_width2
                    min_int_width2_d = [[d[i1], d[i1+l1]], [d[i2], d[i2+l2]]]
                    print(l1, v, n_samples, min_int_width2, round(d[i2],2), round(d[i1+l1],2), round(d[i2]-d[i1+l1],2))
    print(min_int_width2_d)

    print(np.min(int_width), n, n_samples)
    print(np.array([d[min_int], d[min_int+n_samples]]))
    input()
    '''
    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

def hpd_histo(lista, frac):
    lista_sorted = np.sort(lista[:])[::-1]
    for n in range(len(lista)):
        if sum(lista_sorted[:-n-1]) <= frac * sum(lista):
            min_value = lista_sorted[-n-2]
            break
    region = []
    interval = []
    last = -1
    for k, item in enumerate(lista):
        if item >= min_value:
            if interval == []:
                interval.append(k)
            elif k != last + 1:
                interval.append(last)
                region.append(interval)
                interval = [k]
            last = k
    interval.append(last)
    region.append(interval)

    return region

if flag_TOV:
    blobs_mmax_rho = [item[param_indices["mmax_rho"]] for item in blob_samples]  # TODO

#--------------------------------------------------
#set up figure
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('axes', labelsize=7)

#####################################################################
#M-R
if flagMR:
    figMR = plt.figure(figsize=(3.54, 3.0)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsMR = []
    axsMR.append( plt.subplot(gs[0,0]) )

    for ax in axsMR:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Nr = 200 #number of radius histogram bins

    mass_grid = param_indices["mass_grid"]
    rad = np.zeros((nsamples, Ngrid))

    rad_grid = np.linspace(9.0, 16.0, Nr)

    #get rad from blobs
    for im, mass  in enumerate(param_indices['mass_grid']):
        ci = param_indices['rad_'+str(im)]
        rad[:, im] = blob_samples[:, ci]

    rad_hist = np.zeros((Nr-1, Ngrid))
    for im, mass  in enumerate(param_indices['mass_grid']):

        radslice = rad[:,im]
        rads = radslice[ radslice > 0.0 ]
        radm = np.mean(rads)
        print("M= {} R= {}".format(mass,radm))

        hist, bin_edges = np.histogram(rads, bins=rad_grid)

        rad_hist[:,im] = (1.0*hist[:])/hist.max()

    hdata_masked = np.ma.masked_where(rad_hist <= 0.0, rad_hist)

    axsMR[0].set_xlim((9.0, 16.0))
    axsMR[0].set_ylim((0.5,  2.6))
    axsMR[0].set_ylabel("Mass $M$ (M$_{\odot}$)")
    axsMR[0].set_xlabel("Radius $R$ (km)")

    im = axsMR[0].imshow(
            hdata_masked.T,
            extent=[rad_grid[0], rad_grid[-1], mass_grid[0], mass_grid[-1] ],
            origin='lower',
            interpolation='nearest',
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect='auto',
            )

    #causality limit (orange)
    rad0 = rad_grid[0]
    mass0 = rad0/(2.95*1.52)
    mass1 = mass_grid[-1]
    rad1 = mass1*2.95*1.52
    axsMR[0].fill_between([rad0, rad1], [mass0, mass1], [3.0, 3.0], color='orange', visible=True)
    #text(5.0, 2.3, "GR")

    #GR limits (darkorange)
    x_gr = [0, 17.5]
    y_gr = [0, 6.0]
    axsMR[0].fill_between(x_gr, y_gr, [3.0, 3.0], color='darkorange')
    #ax.text(7.4, 2.3, "Causality")
    #ax.text(5.5, 1.8, "Causality", rotation=39, size=10)

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    figMR.savefig("csc/mass_radius_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")



##################################################
# eps - P
if flagEP:
    figEP = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsEP = []
    axsEP.append( plt.subplot(gs[0,0]) )

    for ax in axsEP:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Nr = 200 #number of radius histogram bins

    rho_grid = param_indices["eps_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.logspace(-0.3, 4, Nr)

    eps1_v1 = []
    sigma1_v1 = []
    eps1_v2 = []
    sigma1_v2 = []

    eps2_v1 = []
    sigma2_v1 = []
    eps2_v2 = []
    sigma2_v2 = []

    #get P from blobs
    for ir, eps  in enumerate(param_indices['eps_grid']):
        ci = param_indices['Peps_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    press_hist = np.zeros((Nr-1, Ngrid))
    for ir, eps  in enumerate(param_indices['eps_grid']):
        pressslice = press[:,ir]
        press_s = pressslice[ pressslice > 0.0 ]
        pressm = np.mean(press_s)
        print("eps= {} P= {}".format(eps,pressm))

        hist, bin_edges = np.histogram(press_s, bins=press_grid)
        press_hist[:,ir] = (1.0*hist[:])/hist.max()

        if flag_intervals:
            sigma1_v2_tmp, _, _, _ = hpd_grid(press_s, alpha=1.-.682689492137, roundto=4)
            print(sigma1_v2_tmp)
            if len(sigma1_v2_tmp) == 2:
                eps1_v2.append(eps)
                sigma1_v2.append(sigma1_v2_tmp[1])
                #if not eps1_v1:
                #    eps1_v1.append(eps1_v2[-2])
                #    sigma1_v1.append([max(sigma1_v2_tmp[0]), max(sigma1_v2_tmp[0])])
                eps1_v1.append(eps)
                sigma1_v1.append(sigma1_v2_tmp[0])
            #elif eps < XXX:
            #    eps1_v2.append(eps)
            #    sigma1_v2.append(sigma1_v2_tmp[0])
            else:
                eps1_v1.append(eps)
                sigma1_v1.append(sigma1_v2_tmp[0])

            sigma2_v2_tmp, _, _, _ = hpd_grid(press_s, alpha=1.-.954499736104, roundto=4)
            print(sigma2_v2_tmp)
            if len(sigma2_v2_tmp) == 2:
                eps2_v2.append(eps)
                sigma2_v2.append(sigma2_v2_tmp[1])
                #if not eps2_v1:
                #    eps2_v1.append(eps2_v2[-2])
                #    sigma2_v1.append([max(sigma2_v2_tmp[0]), max(sigma2_v2_tmp[0])])
                eps2_v1.append(eps)
                sigma2_v1.append(sigma2_v2_tmp[0])
            #elif nsat < 5.:
            #    nsat2_v2.append(nsat)
            #    sigma2_v2.append(sigma2_v2_tmp[0])
            else:
                eps2_v1.append(eps)
                sigma2_v1.append(sigma2_v2_tmp[0])

    # pQCD results for different Xs
    import pQCD

    confacinv = 1000.0 / cgs.GeVfm_per_dynecm
    confac = cgs.GeVfm_per_dynecm * 0.001
    edens = np.empty((3, 200))
    press = np.logspace(-0.30103, 4., 200)
    for j, x in enumerate([1., 2., 4.]):
        for i, pr in enumerate(press):
            edens[j][i] = pQCD.eQCD_pressure(pr * confac, x) * cgs.c**2 * confacinv

    # Fermi-Dirac limit
    edens_SB = [3. * pr for pr in press]

    # Plot these extra lines
    #line_x1 = axsEP[0].plot(edens[0], press, color='k', linewidth=1, linestyle='--')
    #line_x2 = axsEP[0].plot(edens[1], press, color='k', linewidth=1, linestyle='--')
    #line_x4 = axsEP[0].plot(edens[2], press, color='k', linewidth=1, linestyle='--')
    #line_SB = axsEP[0].plot(edens_SB, press, color='k', linewidth=1, linestyle='-')

    # Density map
    hdata_masked = np.ma.masked_where(press_hist <= 0.0, press_hist)

    if flag_intervals:
        line_sigm1_v1 = axsEP[0].plot(eps1_v1, sigma1_v1, color='b', linewidth=1, linestyle=':')
        #line_sigm1_v2 = axsEP[0].plot(eps1_v2, sigma1_v2, color='c', linewidth=1, linestyle=':')
        line_sigm2_v1 = axsEP[0].plot(eps2_v1, sigma2_v1, color='b', linewidth=1, linestyle='-')
        #line_sigm2_v2 = axsEP[0].plot(eps2_v2, sigma2_v2, color='c', linewidth=1, linestyle='-')

    axsEP[0].set_xlim((1.0e2, 2.0e4))
    axsEP[0].set_ylim((0.5,  1.e4))
    axsEP[0].set_ylabel(r"Pressure $P$ (MeV/fm$^3$)")
    axsEP[0].set_xlabel(r"Energy density $\epsilon$ (MeV/fm$^3$)")
    axsEP[0].set_xscale('log')
    axsEP[0].set_yscale('log')

    press_grid = press_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(rho_grid, press_grid)

    im = axsEP[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            #norm=colors.LogNorm(vmin=hdata_masked.min(), vmax=hdata_masked.max()),
            )

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    figEP.savefig("csc/eps_P_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# eps - P txt file
if flagEP_tmp:
    with open("/home/eannala/work/koodii/multi-polytrope/tmp/AK/n_eps_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".dat", 'w') as f:
        print("Energy density [MeV/fm^3], min(Pressure) [MeV/fm^3], EoS id (p_min), max(Pressure) [MeV/fm^3], EoS id (p_max)", file=f)
        for ir, eps  in enumerate(param_indices['eps_grid']):
            ci = param_indices['Peps_'+str(ir)]
            val_min, idx_min = min((val, idx) for (idx, val) in enumerate(blob_samples[:, ci]))
            val_max, idx_max = max((val, idx) for (idx, val) in enumerate(blob_samples[:, ci]))
            ind_min = idx_min * thin + burnin
            ind_max = idx_max * thin + burnin
            print(eps, val_min, ind_min, val_max, ind_max, file=f)

##################################################
# n - press
if flagNP:
    figNP = plt.figure(figsize=(5.0, 4.5))
    gs = plt.GridSpec(1, 1)

    axsNP = []
    axsNP.append( plt.subplot(gs[0,0]) )

    for ax in axsNP:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.logspace(-0.3, 4.0, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_p_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    press_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        pressslice = press[:,ir]
        press_s = pressslice[ pressslice > 0.0 ]
        pressm = np.mean(press_s)
        print("n= {} P= {}".format(nsat,pressm))

        hist, bin_edges = np.histogram(press_s, bins=press_grid)
        press_hist[:,ir] = (1.0*hist[:])/hist.max()


    hdata_masked = np.ma.masked_where(press_hist <= 0.0, press_hist)

    axsNP[0].set_xlim((1.0, 46.0))
    axsNP[0].set_ylim((0.5,  1.e4))
    axsNP[0].set_ylabel(r"Pressure $P$ (MeV/fm$^3$)")
    axsNP[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")
    axsNP[0].set_yscale('log')

    press_grid = press_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, press_grid)

    im_ng = axsNP[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    '''
    # Different (low-density) cEFT models
    press_d_a = [0.47358105, 0.50707006, 0.63086134, 0.85528344, 1.191147, 1.5882902, 2.1354125, 2.4657881, 2.7346117]
    dens_d_a = [0.5625855, 0.579283, 0.6325189, 0.712665, 0.8087194, 0.90084237, 1.0033736, 1.0567989, 1.1000093]

    press_d_y = [0.6116683, 0.7649478, 1.0253471, 1.2392144, 1.3268421, 1.4624426, 1.6661912, 1.9220439, 2.144941, 2.4513721, 3.2084174, 3.7667534, 4.4130864]
    dens_d_y = [0.56260926, 0.6154528, 0.6851905, 0.73272896, 0.750212, 0.77653426, 0.8116961, 0.85235745, 0.8841791, 0.9209122, 0.9996805, 1.0493764, 1.1002502]

    press_h_a = [0.43274823, 0.47845787, 0.5909578, 0.77507275, 0.9099573, 1.0683063, 1.2977769, 1.4800721, 1.7037432, 2.1194534]
    dens_h_a = [0.5794646, 0.6065727, 0.6633424, 0.7379872, 0.7857194, 0.8367898, 0.90278727, 0.9495349, 1.005709, 1.0999856]

    press_h_y = [0.6918851, 0.8148284, 0.9389827, 1.1407166, 1.3886386, 1.6992031, 2.0366108, 2.5205762, 2.9591374, 3.590999]
    dens_h_y = [0.57891923, 0.623117, 0.6612255, 0.7130845, 0.77240574, 0.8360475, 0.89576, 0.97098845, 1.0305027, 1.100231]

    press_t_a = [0.42168525, 0.5186973, 0.68740445, 0.91570795, 1.0662277, 1.30068, 1.6452082, 2.0110593, 2.407884, 2.8562133]
    dens_t_a = [0.5792659, 0.6250387, 0.68770605, 0.753123, 0.788876, 0.83994997, 0.9053621, 0.9709675, 1.0326436, 1.1002097]

    press_t_y = [0.77378654, 0.9838406, 1.2126662, 1.3603245, 1.5921072, 1.851837, 2.1651025, 2.5976758, 3.0089011, 3.41735, 4.0621114]
    dens_t_y = [0.57932234, 0.64139676, 0.69502443, 0.727043, 0.7739894, 0.82093525, 0.8716125, 0.93682337, 0.99083805, 1.0391562, 1.1000462]

    press_l_a = [0.4112818, 0.47687966, 0.547814, 0.6019027, 0.6558796, 0.7146967, 0.75890017, 0.80250376, 0.9180314, 1.0086538, 1.0900255, 1.1610217]
    dens_l_a = [0.62521344, 0.68237007, 0.73599124, 0.7772371, 0.815144, 0.8532472, 0.8803515, 0.9082408, 0.97462547, 1.0235296, 1.0651667, 1.1001259]

    press_l_y = [0.80977595, 0.8760524, 0.97764766, 1.12544, 1.3185658, 1.5935394, 1.923855, 2.2915807, 2.732418, 3.1946306, 3.5650537]
    dens_l_y = [0.6248838, 0.6480624, 0.67968774, 0.7191707, 0.76631355, 0.82681227, 0.88868546, 0.9489865, 1.0098767, 1.0625176, 1.1002303]

    line_d_a = axsNP[0].plot(dens_d_a, press_d_a, color='k', linewidth=1, linestyle='-', label="Drischler et al. '19")
    line_d_y = axsNP[0].plot(dens_d_y, press_d_y, color='k', linewidth=1, linestyle='-')

    line_h_a = axsNP[0].plot(dens_h_a, press_h_a, color='c', linewidth=1, linestyle='-', label="Hebeler et al. '13")
    line_h_y = axsNP[0].plot(dens_h_y, press_h_y, color='c', linewidth=1, linestyle='-')

    line_t_a = axsNP[0].plot(dens_t_a, press_t_a, color='y', linewidth=1, linestyle='-', label="Tews et al. '13")
    line_t_y = axsNP[0].plot(dens_t_y, press_t_y, color='y', linewidth=1, linestyle='-')

    line_l_a = axsNP[0].plot(dens_l_a, press_l_a, color='b', linewidth=1, linestyle='-', label="Lynn et al. '16")
    line_l_y = axsNP[0].plot(dens_l_y, press_l_y, color='b', linewidth=1, linestyle='-')

    #axsNP[0].legend(handles=[line_d_a, line_h_a, line_t_a, line_l_a], loc="upper left")
    '''

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    figNP.savefig("csc/n_press_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - eps
if flagNE:
    figNE = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNE = []
    axsNE.append( plt.subplot(gs[0,0]) )

    for ax in axsNE:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    eps = np.zeros((nsamples, Ngrid))

    eps_grid = np.logspace(-0.3, 4.3, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_eps_'+str(ir)]
        eps[:, ir] = blob_samples[:, ci]

    eps_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        epsslice = eps[:,ir]
        eps_s = epsslice[ epsslice > 0.0 ] #* 2.99792458e10**2 * 1000.0 / (1.e9 * 1.602177e-12 / (1.0e-13)**3)
        epsm = np.mean(eps_s)
        print("n= {} eps= {}".format(nsat,epsm))

        hist, bin_edges = np.histogram(eps_s, bins=eps_grid)
        eps_hist[:,ir] = (1.0*hist[:])/hist.max()


    hdata_masked = np.ma.masked_where(eps_hist <= 0.0, eps_hist)

    axsNE[0].set_xlim((1.0, 46.0))
    axsNE[0].set_ylim((1.0e2, 5.0e4))
    axsNE[0].set_ylabel(r"Energy density $\epsilon$ (MeV/fm$^3$)")
    axsNE[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")
    axsNE[0].set_yscale('log')


    eps_grid = eps_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, eps_grid)

    im_ng = axsNE[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_eps_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


##################################################
# n - gamma
if flagNG:
    figNG = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNG = []
    axsNG.append( plt.subplot(gs[0,0]) )

    for ax in axsNG:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    gamma = np.zeros((nsamples, Ngrid))
    gamma_grid = np.linspace(0.0, 5.0, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_gamma_'+str(ir)]
        gamma[:, ir] = blob_samples[:, ci]
        #tmp = blob_samples[:, ci]
        #for i in range(len(blob_samples[:, ci])):
        #    if nsat > blobs_mmax_rho[i]:
        #        tmp[i] = np.nan
        #gamma[:, ir] = tmp

    nsat1_v1 = []
    sigma1_v1 = []
    nsat1_v2 = []
    sigma1_v2 = []

    nsat2_v1 = []
    sigma2_v1 = []
    nsat2_v2 = []
    sigma2_v2 = []

    gamma_hist = np.zeros((Ng-1, Ngrid))
    tmp = [item[param_indices['mmax_rho']] for item in blob_samples]
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        gammaslice = gamma[:,ir]
        gammas = gammaslice[ gammaslice > 0.0 ]
        if flag_mmax:
            gammas = gammas[ tmp >= nsat ]
        gammam = np.mean( gammas )
        print("n= {} G= {}".format(nsat, gammam))

        hist_ng, bin_edges = np.histogram(gammas, bins=gamma_grid)
        gamma_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()

        if flag_intervals:
            sigma1_v2_tmp, _, _, _ = hpd_grid(gammas, alpha=1.-.682689492137, roundto=4)
            print(sigma1_v2_tmp)
            if len(sigma1_v2_tmp) == 2:
                nsat1_v2.append(nsat)
                sigma1_v2.append(sigma1_v2_tmp[1])
                if not nsat1_v1:
                    nsat1_v1.append(nsat1_v2[-2])
                    sigma1_v1.append([max(sigma1_v2_tmp[0]), max(sigma1_v2_tmp[0])])
                nsat1_v1.append(nsat)
                sigma1_v1.append(sigma1_v2_tmp[0])
            #elif nsat < 5.:
            #    nsat1_v2.append(nsat)
            #    sigma1_v2.append(sigma1_v2_tmp[0])
            else:
                nsat1_v1.append(nsat)
                sigma1_v1.append(sigma1_v2_tmp[0])

            sigma2_v2_tmp, _, _, _ = hpd_grid(gammas, alpha=1.-.954499736104, roundto=4)
            print(sigma2_v2_tmp)
            if len(sigma2_v2_tmp) == 2:
                nsat2_v2.append(nsat)
                sigma2_v2.append(sigma2_v2_tmp[1])
                #if not nsat2_v1:
                #    nsat2_v1.append(nsat2_v2[-2])
                #    sigma2_v1.append([max(sigma2_v2_tmp[0]), max(sigma2_v2_tmp[0])])
                nsat2_v1.append(nsat)
                sigma2_v1.append(sigma2_v2_tmp[0])
            #elif nsat < 5.:
            #    nsat2_v2.append(nsat)
            #    sigma2_v2.append(sigma2_v2_tmp[0])
            else:
                nsat2_v1.append(nsat)
                sigma2_v1.append(sigma2_v2_tmp[0])

    hdata_masked_ng = np.ma.masked_where(gamma_hist <= 0.0, gamma_hist)

    if flag_intervals:
        line_sigm1_v1 = axsNG[0].plot(nsat1_v1, sigma1_v1, color='b', linewidth=1, linestyle=':')
        #line_sigm1_v2 = axsNG[0].plot(nsat1_v2, sigma1_v2, color='b', linewidth=1, linestyle=':')
        line_sigm2_v1 = axsNG[0].plot(nsat2_v1, sigma2_v1, color='b', linewidth=1, linestyle='-')
        #line_sigm2_v2 = axsNG[0].plot(nsat2_v2, sigma2_v2, color='b', linewidth=1, linestyle='-')

    # conformal limit
    line_conformal = axsNG[0].plot([1., 55.0], [1., 1.], color='k', linewidth=1, linestyle='--')
    line_175 = axsNG[0].plot([1., 55.0], [1.75, 1.75], color='k', linewidth=1, linestyle='-.')

    axsNG[0].set_xlim((1.0, 46.0))
    if flag_mmax:
        axsNG[0].set_xlim((1.0, 10.0))
    axsNG[0].set_ylim((0.0, 5.0))
    axsNG[0].set_ylabel(r"$\Gamma$")
    axsNG[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    gamma_grid = gamma_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, gamma_grid)

    im_ng = axsNG[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,#hdata_masked_ng.min()
            vmax=1.0,#np.amax(hist_ng),
            #norm=colors.LogNorm(vmin=1.0e-1, vmax=np.amax(hdata_masked_ng), clip=True),
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_v2.pdf")

##################################################
# n - c^2
if flagNC:
    #figNC = plt.figure(figsize=(3.7, 3.5)) #single column fig
    figNC = plt.figure(figsize=(5.0, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure

    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsNC = []
    axsNC.append( plt.subplot(gs[0,0]) )

    for ax in axsNC:
        ax.minorticks_on()
    '''
    figNC = plt.figure(figsize=(10.0, 11.)) #single column fig
    gs = plt.GridSpec(3, 2, figure=figNC)
    figNC_ax1 = figNC.add_subplot(gs[0, 0])
    figNC_ax2 = figNC.add_subplot(gs[0, 1])
    figNC_ax3 = figNC.add_subplot(gs[1, 0])
    figNC_ax4 = figNC.add_subplot(gs[1, 1])
    figNC_ax5 = figNC.add_subplot(gs[2, 0])
    figNC_ax6 = figNC.add_subplot(gs[2, 1])
    '''
    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    c2 = np.zeros((nsamples, Ngrid))
    c2_grid = np.linspace(0.0, 1.0, Ng)
    #c2_grid = np.linspace(0.0, 0.15, Ng)

    nsat1_v1 = []
    sigma1_v1 = []
    nsat1_v2 = []
    sigma1_v2 = []

    nsat2_v1 = []
    sigma2_v1 = []
    nsat2_v2 = []
    sigma2_v2 = []

    #get c2 from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_c2_'+str(ir)]
        c2[:, ir] = blob_samples[:, ci]

    c2_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        c2slice = c2[:,ir]
        c2s = c2slice[ c2slice > 0.0 ]
        c2m = np.mean( c2s )

        if flag_intervals:
            sigma1_v2_tmp, _, _, _ = hpd_grid(c2s, alpha=1.-.682689492137, roundto=4)
            print(sigma1_v2_tmp)
            if len(sigma1_v2_tmp) == 2:
                nsat1_v2.append(nsat)
                sigma1_v2.append(sigma1_v2_tmp[1])
                if not nsat1_v1:
                    nsat1_v1.append(nsat1_v2[-2])
                    sigma1_v1.append([max(sigma1_v2_tmp[0]), max(sigma1_v2_tmp[0])])
                nsat1_v1.append(nsat)
                sigma1_v1.append(sigma1_v2_tmp[0])
            #elif nsat < 5.:
            #    nsat1_v2.append(nsat)
            #    sigma1_v2.append(sigma1_v2_tmp[0])
            else:
                nsat1_v1.append(nsat)
                sigma1_v1.append(sigma1_v2_tmp[0])

            sigma2_v2_tmp, _, _, _ = hpd_grid(c2s, alpha=1.-.954499736104, roundto=4)
            print(sigma2_v2_tmp)
            if len(sigma2_v2_tmp) == 2:
                nsat2_v2.append(nsat)
                sigma2_v2.append(sigma2_v2_tmp[1])
                #if not nsat2_v1:
                #    nsat2_v1.append(nsat2_v2[-2])
                #    sigma2_v1.append([max(sigma2_v2_tmp[0]), max(sigma2_v2_tmp[0])])
                nsat2_v1.append(nsat)
                sigma2_v1.append(sigma2_v2_tmp[0])
            #elif nsat < 5.:
            #    nsat2_v2.append(nsat)
            #    sigma2_v2.append(sigma2_v2_tmp[0])
            else:
                nsat2_v1.append(nsat)
                sigma2_v1.append(sigma2_v2_tmp[0])


        if False:
            if ir == 2:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax1.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax1.hist(c2s, bins=c2_grid)
                figNC_ax1.axvline(x=np.mean(c2s), color='black')
                figNC_ax1.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax1.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax1.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax1.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax1.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax1.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax1.transAxes)
                figNC_ax1.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax1.transAxes)
                figNC_ax1.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax1.transAxes)
                figNC_ax1.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax1.transAxes)
                figNC_ax1.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax1.transAxes)
                figNC_ax1.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax1.transAxes)
            if ir == 7:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax2.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax2.hist(c2s, bins=c2_grid)
                figNC_ax2.axvline(x=np.mean(c2s), color='black')
                figNC_ax2.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax2.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax2.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax2.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax2.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax2.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax2.transAxes)
                figNC_ax2.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax2.transAxes)
                figNC_ax2.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax2.transAxes)
                figNC_ax2.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax2.transAxes)
                figNC_ax2.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax2.transAxes)
                figNC_ax2.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax2.transAxes)
            elif ir == 15:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax3.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax3.hist(c2s, bins=c2_grid)
                figNC_ax3.axvline(x=np.mean(c2s), color='black')
                figNC_ax3.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax3.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax3.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax3.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax3.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax3.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax3.transAxes)
                figNC_ax3.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax3.transAxes)
                figNC_ax3.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax3.transAxes)
                figNC_ax3.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax3.transAxes)
                figNC_ax3.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax3.transAxes)
                figNC_ax3.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax3.transAxes)
            elif ir == 33:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax4.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax4.hist(c2s, bins=c2_grid)
                figNC_ax4.axvline(x=np.mean(c2s), color='black')
                figNC_ax4.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax4.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax4.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax4.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax4.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax4.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax4.transAxes)
                figNC_ax4.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax4.transAxes)
                figNC_ax4.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax4.transAxes)
                figNC_ax4.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax4.transAxes)
                figNC_ax4.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax4.transAxes)
                figNC_ax4.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax4.transAxes)
            elif ir == 52:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax5.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax5.hist(c2s, bins=c2_grid)
                figNC_ax5.axvline(x=np.mean(c2s), color='black')
                figNC_ax5.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax5.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax5.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax5.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax5.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax5.set_xlabel("$c_s^2$")
                figNC_ax5.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax5.transAxes)
                figNC_ax5.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax5.transAxes)
                figNC_ax5.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax5.transAxes)
                figNC_ax5.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax5.transAxes)
                figNC_ax5.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax5.transAxes)
                figNC_ax5.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax5.transAxes)
            elif ir == 70:
                hpd_1 = hpd(c2s, 0.682689492137)
                hpd_2 = hpd(c2s, 0.954499736104)
                figNC_ax6.set_title("${} n_s$".format(round(nsat, 1)))
                figNC_ax6.hist(c2s, bins=c2_grid)
                figNC_ax6.axvline(x=np.mean(c2s), color='black')
                figNC_ax6.axvline(x=np.median(c2s), color='magenta', ls = '-')
                figNC_ax6.axvline(x=hpd_1[0], color='red', ls = '-.')
                figNC_ax6.axvline(x=hpd_1[1], color='red', ls = '-.')
                figNC_ax6.axvline(x=hpd_2[0], color='green', ls = '-.')
                figNC_ax6.axvline(x=hpd_2[1], color='green', ls = '-.')
                figNC_ax6.set_xlabel("$c_s^2$")
                figNC_ax6.text(0.6, 0.9, "Mean: {}".format(round(np.mean(c2s), 3)), fontsize=12, transform=figNC_ax6.transAxes)
                figNC_ax6.text(0.6, 0.8, "Median: {}".format(round(np.median(c2s), 3)), fontsize=12, color='magenta',transform=figNC_ax6.transAxes)
                figNC_ax6.text(0.6, 0.7, "$1\sigma$ (l): {}".format(round(hpd_1[0], 3)), fontsize=12, color='red',transform=figNC_ax6.transAxes)
                figNC_ax6.text(0.6, 0.6, "$1\sigma$ (u): {}".format(round(hpd_1[1], 3)), fontsize=12, color='red',transform=figNC_ax6.transAxes)
                figNC_ax6.text(0.6, 0.5, "$2\sigma$ (l): {}".format(round(hpd_2[0], 3)), fontsize=12, color='green',transform=figNC_ax6.transAxes)
                figNC_ax6.text(0.6, 0.4, "$2\sigma$ (u): {}".format(round(hpd_2[1], 3)), fontsize=12, color='green',transform=figNC_ax6.transAxes)
            continue
            print(ir)  # 5n_s = 15, 10 = 33, 20 = 70
        print("n= {} c2= {}".format(nsat, c2m))

        hist_ng, bin_edges = np.histogram(c2s, bins=c2_grid)
        c2_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()
    #plt.savefig("csc/n_c2_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"--.pdf")
    #input()
    #plt.savefig("csc/n_c2_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"_histo.pdf")
    hdata_masked_ng = np.ma.masked_where(c2_hist <= 0.0, c2_hist)

    if flag_intervals:
        line_sigm1_v1 = axsNC[0].plot(nsat1_v1, sigma1_v1, color='b', linewidth=1, linestyle=':')
        #line_sigm1_v2 = axsNC[0].plot(nsat1_v2, sigma1_v2, color='c', linewidth=1, linestyle=':')
        line_sigm2_v1 = axsNC[0].plot(nsat2_v1, sigma2_v1, color='b', linewidth=1, linestyle='-')
        #line_sigm2_v2 = axsNC[0].plot(nsat2_v2, sigma2_v2, color='c', linewidth=1, linestyle='-')

    # conformal limit
    line_conformal = axsNC[0].plot([1., 55.0], [1./3., 1./3.], color='k', linewidth=1, linestyle='--')

    axsNC[0].set_xlim((1.0, 46.0))
    axsNC[0].set_ylim((0.0, 1.0))
    axsNC[0].set_ylabel(r"$c_s^2$")
    axsNC[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    c2_grid = c2_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, c2_grid)

    im_ng = axsNC[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            #vmin=0.0,
            #vmax=1.0,
            norm=colors.LogNorm(vmin=1.0e-1, vmax=np.amax(hdata_masked_ng), clip=True)
            #norm=colors.LogNorm(vmin=hdata_masked_ng.min(), vmax=hdata_masked_ng.max()),
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_c2_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - press/pFD
if flagNPP:
    figNPP = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNPP = []
    axsNPP.append( plt.subplot(gs[0,0]) )

    for ax in axsNPP:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of radius histogram bins

    nsat_grid = param_indices["nsat_long_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.linspace(0.0, 0.8, Ng)

    #get press from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_press_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    press_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        pressslice = press[:,ir]
        press_s = pressslice[ pressslice > 0.0 ]
        pressm = np.mean(press_s)
        print("n= {} p/pFD= {}".format(nsat, pressm))

        hist, bin_edges = np.histogram(press_s, bins=press_grid)
        press_hist[:,ir] = (1.0*hist[:])/hist.max()

    hdata_masked_ng = np.ma.masked_where(press_hist <= 0.0, press_hist)

    axsNPP[0].set_xlim((1.0, 46.0))
    axsNPP[0].set_ylim((0,0.8))
    axsNPP[0].set_ylabel(r"Normalized pressure $P/P_{FD}$")
    axsNPP[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    press_grid = press_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, press_grid)

    im_ng = axsNPP[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_pressFD_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# mass - gamma
if flagMG:
    flag_mmax = True
    figMG = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsMG = []
    axsMG.append( plt.subplot(gs[0,0]) )

    for ax in axsMG:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of mass histogram bins
    #Ngrid2 = len(param_indices['nsat_short_grid'])

    mass_grid = np.linspace(.5, 2.6, Ng)
    if flag_mmax:
        mass_grid = np.linspace(0.25, 1, Ng)

    nsat_grid = param_indices["nsat_short_grid"]
    gamma = np.zeros((nsamples, len(mass_grid)))
    gamma_grid = np.linspace(0., 8., Ng)
    #mass = np.zeros((nsamples, Ngrid2))
    #mass_grid = np.linspace(0.2, 3.0, Ng)

    for ib, item in enumerate(blob_samples):
        mass_list = [item[param_indices["nsat_mass_"+str(i)]] for i, _ in enumerate(param_indices["nsat_short_grid"])]
        gamma_list = [item[param_indices["nsat_gamma_"+str(i)]] for i, _ in enumerate(param_indices["nsat_long_grid"])]

        rho_list_short = param_indices["nsat_short_grid"]
        rho_list_short = [item for i, item in enumerate(rho_list_short) if mass_list[i] > 0]
        mass_list = [item for item in mass_list if item > 0]
        if item[param_indices["mmax"]] > mass_list[-1]:
            if item[param_indices["mmax_rho"]] < rho_list_short[-1]:
                mass_list.pop()
                rho_list_short.pop()
            mass_list.append(item[param_indices["mmax"]])
            rho_list_short.append(item[param_indices["mmax_rho"]])

        func_rho = interpolate.interp1d(mass_list, rho_list_short)
        func_gamma = interpolate.interp1d(param_indices["nsat_long_grid"], gamma_list)

        for im, mass in enumerate(mass_grid):
            tmp = 0.
            if flag_mmax:
                rho = func_rho(mass * item[param_indices["mmax"]])
                tmp = func_gamma(rho)
            else:
                if item[param_indices["mmax"]] > mass:
                    #print(mass, item[param_indices["mmax"]], item[param_indices["mmax_rho"]])
                    rho = func_rho(mass)
                    #print(rho)
                    tmp = func_gamma(rho)
            gamma[ib, im] = tmp

    #get mass from blobs
   # for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
   #     ci = param_indices['nsat_mass_'+str(ir)]
   #     mass[:, ir] = blob_samples[:, ci]

    gamma_hist = np.zeros((Ng-1, len(mass_grid)))
    for ir, mass in enumerate(mass_grid):
        gammaslice = gamma[:,ir]
        gamma_s = gammaslice[ gammaslice > 0.0 ]
        gammam = np.mean(gamma_s)
        print("M= {} G= {}".format(mass, gammam))

        hist, bin_edges = np.histogram(gamma_s, bins=gamma_grid)
        gamma_hist[:,ir] = (1.0*hist[:])/hist.max()

    hdata_masked_ng = np.ma.masked_where(gamma_hist <= 0.0, gamma_hist)

    axsMG[0].set_ylim((0, 8.))
    axsMG[0].set_xlim((0.5,  2.6))
    axsMG[0].set_xlabel("Mass $M$ (M$_{\odot}$)")
    if flag_mmax:
        axsMG[0].set_xlim((0.25,  1))
        axsMG[0].set_xlabel("Normalized mass $M/M_{TOV}$")
    axsMG[0].set_ylabel(r"Polytropic exponent $\Gamma$")

    gamma_grid = gamma_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(mass_grid, gamma_grid)

    im_ng = axsMG[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    if flag_mmax:
        plt.savefig("csc/nmass_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")
    else:
        plt.savefig("csc/mass_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - mass
if flagNM:
    figNM = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNM = []
    axsNM.append( plt.subplot(gs[0,0]) )

    for ax in axsNM:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of mass histogram bins
    Ngrid2 = len(param_indices['nsat_short_grid'])

    nsat_grid = param_indices["nsat_short_grid"]
    mass = np.zeros((nsamples, Ngrid2))
    mass_grid = np.linspace(0.2, 3.0, Ng)

    #get mass from blobs
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        ci = param_indices['nsat_mass_'+str(ir)]
        mass[:, ir] = blob_samples[:, ci]

    mass_hist = np.zeros((Ng-1, Ngrid2))
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        massslice = mass[:,ir]
        mass_s = massslice[ massslice > 0.0 ]
        massm = np.mean(mass_s)
        print("n= {} M= {}".format(nsat, massm))

        hist, bin_edges = np.histogram(mass_s, bins=mass_grid)
        mass_hist[:,ir] = (1.0*hist[:])/hist.max()

    hdata_masked_ng = np.ma.masked_where(mass_hist <= 0.0, mass_hist)

    axsNM[0].set_xlim((1.0, 12.0))
    axsNM[0].set_ylim((0.2,  3.0))
    axsNM[0].set_ylabel("Mass $M$ (M$_{\odot}$)")
    axsNM[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    mass_grid = mass_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, mass_grid)

    im_ng = axsNM[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_mass_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - radius
if flagNR:
    figNR = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNR = []
    axsNR.append( plt.subplot(gs[0,0]) )

    for ax in axsNR:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of radius histogram bins
    Ngrid2 = 100

    nsat_grid = param_indices["nsat_short_grid"]
    rad = np.zeros((nsamples, Ngrid2))
    rad_grid = np.linspace(9.0, 16.0, Ng)

    #get radius from blobs
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        ci = param_indices['nsat_radius_'+str(ir)]
        rad[:, ir] = blob_samples[:, ci]

    rad_hist = np.zeros((Ng-1, Ngrid2))
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        radslice = rad[:,ir]
        rad_s = radslice[ radslice > 0.0 ]
        radm = np.mean(rad_s)
        print("n= {} R= {}".format(nsat, radm))

        hist, bin_edges = np.histogram(rad_s, bins=rad_grid)
        rad_hist[:,ir] = (1.0*hist[:])/hist.max()

    hdata_masked_ng = np.ma.masked_where(rad_hist <= 0.0, rad_hist)

    axsNR[0].set_xlim((1.0, 12.0))
    axsNR[0].set_ylim((9.0, 16.0))
    axsNR[0].set_ylabel("Radius $R$ (km)")
    axsNR[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    rad_grid = rad_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, rad_grid)

    im_ng = axsNR[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_radius_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - TD
if flagNL:
    figNL = plt.figure(figsize=(4.1, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNL = []
    axsNL.append( plt.subplot(gs[0,0]) )

    for ax in axsNL:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of TD histogram bins
    Ngrid2 = 100

    nsat_grid = param_indices["nsat_short_grid"]
    TD = np.zeros((nsamples, Ngrid2))
    TD_grid = np.linspace(0.0, 1600.0, Ng)

    #get TD from blobs
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        ci = param_indices['nsat_TD_'+str(ir)]
        TD[:, ir] = blob_samples[:, ci]

    TD_hist = np.zeros((Ng-1, Ngrid2))
    for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
        TDslice = TD[:,ir]
        TD_s = TDslice[ TDslice > 0.0 ]
        TDm = np.mean(TD_s)
        print("n= {} TD= {}".format(nsat, TDm))

        hist, bin_edges = np.histogram(TD_s, bins=TD_grid)
        TD_hist[:,ir] = (1.0*hist[:])/hist.max()

    hdata_masked_ng = np.ma.masked_where(TD_hist <= 0.0, TD_hist)

    axsNL[0].set_xlim((1.0, 10.0))
    axsNL[0].set_ylim((0.0,  1600.0))
    axsNL[0].set_ylabel("Tidal deformability")
    axsNL[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    TD_grid = TD_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, TD_grid)

    im_ng = axsNL[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_TD_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# M - TD
if flagML:
    figML = plt.figure(figsize=(4.1, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsML = []
    axsML.append( plt.subplot(gs[0,0]) )

    for ax in axsML:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Nr = 200 #number of TD histogram bins

    mass_grid = param_indices["mass_grid"]
    TD = np.zeros((nsamples, Ngrid))

    TD_grid = np.linspace(0.0, 1600.0, Nr)

    #get TD from blobs
    for im, mass  in enumerate(param_indices['mass_grid']):
        ci = param_indices['TD_'+str(im)]
        TD[:, im] = blob_samples[:, ci]


    TD_hist = np.zeros((Nr-1, Ngrid))
    for im, mass  in enumerate(param_indices['mass_grid']):

        TDslice = TD[:,im]
        TDs = TDslice[ TDslice > 0.0 ]
        TDm = np.mean(TDs)
        print("M= {} TD= {}".format(mass,TDm))

        hist, bin_edges = np.histogram(TDs, bins=TD_grid)

        TD_hist[:,im] = (1.0*hist[:])/hist.max()

    hdata_masked = np.ma.masked_where(TD_hist <= 0.0, TD_hist)

    axsML[0].set_ylim((0.0,  1600.0))
    axsML[0].set_xlim((0.5,  3.0))
    axsML[0].set_xlabel("Mass $M$ (M$_{\odot}$)")
    axsML[0].set_ylabel("Tidal deformability")

    TD_grid = TD_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(mass_grid, TD_grid)

    im = axsML[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/mass_TD_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


if flagTriangleCGP:###########################################XXX
    #f, axs = plt.subplots(2,2)
    #P/P_FD
    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of radius histogram bins

    nsat_grid = param_indices["nsat_long_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.linspace(0.0, 0.85, Ng)

    #get press from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_press_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    #gamma
    nsat_grid = param_indices["nsat_long_grid"]
    gamma = np.zeros((nsamples, Ngrid))
    gamma_grid = np.linspace(0.0, 6.0, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_gamma_'+str(ir)]
        gamma[:, ir] = blob_samples[:, ci]

    #c_s^2
    nsat_grid = param_indices["nsat_long_grid"]
    c2 = np.zeros((nsamples, Ngrid))
    c2_grid = np.linspace(0.0, 1.0, Ng)

    #get c2 from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_c2_'+str(ir)]
        c2[:, ir] = blob_samples[:, ci]

    c2_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        c2slice = c2[:,ir]
        c2s = c2slice[ c2slice > 0.0 ]
        c2m = np.mean( c2s )
        print("n= {} G= {}".format(nsat, c2m))

        hist_ng, bin_edges = np.histogram(c2s, bins=c2_grid)
        c2_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()

    hdata_masked_c2 = np.ma.masked_where(c2_hist <= 0.0, c2_hist)
    import random
    mySet = random.sample(range(1, len(gamma)), 1000)
    if False:
        #p/p_FD vs gamma
        figPG = plt.figure(figsize=(3.7, 3.5)) #single column fig
        #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
        gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

        axsPG = []
        axsPG.append( plt.subplot(gs[0,0]) )

        for ax in axsPG:
            ax.minorticks_on()

        axsPG[0].set_ylim((0.0, 6.0))
        axsPG[0].set_xlim((0,0.85))

        axsPG[0].set_ylabel(r"$\Gamma$")
        axsPG[0].set_xlabel(r"Normalized pressure $P/P_{FD}$")

        for i in mySet:
            plt.plot(press[i,:],gamma[i,:],'k',linewidth=0.05)
        plt.savefig("csc/press_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

    if True:
        #p/p_FD vs gamma
        figPG = plt.figure(figsize=(3.7, 3.5)) #single column fig
        #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
        gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

        axsPG = []
        axsPG.append( plt.subplot(gs[0,0]) )

        for ax in axsPG:
            ax.minorticks_on()

        axsPG[0].set_ylim((0.0, 6.0))
        axsPG[0].set_xlim((0,0.85))

        axsPG[0].set_ylabel(r"$\Gamma$")
        axsPG[0].set_xlabel(r"Normalized pressure $P/P_{FD}$")

        press_grid = np.linspace(0.0, 0.85, Ng)
        gamma_grid = np.linspace(0.0, 6.0, Ng)

        print(param_indices['nsat_long_grid'][1:35])
        pr = np.asarray(press[:,1:35])
        g  = np.asarray(gamma[:,1:35])
        hdata_masked_g = np.ma.masked_where(c2_hist <= 0.0, c2_hist)
        H, xedges, yedges = np.histogram2d(pr.flatten(), g.flatten(), bins=(press_grid, gamma_grid))
        H = H.T / H.max()

        H = np.ma.masked_where(H <= 0.0, H)

        X, Y = np.meshgrid(xedges, yedges)

        im_ng = axsPG[0].pcolormesh(
                X,Y,H,
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                )

        cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")   

        plt.savefig("csc/press_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +"+.pdf")

    if False:
        #c2 vs gamma
        figCG = plt.figure(figsize=(3.7, 3.5)) #single column fig
        #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
        gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

        axsCG = []
        axsCG.append( plt.subplot(gs[0,0]) )

        for ax in axsCG:
            ax.minorticks_on()

        axsCG[0].set_ylim((0.0, 6.0))
        axsCG[0].set_xlim((0.0, 1.0))

        axsCG[0].set_ylabel(r"$\Gamma$")
        axsCG[0].set_xlabel(r"$c^2$")

        for i in mySet:
            plt.plot(c2[i,:],gamma[i,:],'k',linewidth=0.05)

        plt.savefig("csc/cs2_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

    if False:
        #p/p_FD vs c2
        figPC = plt.figure(figsize=(3.7, 3.5)) #single column fig
        #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
        gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

        axsPC = []
        axsPC.append( plt.subplot(gs[0,0]) )

        for ax in axsPC:
            ax.minorticks_on()

        axsPC[0].set_ylim((0.0, 1.0))
        axsPC[0].set_xlim((0,0.85))

        axsPC[0].set_ylabel(r"$c^2$")
        axsPC[0].set_xlabel(r"Normalized pressure $P/P_{FD}$")

        for i in mySet:
            plt.plot(press[i,:],c2[i,:],'k',linewidth=0.05)

        plt.savefig("csc/press_cs2_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


##################################################
# histogram
if histo:
    figH = plt.figure(figsize=(4.5, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsH = []
    axsH.append( plt.subplot(gs[0,0]) )

    for ax in axsH:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape

    c2histo = False
    mmaxhisto = True

    if c2histo:
        Ng = 70 #number of histogram bins
        c2 = np.zeros((nsamples, ))
        c2_grid = np.linspace(0.3, 1.0, Ng)

        ci = param_indices['c2max']
        c2[:] = blob_samples[:,ci]

        axsH[0].set_xlim((0.3, 1.0))
        #axsH[0].set_ylim((0.0,  1600.0))
        axsH[0].set_ylabel("#")
        axsH[0].set_xlabel(r"$c^2_{max}$")

        plt.hist(c2, bins=c2_grid)

        plt.savefig("csc/c2max_histo_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

    if mmaxhisto:
        Ng = 80 #number of histogram bins
        mmax = np.zeros((nsamples, ))
        mmax_grid = np.linspace(1.9, 2.7, Ng)

        ci = param_indices['mmax']
        mmax[:] = blob_samples[:,ci]

        axsH[0].set_xlim((1.9, 2.7))
        #axsH[0].set_ylim((0.0,  1600.0))
        axsH[0].set_ylabel("#")
        axsH[0].set_xlabel(r"$M_{max}$ (M$_{\odot}$)")

        plt.hist(mmax, bins=mmax_grid)

        plt.savefig("csc/Mmax_histo_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


##################################################
# n - trace anomaly
if flagNTa:
    figNTa = plt.figure(figsize=(3.7, 3.5)) #single column fig
    gs = plt.GridSpec(1, 1)

    axsNTa = []
    axsNTa.append( plt.subplot(gs[0,0]) )

    for ax in axsNTa:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    ta = np.zeros((nsamples, Ngrid))

    ta_grid = np.linspace(-0.0005, 0.0025, Ng) #logspace(-0.3, 4.3, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(nsat_grid):
        ci_eps = param_indices['nsat_eps_'+str(ir)]
        eps = blob_samples[:, ci_eps]

        ci_press = param_indices['nsat_p_'+str(ir)]
        press = blob_samples[:, ci_press]

        ta[:, ir] = (eps - 3. * press) * 197.3269804**3 * (0.16 * nsat / (eps + press))**4

    ta_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(nsat_grid):
        taslice = ta[:,ir]
        #ta_s = taslice[ taslice > 0.0 ] #* 2.99792458e10**2 * 1000.0 / (1.e9 * 1.602177e-12 / (1.0e-13)**3)
        ta_s = taslice
        tam = np.mean(ta_s)

        print("n= {} ta= {}".format(nsat, tam))
        print(np.amin(ta_s), np.amax(ta_s))
        #print(ta[0, ir])

        hist, bin_edges = np.histogram(ta_s, bins=ta_grid)
        ta_hist[:,ir] = (1.0*hist[:])/hist.max()


    hdata_masked = np.ma.masked_where(abs(ta_hist) == 0.0, ta_hist)

    axsNTa[0].plot([1, 46], [0, 0], color='k', linewidth=1, linestyle=':')

    axsNTa[0].set_xlim((1.0, 46.0))
    axsNTa[0].set_ylim((-0.0005, .0025))
    axsNTa[0].set_ylabel(r"Trace anomaly")
    axsNTa[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")
    #axsNTa[0].set_yscale('log')


    ta_grid = ta_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, ta_grid)

    im_ng = axsNTa[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0,
            vmax=1,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_ta_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


if flag_mr_dara:
    fig, ax = plt.subplots(7, 2, figsize=(5, 7*2.5))
    ax_flatten = [item for sublist in ax for item in sublist]
    for j in range(15):
        color="Reds"
        if j==0:
            r = [item[param_indices['GW170817_r1']] for item in blob_samples]
            mc = [item[labels.index('$\mathcal{M}_{GW170817}$')] for item in samples]
            q = [item[labels.index('$q_{GW170817}$')] for item in samples]
            m = [(1.0 + item)**0.2 / item**0.6 * mc[i] for i, item in enumerate(q)]
            ax_flatten[j].title.set_text("GW170817")
            j += 1
        elif j==1:
            r = [item[param_indices['GW170817_r2']] for item in blob_samples]
            #mc = [item[labels.index('$\mathcal{M}_{GW170817}$')] for item in samples]
            #q = [item[labels.index('$q_{GW170817}$')] for item in samples]
            #m = [(1.0 + item)**0.2 / item**0.6 * mc[i] for i, item in enumerate(q)]
            m = [m[i] * item for i, item in enumerate(q)]
            #ax_flatten[j-1].title.set_text("GW170817, lighter")
            color="Blues"
        elif j==2:
            r = [item[param_indices['r0348']] for item in blob_samples]
            m = [item[labels.index('$M_{0348}$')] for item in samples]
            ax_flatten[j-1].title.set_text("PSR J0348+0432")
        elif j==3:
            r = [item[param_indices['r0740']] for item in blob_samples]
            m = [item[labels.index('$M_{0740}$')] for item in samples]
            ax_flatten[j-1].title.set_text("PSR J0740+6620")
        elif j==4:
            r = [item[param_indices['r1702']] for item in blob_samples]
            m = [item[labels.index('$M_{1702}$')] for item in samples]
            ax_flatten[j-1].title.set_text("4U 1702−429")
        elif j==5:
            r = [item[param_indices['r6304']] for item in blob_samples]
            m = [item[labels.index('$M_{6304}$')] for item in samples]
            ax_flatten[j-1].title.set_text("NGC 6304")
        elif j==6:
            r = [item[param_indices['r6397']] for item in blob_samples]
            m = [item[labels.index('$M_{6397}$')] for item in samples]
            ax_flatten[j-1].title.set_text("NGC 6397")
        elif j==7:
            r = [item[param_indices['rM28']] for item in blob_samples]
            m = [item[labels.index('$M_{M28}$')] for item in samples]
            ax_flatten[j-1].title.set_text("M28")
        elif j==8:
            r = [item[param_indices['rM30']] for item in blob_samples]
            m = [item[labels.index('$M_{M30}$')] for item in samples]
            ax_flatten[j-1].title.set_text("M30")
        elif j==9:
            r = [item[param_indices['rX7']] for item in blob_samples]
            m = [item[labels.index('$M_{X7}$')] for item in samples]
            ax_flatten[j-1].title.set_text("47 Tuc X7")
        elif j==10:
            r = [item[param_indices['rwCen']] for item in blob_samples]
            m = [item[labels.index('$M_{\omega Cen}$')] for item in samples]
            ax_flatten[j-1].title.set_text(r"$\omega$Cen")
        elif j==11:
            r = [item[param_indices['rM13']] for item in blob_samples]
            m = [item[labels.index('$M_{M13}$')] for item in samples]
            ax_flatten[j-1].title.set_text("M13")
        elif j==12:
            r = [item[param_indices['r1724']] for item in blob_samples]
            m = [item[labels.index('$M_{1724}$')] for item in samples]
            ax_flatten[j-1].title.set_text("4U 1724-307")
        elif j==13:
            r = [item[param_indices['r1810']] for item in blob_samples]
            m = [item[labels.index('$M_{1810}$')] for item in samples]
            ax_flatten[j-1].title.set_text("SAX J1810.8-260")
        elif j==14:
            r = [item[param_indices['r0030']] for item in blob_samples]
            m = [item[labels.index('$M_{0030}$')] for item in samples]
            ax_flatten[j-1].title.set_text("PSR J0030+0451")
          
        # Creating bins
        r_min = np.min(r)
        r_max = np.max(r)
          
        m_min = np.min(m)
        m_max = np.max(m)

        r_bins = np.linspace(r_min, r_max, 50)
        m_bins = np.linspace(m_min, m_max, 50)

        # Creating plot
        #plt.hist2d(r, m, bins =[r_bins, m_bins], cmap="Reds")
        hist, xbins, ybins = np.histogram2d(r , m, bins=[r_bins, m_bins], density=True)
        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
        im = ax_flatten[j-1].imshow(np.ma.masked_where(hist == 0, hist).T/np.amax(hist), interpolation='none', origin='lower', extent=extent, cmap=color, aspect='auto')
        if j==2:
            cb = fig.colorbar(im, ax=ax[:,:], pad=0.03, aspect=50, orientation='horizontal')
            #cb = colorbar(im, loc="top", orientation="horizontal", pad=-0.3, size='3%')

        if j%2==1:
            ax_flatten[j-1].set_ylabel("Mass $M$ (M$_{\odot}$)")
        else:
            ax_flatten[j-1].yaxis.set_major_formatter(plt.NullFormatter())
        if j>12:
            ax_flatten[j-1].set_xlabel("Radius $R$ (km)")
        else:
            ax_flatten[j-1].xaxis.set_major_formatter(plt.NullFormatter())

        ax_flatten[j-1].set_xlim([9, 16])
        ax_flatten[j-1].set_ylim([0.5, 2.6])

    plt.savefig("csc/mr_data_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


if flagTriangle_td:
    lambda1 = [item[param_indices['GW170817_TD1']] for item in blob_samples]
    lambda2 = [item[param_indices['GW170817_TD2']] for item in blob_samples]

    mc = [item[labels.index('$\mathcal{M}_{GW170817}$')] for item in samples]
    q = [item[labels.index('$q_{GW170817}$')] for item in samples]
    mass1 = [(1.0 + item)**0.2 / item**0.6 * mc[i] for i, item in enumerate(q)]
    mass2 = [mass1[i] * item for i, item in enumerate(q)]

    samples_td = []
    for i, item in enumerate(samples):
        lambda_tilde = 1.23076923077 * ( (mass1[i] + 12.0 * mass2[i]) * mass1[i]**4 * lambda1[i] + (mass2[i] + 12.0 * mass1[i]) * mass2[i]**4 * lambda2[i] ) / (mass1[i] + mass2[i])**5
        samples_td.append([lambda1[i], lambda2[i], lambda_tilde, mc[i], q[i]])

    labels_td = [r"$\Lambda_1$", r"$\Lambda_2$", r"$\tilde{\Lambda}$", r"$\mathcal{M}_{GW170817}$", r"$q_{GW170817}$"]

    fig = corner.corner(
            samples_td,
            bins = 30,
            quantiles=[0.0227501, 0.158655, 0.5, 0.841345, 0.97725],
            levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-4.5),),# 1-np.exp(-8.), ),
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={"fontsize": 12},
            labels=labels_td,
            #truths=[1., 1./3., 1.]
            )

    plt.savefig("csc/triangle_td_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")
