import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.insert(1, '/home/eannala/work/koodii/multi-polytrope')
import units as cgs

import argparse
import os

##################################################
#flags
flagTriangle = True
flagTriangleCGP = False
flagMR = True
flagEP = True
flagNP = True
flagNE = True
flagNG = True
flagNC = True
flagNPP = True
flagNM = True
flagNR = True
flagNL = True
flagML = True

##################################
# Parses
def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', 
            dest='file', 
            default="",
            type=str,
            help='File name (e.g. M1_S3_PT0-s587198-w5-g200-n20000)')

    #parser.add_argument('--debug', 
    #        dest='debug', 
    #        default=False,
    #        type=bool,
    #        help='Debug mode (default: False)')

    args = parser.parse_args()

    return args

##################################################
# read chain
args = parse_cli()

prefix=args.file #'M1_S3_PT0-s587198-w5-g200-n20000'
filename = '../chains/csc/'+prefix+'run.h5'

reader = emcee.backends.HDFBackend(filename)#, name='initialization')

##################################################
# # of segments
eos_Nsegment_pos1 = prefix.find("S")
eos_Nsegment_pos2 = prefix.find("_", eos_Nsegment_pos1)
eos_Nsegment      = int(prefix[eos_Nsegment_pos1+1:eos_Nsegment_pos2])

Ngrid_pos1 = prefix.find("g")
Ngrid_pos2 = prefix.find("-", Ngrid_pos1)
Ngrid      = int(prefix[Ngrid_pos1+1:Ngrid_pos2])

phaseTransition_pos1 = prefix.find("PT")
phaseTransition_pos2 = prefix.find("-", phaseTransition_pos1)
phaseTransition      = int(prefix[phaseTransition_pos1+2:phaseTransition_pos2])

eos_model_pos1 = prefix.find("M")
eos_model_pos2 = prefix.find("_", eos_model_pos1)
eos_model = int(prefix[eos_model_pos1+1:eos_model_pos2])


#metadata of the run
#eos_Nsegment = 3
#Ngrid = 200
#phaseTransition = 0
#eos_model = 1

Ngrid2 = 100

parameters2 = []
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0,   Ngrid),
        'eps_grid':  np.logspace(2.0, 4.3, Ngrid),
        'nsat_long_grid':  np.linspace(1.1, 45.0, Ngrid), #TODO limits
        #'nsat_short_grid': np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS), Ngrid2) / cgs.rhoS, #TODO
        'nsat_short_grid': np.logspace(np.log10(1.1*cgs.rhoS), np.log10(12.0*cgs.rhoS), Ngrid2) / cgs.rhoS, #TODO
               }

#add M-R grid
#ci = n_params #current running index of the parameters list
ci = 0
for im, mass  in enumerate(param_indices['mass_grid']):
    parameters2.append('rad_'+str(im))
    param_indices['rad_'+str(im)] = ci
    ci += 1

#add eps-P grid
for ir, eps  in enumerate(param_indices['eps_grid']):
    parameters2.append('Peps_'+str(ir))
    param_indices['Peps_'+str(ir)] = ci
    ci += 1

#add nsat - p grid
for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
    parameters2.append('nsat_p_'+str(ir))
    param_indices['nsat_p_'+str(ir)] = ci
    ci += 1

#add nsat - eps grid
for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
    parameters2.append('nsat_eps_'+str(ir))
    param_indices['nsat_eps_'+str(ir)] = ci
    ci += 1

#add nsat - gamma grid
for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
    parameters2.append('nsat_gamma_'+str(ir))
    param_indices['nsat_gamma_'+str(ir)] = ci
    ci += 1

#add nsat - c^2 grid
for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
    parameters2.append('nsat_c2_'+str(ir))
    param_indices['nsat_c2_'+str(ir)] = ci
    ci += 1

#add nsat - press grid
for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
    parameters2.append('nsat_press_'+str(ir))
    param_indices['nsat_press_'+str(ir)] = ci
    ci += 1

#add nsat - mass grid
for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
    parameters2.append('nsat_mass_'+str(ir))
    param_indices['nsat_mass_'+str(ir)] = ci
    ci += 1

#add nsat - radius grid
for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
    parameters2.append('nsat_radius_'+str(ir))
    param_indices['nsat_radius_'+str(ir)] = ci
    ci += 1

#add nsat - TD grid
for ir, nsat  in enumerate(param_indices['nsat_short_grid']):
    parameters2.append('nsat_TD_'+str(ir))
    param_indices['nsat_TD_'+str(ir)] = ci
    ci += 1

#add M-TD grid
for im, mass  in enumerate(param_indices['mass_grid']):
    parameters2.append('TD_'+str(im))
    param_indices['TD_'+str(im)] = ci
    ci += 1


#tau = reader.get_autocorr_time()
#print(tau)
#burnin = int(2*np.max(tau))
#thin = int(0.5*np.min(tau))

samples = reader.get_chain(discard=0, flat=True, thin=1)
burnin = int(samples.shape[0] / (4 * samples.shape[1] * 5)) # 200000
thin = 1

samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
blob_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
print(len(blob_samples),blob_samples.shape,blob_samples[0])
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat blob shape: {0}".format(blob_samples.shape))

#all_samples = np.concatenate((
#    samples, 
#    log_prob_samples[:, None], 
#    blob_samples[:, None]
#), axis=1)

all_samples = samples

##################################################
#create labels
#labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
#labels += ["log prob", "log prior"]

#labels = [r"$a$", r"$\alpha$", r"$b$", r"$\beta$", r"$X$"]
labels = [r"$\alpha_L$", r"$\eta_L$", r"$X$"]

if eos_model == 0:
    #append gammas
    for itrope in range(eos_Nsegment-2):
        if itrope + 1 != phaseTransition:
            #parameters.append("gamma"+str(3+itrope))
            labels.append(r"$\gamma_{{{0}}}$".format((3+itrope)))

    #append transition depths
    for itrope in range(eos_Nsegment-1):
        #parameters.append("trans_delta"+str(1+itrope))
        labels.append(r"$\Delta n_{{{0}}}$".format(1+itrope))

elif eos_model == 1:
    #append chemical potential depths (NB last one will be determined)
    for itrope in range(eos_Nsegment-2):
        #parameters.append("mu_delta"+str(1+itrope))
        labels.append(r"$\Delta\mu_{{{0}}}$".format((1+itrope)))

    #append speed of sound squared (NB last one will be determined)
    for itrope in range(eos_Nsegment-2):
        #parameters.append("speed"+str(1+itrope))
        labels.append(r"$c^2_{{{0}}}$".format(1+itrope))

labels.append(r"$\mathcal{M}_{GW170817}$")
labels.append(r"$q_{GW170817}$")

labels.append(r"$M_{0432}$")
labels.append(r"$M_{6620}$")

labels.append(r"$M_{1702}$")
labels.append(r"$M_{6304}$")
labels.append(r"$M_{6397}$")
labels.append(r"$M_{M28}$")
labels.append(r"$M_{M30}$")
labels.append(r"$M_{X7}$")
labels.append(r"$M_{\omega Cen}$")
labels.append(r"$M_{M13}$")
labels.append(r"$M_{1724}$")
labels.append(r"$M_{1810}$")
labels.append(r"$M_{0437}$")

#labels.append(r"$\log L$")


##################################################
# triangle/corner plot

if flagTriangle:
    fig = corner.corner(all_samples, 
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_kwargs={"fontsize": 12},
            labels=labels
            )
    
    plt.savefig("csc/triangle_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")


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
#set up figure
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('axes', labelsize=7)

#M-R
if flagMR:
    figMR = plt.figure(figsize=(3.54, 3.0)) #single column fig
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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
        if im==118:
            print("MASS", mass)

    #print("LOL", rad.max(), np.argmax(rad, axis=0), np.argmax(rad[:,118]), rad[1039,:])
    #print(rad[637,:])
    rad_hist = np.zeros((Nr-1, Ngrid))
    for im, mass  in enumerate(param_indices['mass_grid']):

        radslice = rad[:,im]
        rads = radslice[ radslice > 0.0 ]
        radm = np.mean(rads)
        print("M= {} R= {}".format(mass,radm))

        hist, bin_edges = np.histogram(rads, bins=rad_grid)

        rad_hist[:,im] = (1.0*hist[:])/hist.max()
    print("RMAX", rad.max())
    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(rad_hist)

    hdata_masked = np.ma.masked_where(rad_hist <= 0.0, rad_hist)

    axsMR[0].set_xlim((9.0, 16.0))
    axsMR[0].set_ylim((0.5,  3.0))
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

    #causality
    rad0 = rad_grid[0]
    mass0 = rad0/(2.95*1.52)
    mass1 = mass_grid[-1]
    rad1 = mass1*2.95*1.52
    axsMR[0].fill_between([rad0, rad1], [mass0, mass1], [3.0, 3.0], color='orange', visible=True)
    #text(5.0, 2.3, "GR")

    #GR limits
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
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsEP = []
    axsEP.append( plt.subplot(gs[0,0]) )

    for ax in axsEP:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Nr = 200 #number of radius histogram bins

    rho_grid = param_indices["eps_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.logspace(-0.3, 4.0, Nr)

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


    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(press_hist)

    hdata_masked = np.ma.masked_where(press_hist <= 0.0, press_hist)

    axsEP[0].set_xlim((1.0e2, 2.0e4))
    #axsEP[0].set_ylim((1.0e33,  1.0e37))
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
            )

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    figEP.savefig("csc/eps_P_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - press
if flagNP:
    figNP = plt.figure(figsize=(3.7, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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

    axsNP[0].set_xlim((1.0, 45.0))
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

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_press_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - eps
if flagNE:
    figNE = plt.figure(figsize=(3.7, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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

    axsNE[0].set_xlim((1.0, 45.0))
    axsNE[0].set_ylim((1.0e2, 2.0e4))
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
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsNG = []
    axsNG.append( plt.subplot(gs[0,0]) )

    for ax in axsNG:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

    nsat_grid = param_indices["nsat_long_grid"]
    gamma = np.zeros((nsamples, Ngrid))
    gamma_grid = np.linspace(0.0, 10.0, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        ci = param_indices['nsat_gamma_'+str(ir)]
        gamma[:, ir] = blob_samples[:, ci]

    gamma_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_long_grid']):
        gammaslice = gamma[:,ir]
        gammas = gammaslice[ gammaslice > 0.0 ]
        gammam = np.mean( gammas )
        print("n= {} G= {}".format(nsat, gammam))

        hist_ng, bin_edges = np.histogram(gammas, bins=gamma_grid)
        gamma_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()


    hdata_masked_ng = np.ma.masked_where(gamma_hist <= 0.0, gamma_hist)

    axsNG[0].set_xlim((1.0, 45.0))
    axsNG[0].set_ylim((0.0, 10.0))
    axsNG[0].set_ylabel(r"$\Gamma$")
    axsNG[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    gamma_grid = gamma_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, gamma_grid)

    im_ng = axsNG[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_gamma_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - c^2
if flagNC:
    figNC = plt.figure(figsize=(3.7, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsNC = []
    axsNC.append( plt.subplot(gs[0,0]) )

    for ax in axsNC:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200

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


    hdata_masked_ng = np.ma.masked_where(c2_hist <= 0.0, c2_hist)

    axsNC[0].set_xlim((1.0, 45.0))
    axsNC[0].set_ylim((0.0, 1.0))
    axsNC[0].set_ylabel(r"$c^2$")
    axsNC[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    c2_grid = c2_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, c2_grid)

    im_ng = axsNC[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("csc/n_c2_"+prefix+"_B"+ str(burnin) +"_T"+ str(thin) +".pdf")

##################################################
# n - press/pFD
if flagNPP:
    figNPP = plt.figure(figsize=(3.7, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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

    axsNPP[0].set_xlim((1.0, 45.0))
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
# n - mass
if flagNM:
    figNM = plt.figure(figsize=(3.7, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

    axsNM = []
    axsNM.append( plt.subplot(gs[0,0]) )

    for ax in axsNM:
        ax.minorticks_on()

    nsamples, nblobs = blob_samples.shape
    Ng = 200 #number of mass histogram bins
    Ngrid2 = 100

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
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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
    #print(rad[637,:])
    #print(rad[1039,:])
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
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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
#M-TD
if flagML:
    figML = plt.figure(figsize=(4.1, 3.5)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

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

    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(rad_hist)

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

#    im = axs[0].imshow(
#            hdata_masked.T,
#            origin='lower',
#            interpolation='nearest',
#            cmap="Reds",
#            vmin=0.0,
#            vmax=1.0,
#            aspect='auto',
#            )

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
