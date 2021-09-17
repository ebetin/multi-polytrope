#--------------------------------------------------
# system packages
import numpy as np
import math
import os
import argparse
from scipy.stats import norm

#  problem physics modules
#--------------------------------------------------
from priors import check_uniform, check_cEFT
from structure import structureC2AGKNVwithCEFT as structure_c2 # c2 interpolation
from structure import structurePolytropeWithCEFT as structure_poly # polytropes
import units as cgs
from pQCD import nQCD, pSB_csg
from c2Interpolation import c2AGKNV

# measurements
#--------------------------------------------------
from measurements import measurement_MR
from measurements import NSK17         #1702 measurement
from measurements import SHB18_6304_He #6304 measurement
from measurements import SHB18_6397_He #6397 measurement
from measurements import SHB18_M28_He  #M28 measurement
from measurements import SHB18_M30_H   #M30 measurement
from measurements import SHB18_X7_H    #X7 measurement
from measurements import SHB18_wCen_H  #wCen measurement
from measurements import SHS18_M13_H   #M13 measurement
from measurements import NKS15_1724    #1724 measurement
from measurements import NKS15_1810    #1810 measurement
from measurements import NICER_0030

from measurements import measurement_M
from measurements import J0348
from measurements import J0740

from measurements import measurement_TD
from measurements import GW170817

# emcee stuff
#--------------------------------------------------
import sys
import emcee

# only print as a master rank
#--------------------------------------------------
from mpi4py import MPI
def mpi_print(*args):
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0: #only master rank prints
        for arg in args:
            print(arg)

#--------------------------------------------------
# cli arguments
from input_parser import parse_cli
args = parse_cli()

if args.seed != 0:
    np.random.seed(args.seed) #for reproducibility

if not os.path.exists(args.outputdir): os.mkdir(args.outputdir)

##################################################
# global flags for different run modes

# polytrope order
eos_Nsegment = args.eos_nseg

# flag for additional debug printing
debug = args.debug

# position of the 1st order transition
# NB currently works only with polytropes TODO
# after first two monotropes, 0: no phase transition
# in other words, the first two monotrope do not behave
# like a latent heat (ie. gamma != 0)
phaseTransition = args.ptrans

# interpolation model (Polytropic = 0, c_s^2 = 1)
eos_model = args.model

# cEFT model (HLPS, HLPS3, HLPS+)
ceft_model = args.ceft

# PDF for the pQCD parameter X (uniform, log-uniform, log-normal)
x_model = args.xmodel

##################################################
# extra restrictions

# calculating MR curve
flag_TOV   = True

# using GW170817 event as a constrain (NB usable if flag_TOV = True)
flag_GW    = True

# using mass measurement data (NB usable if flag_TOV = True)
flag_Mobs  = True

# using mass-radius observations (NB usable if flag_TOV = True)
flag_MRobs = True

# discarding subconformal (c_s^2 > 1/3) EoSs (default: False)
flagSubConformal = args.subconf

##################################################
# conversion factor

# Ba (= g/cm/s^2 = .1 Pa) -> Mev/fm^3
confacinv = 1000.0 / cgs.GeVfm_per_dynecm

# MeV/fm^3 -> Ba
confac = cgs.GeVfm_per_dynecm * 0.001

# (g/cm/s)^2 -> GeV [chemical potential related]
mB_nu = cgs.mB * cgs.erg_per_kev * 1.0e-6

# Inversed satuaration density (cm^3/g)
rhoS_inv = 1.0 / cgs.rhoS

# Inversed speed of sound square (s^2/cm^2)
c2inv = 1.0 / cgs.c**2

# Mev/fm^3 -> g/cm^3
confacC2inv = confac * c2inv

# Inversed pi square
pi2inv = 1.0 / math.pi**2

# 1/3
inv3 = 0.3333333333333333333333333

##################################################
# Perturbative QCD parameter, see e.g. Fraga et al. (2014, arXiv:1311.5154) for details
# Transition (matching) chemical potential (GeV) between interpolated and pQCD parts
muQCD = 2.74  # 2.739512831322249

##################################################
#auto-generated parameter names for c2 interpolation

#cEFT parameters
if ceft_model == 'HLPS':
    parameters = ["alphaL", "etaL"]
    gamma  = 4.0 * inv3 # unitless
elif ceft_model == 'HLPS3':
    parameters = ["alphaL", "etaL", "gamma"]
elif ceft_model == 'HLPS+':
    parameters = ["alphaL", "etaL", "gammaL", "zetaL", "rho0"]

#pQCD parameters
parameters.append("X")

#Interpolation parameters

if eos_model == 0:
    #append gammas
    for itrope in range(eos_Nsegment-2):
        if itrope + 1 != phaseTransition:
            parameters.append("gamma"+str(3+itrope))

    #append transition depths
    for itrope in range(eos_Nsegment-1):
        parameters.append("trans_delta"+str(1+itrope))

elif eos_model == 1:
    #append chemical potential depths (NB last one will be determined)
    for itrope in range(eos_Nsegment-2):
        parameters.append("mu_delta"+str(1+itrope))

    #append speed of sound squared (NB last one will be determined)
    for itrope in range(eos_Nsegment-2):
        parameters.append("speed"+str(1+itrope))

#GW170817
parameters.append("chrip_mass_GW170817")
parameters.append("mass_ratio_GW170817")

#finally add individual object masses (needed for measurements)
parameters.append("mass_0432")
parameters.append("mass_6620")

parameters.append("mass_1702")

parameters.append("mass_6304")
parameters.append("mass_6397")
parameters.append("mass_M28")
parameters.append("mass_M30")
parameters.append("mass_X7")
parameters.append("mass_wCen")
parameters.append("mass_M13")
parameters.append("mass_1724")
parameters.append("mass_1810")

parameters.append("mass_0030")


mpi_print("Parameters to be sampled are:")
mpi_print(parameters)


n_params = len(parameters)
prefix = "chains/M{}_S{}_PT{}-s{}-w{}-g{}-n{}-{}-{}".format(eos_model, eos_Nsegment, phaseTransition, args.seed, args.walkers, args.ngrid, args.nsteps, ceft_model, x_model)


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

parameters2 = []

Ngrid = args.ngrid
Ngrid2 = 109#1081 # TODO value?
# TODO check short!
param_indices = {
        'mass_grid':       np.linspace(0.5, 3.0,   Ngrid),
        'eps_grid':        np.logspace(2.0, 4.3, Ngrid),
        'nsat_long_grid':  np.linspace(1.0, 50.0, Ngrid),
        'nsat_short_grid': np.linspace(1.2, 12., Ngrid2),
        #'nsat_short_grid': np.logspace(math.log10(1.2*cgs.rhoS), math.log10(12.0*cgs.rhoS), Ngrid2) * rhoS_inv, # TODO only TOV
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


#add mmax parameters
parameters2.append('mmax')
param_indices['mmax'] = ci
parameters2.append('mmax_rad')
param_indices['mmax_rad'] = ci+1
parameters2.append('mmax_rho')
param_indices['mmax_rho'] = ci+2
parameters2.append('mmax_press')
param_indices['mmax_press'] = ci+3
parameters2.append('mmax_edens')
param_indices['mmax_edens'] = ci+4
parameters2.append('mmax_ppFD')
param_indices['mmax_ppFD'] = ci+5
parameters2.append('mmax_c2')
param_indices['mmax_c2'] = ci+6
parameters2.append('mmax_gamma')
param_indices['mmax_gamma'] = ci+7

# crust-core transition (mass) density (g/cm^3)
parameters2.append('rho_cc')
param_indices['rho_cc'] = ci+8

# max squared speed of sound
parameters2.append('c2max')
param_indices['c2max'] = ci+9

if eos_model == 0:
    # first gamma
    parameters2.append('gamma1')
    param_indices['gamma1'] = ci+9
    # second gamma
    parameters2.append('gamma2')
    param_indices['gamma2'] = ci+10
elif eos_model == 1:
    # solved chemical potential (Gev)
    parameters2.append('mu_param')
    param_indices['mu_param'] = ci+9
    # solved squared speed of sound
    parameters2.append('c2_param')
    param_indices['c2_param'] = ci+10

ci = ci+11

#radii
parameters2.append('r1702')
param_indices['r1702'] = ci
parameters2.append('r6304')
param_indices['r6304'] = ci+1
parameters2.append('r6397')
param_indices['r6397'] = ci+2
parameters2.append('rM28')
param_indices['rM28'] = ci+3
parameters2.append('rM30')
param_indices['rM30'] = ci+4
parameters2.append('rX7')
param_indices['rX7'] = ci+5
parameters2.append('rwCen')
param_indices['rwCen'] = ci+6
parameters2.append('rM13')
param_indices['rM13'] = ci+7
parameters2.append('r1724')
param_indices['r1724'] = ci+8
parameters2.append('r1810')
param_indices['r1810'] = ci+9
parameters2.append('r0030')
param_indices['r0030'] = ci+10
parameters2.append('r6620')
param_indices['r6620'] = ci+11

mpi_print("Parameters to be only stored (blobs):")
mpi_print(len(parameters2))
n_blobs = len(parameters2)

######################################################################
from scipy.optimize import brentq
from pQCD import nQCD_nu_wo_errors

def rho_qcd(x, mu):
    return nQCD_nu_wo_errors(mu, x)

# Smallest realistic value for the scale parameter X of the NNNLO pQCD calculations
# (For detail, see Fraga et al. (2014, arXiv:1311.5154) as well)
x_min = brentq(rho_qcd, .5, 1., args=(muQCD,))
# x_min_ln = math.log(x_min)
######################################################################

#5D distribution of the cEFT fit parameters
if ceft_model == 'HLPS+':
    from scipy.stats import multivariate_normal as mvn
    import h5py

    f1 = h5py.File('CET/CEFT_prior_Lambda-500_rhoMin-80_rhoMax-195.h5', 'r')

    cEFT_means = f1['means']
    cEFT_cov = f1['covariances']
    cEFT_weights = f1['weights']

    cEFT_mvn = []
    for k in range(cEFT_means.shape[0]):
        cEFT_mvn.append( mvn( cEFT_means[k], cEFT_cov[k] ) )

# probability function
linf = np.inf

##################################################
# Prior function; changes from [0,1] to whatever physical lims

def myprior(cube):
    lps = np.zeros_like(cube)

    ##################################################################################
    # Parameters of the cEFT EoS
    if ceft_model == 'HLPS':
        lps[0] = check_uniform(cube[0], 1.17, 1.61) #alphaL [unitless]
        lps[1] = check_uniform(cube[1], 0.6,  1.15) #etaL [unitless]

        ci = 2
    elif ceft_model == 'HLPS3':
        lps[0] = check_uniform(cube[0], 1.17, 1.61) #alphaL [unitless]
        lps[1] = check_uniform(cube[1], 0.6,  1.15) #etaL [unitless]
        lps[2] = check_uniform(cube[2], 1.2,  2.5) #etaL [unitless]

        ci = 3
    elif ceft_model == 'HLPS+':
        params = cube[:5]
        #5D distribution
        tmp = sum( map( lambda x, y: x * y.pdf(params), cEFT_weights, cEFT_mvn ) ) 
        if tmp == 0:
            lps[0] = -linf
        else:
            lps[0] = math.log( tmp )

        ci = 5

    ##################################################################################
    # Unitless scale parameter (X) of the NNNLO pQCD calculations, arXix:2103.05658
    # (For detail, see Fraga et al. (2014, arXiv:1311.5154) as well)

    if cube[ci] <= x_min:
        # Returns -inf if rhoo <= 0
        lps[ci] = -linf
    else:
        if x_model = 'log_normal':
            # Truncated log-normal distribution demanding X > x_min
            # Params: mu = 0.576702101863589, sigma = 0.8075029237121397
            # Median = 2, mean = 2.7238, var = 5.73697, mode = 0.927411
            # NB assumed that muQCD = 2.74 GeV!
            lnc = math.log( cube[ci] )
            lps[ci] = -0.5833457860175315 - lnc - 0.7667994583650043 * (lnc - 0.576702101863589)**2
        elif x_model = 'log_uniform':
            # Log-uniform distribution from x_min to 10
            lnc = math.log( cube[ci] )
            # lps[ci] = -lnc - math.log( 2.302585092994046 - x_min_ln )
            lps[ci] = 0.991987995279329 - lnc  # NB assuming muQCD = 2.74 GeV!
        elif x_model = 'uniform':
            # Uniform distribution from x_min to 10
            lps[c1] = check_uniform(cube[ci], x_min, 10.)

    ci += 1

    ##################################################################################
    # Interpolation parameters
    # eos_models: polytropic (0) and c2 (1)

    if eos_model == 0:
        # Polytropic exponents excluding the first two ones
        for itrope in range(eos_Nsegment-2):
            if itrope + 1 != phaseTransition:
                if debug:
                    mpi_print("prior for gamma from cube #{}".format(ci))
                lps[ci] = check_uniform(cube[ci], 0.0, 10.0)  #gamma_i [unitless]
                ci += 1


        # Lengths of the first N-1 monotropes (N = # of polytropes)
        for itrope in range(eos_Nsegment-1):
            if debug:
                mpi_print("prior for trans from cube #{}".format(ci))
            lps[ci] = check_uniform(cube[ci], 0.0, 43.0)  #delta_ni [rhoS]
            ci += 1
    elif eos_model == 1:
        # Chemical potential intervalss
        for itrope in range(eos_Nsegment-2):
            if debug:
                mpi_print("prior for mu_delta from cube #{}".format(ci))
            lps[ci] = check_uniform(cube[ci], 0.0, 1.8)  #delta_mui [GeV]
            ci += 1

        # Matching speed of sound squared excluding the last (=pQCD) one
        for itrope in range(eos_Nsegment-2):
            if debug:
                mpi_print("prior for c^2 from cube #{}".format(ci))
            if flagSubConformal:
                lps[ci] = check_uniform(cube[ci], 0.0, inv3)   #c_i^2 [unitless]
            else:
                lps[ci] = check_uniform(cube[ci], 0.0, 1.0)  #c_i^2 [unitless]
            ci += 1

    ##################################################################################
    # TD measurements

    # Chirp-mass distibution of the GW170817 event, see arXiv:1805.11579
    # 1.186 +- 0.001 M_Sun (90%), assumed to be normal
    lps[ci] = 6.486468145459291 - 1.3527717270480054e6 * ( cube[ci] - 1.186 )**2

    # Mass ratio (GW170817), m_2 / m_1 s.t. m_1 >= m_2
    lps[ci+1] = check_uniform(cube[ci+1], 0.0, 1.0)

    ci += 2

    ##################################################################################
    # M measurements

    # PSR J0348+0432, M = 2.01 +- 0.04 M_Sun (68.27%), normal distro, arXiv:1304.6875
    lps[ci]   = 2.2999372916635283 - 312.5 * ( cube[ci] - 2.01 )**2

    # PSR J0740+6620, M = 2.08 +- 0.07 M_Sun (68.3%), normal distro, arXiv:2104.00880
    lps[ci+1] = 1.7403215037281052 - 102.0408163265306 * ( cube[ci+1] - 2.08 )**2

    ci += 2

    ##################################################################################
    # M-R measurements

    # 4U 1702-429, arXiv:1709.09120
    lps[ci]  = check_uniform(cube[ci], 1.0, 2.5)

    # arXiv:1709.05013
    # NGC 6304, helium atmosphere
    lps[ci+1]  = check_uniform(cube[ci+1], 0.5, 2.7)
    # NGC 6397, helium atmosphere
    lps[ci+2]  = check_uniform(cube[ci+2], 0.5, 2.0)
    # M28, helium atmosphere
    lps[ci+3]  = check_uniform(cube[ci+3], 0.5, 2.8)
    # M30, hydrogen atmosphere
    lps[ci+4]  = check_uniform(cube[ci+4], 0.5, 2.5)
    # 47 Tuc X7, hydrogen atmosphere
    lps[ci+5]  = check_uniform(cube[ci+5], 0.5, 2.7)
    # wCen, hydrogen atmosphere
    lps[ci+6]  = check_uniform(cube[ci+6], 0.5, 2.5)

    # M13, hydrogen atmosphere, arXiv:1803.00029
    lps[ci+7]  = check_uniform(cube[ci+7], 0.8, 2.4)

    # arXiv:1509.06561
    # 4U 1724-307
    lps[ci+8]  = check_uniform(cube[ci+8], 0.8, 2.5)
    # SAX J1810.8-260
    lps[ci+9]  = check_uniform(cube[ci+9], 0.8, 2.5)

    # J0030+0451, arXiv:TODO
    lps[ci+10] = check_uniform(cube[ci+10], 1.0129494423462766, 2.2792157996697235)

    return np.sum(lps)


################################################
#constant variables

# Transition ("matching") densities (g/cm^3)
# Here one only defines the starting point of cEFT
# The code fill solve the matching density between the crust and the core
# The starting point of pQCD calculations will be defined after this
if ceft_model == 'HLPS':
    trans_points = [1.1 * cgs.rhoS]  # g/cm^3
    trans_points_s = [1.1]  # n_s
elif ceft_model == 'HLPS3':
    trans_points = [1.1 * cgs.rhoS]  # g/cm^3
    trans_points_s = [1.1]  # n_s
elif ceft_model == 'HLPS+':
    trans_points = [1.21875 * cgs.rhoS]  # g/cm^3
    trans_points_s = [1.21875]  # n_s

# Collection of constant variables
if ceft_model == 'HLPS':
    const_params = trans_points_s + [gamma] + [muQCD]
elif ceft_model == 'HLPS3' or ceft_model == 'HLPS+':
    const_params = trans_points_s + [muQCD]

################################################

icalls = 0
def myloglike(cube):
    """
        General likelihood function that builds the EoS and solves TOV-structure
        equations for it.

        Parameters from 0:nDim are sampled, everything beyond that are just 
        carried along for analysis purposes.

        ## cEFT parameters:
        0 alpha
        1 eta
        (2 gamma)
        (3 zeta)
        (4 rho0)

        #pQCD parameters:
        2/3/5 X

        #interpolation parameters:
        matching polytropic exponents
        matching density intervals
        OR
        matching chemical-potential intercals
        matching speed of sound squared

        # Measurement parameters:
        mass of individual objects
        TD data from GW170817 event
        mass-radius data


    """
    ##################################################
    # initialization
    blobs = np.zeros(n_blobs)
    logl = 0.0 #total likelihood

    if debug:
        global icalls
        icalls += 1
        mpi_print(icalls, cube)

    ################################################## 
    # cEFT low-density EOS parameters
    if ceft_model == 'HLPS' or ceft_model == 'HLPS3':
        if debug:
            mpi_print("Checking cEFT")
        if not(check_cEFT(cube[0], cube[1])):
            logl = -linf
            return logl, blobs

    ################################################## 
    # nuclear EoS

    # Transition ("matching") densities (g/cm^3)
    # Initially, this quantity only contains the starting point of the cEFT phase
    trans = trans_points[:]

    ################################################## 
    # interpolated EoS
 
    # general running index that maps cube array to real EoS parameters
    if ceft_model == 'HLPS':
        ci = 3
    elif ceft_model == 'HLPS3':
        ci = 4
    elif ceft_model == 'HLPS+':
        ci = 6

    if eos_model == 0:
        # Polytropic exponents excluding the first two ones
        gammas = []  
        for itrope in range(eos_Nsegment-2):
            if itrope + 1 != phaseTransition:
                if debug:
                    mpi_print("loading gamma from cube #{}".format(ci))
                gammas.append(cube[ci])
                ci += 1
            else:
                gammas.append(0.0)

        # Transition ("matching") densities (g/cm^3)
        for itrope in range(eos_Nsegment-1):
            if debug:
                mpi_print("loading trans from cube #{}".format(ci))
            trans.append(trans[-1] + cgs.rhoS * cube[ci]) 
            ci += 1
    elif eos_model == 1:
        # Matching chemical potentials (GeV)
        mu_deltas = []  
        for itrope in range(eos_Nsegment-2):
            if debug:
                mpi_print("loading mu_deltas from cube #{}".format(ci))
            mu_deltas.append(cube[ci])
            ci += 1

        speed2 = []
        # Speed of sound squareds (unitless)
        for itrope in range(eos_Nsegment-2):
            if debug:
                mpi_print("loading speed2 from cube #{}".format(ci))
            speed2.append(cube[ci]) 
            ci += 1

    ################################################## 
    # low-density cEFT EoS

    # Parameters of the cEFT EoS
    # gamma  = 4.0 / 3.0  # unitless
    alphaL = cube[0]  # untiless
    etaL   = cube[1]  # untiless

    if ceft_model == 'HLPS':
        gamma = 4.0 * inv3  # unitless
        lowDensity = [gamma, alphaL, etaL]
    elif ceft_model == 'HLPS3':
        gamma = cube[2]  # unitless
        lowDensity = [gamma, alphaL, etaL]
    elif ceft_model == 'HLPS+':
        gamma = cube[2]  # unitless
        zetaL = cube[3]  # unitless
        rho0 = cube[4]   # unitless
        lowDensity = [gamma, alphaL, etaL, zetaL, rho0]


    ################################################## 
    # high-density pQCD EoS

    # Perturbative QCD parameters, see Fraga et al. (2014, arXiv:1311.5154) for details
    if ceft_model == 'HLPS':
        X = cube[2]
    elif ceft_model == 'HLPS3':
        X = cube[3]
    elif ceft_model == 'HLPS+':
        X = cube[5]

    highDensity = [muQCD, X]

    # Check that last transition (matching) point is large enough
    if debug:
        mpi_print("Checking nQCD")

    trans_qcd = nQCD(muQCD, X) * cgs.mB
    if trans_qcd <= trans[-1]:
        logl = -linf
        return logl, blobs

    ##################################################
    # build neutron star structure

    # Construct the EoS
    if debug:
        mpi_print("Structure...")
    if eos_model == 0:
        struc = structure_poly(gammas, trans, lowDensity, highDensity, CEFT_model = ceft_model)
    elif eos_model == 1:
        struc = structure_c2(mu_deltas, speed2, trans, lowDensity, highDensity, approximation = True, CEFT_model = ceft_model)

    # Is the obtained EoS realistic, e.g. causal?
    if debug:
        mpi_print("Checking structure")
    if not struc.realistic:
        logl = -linf
        return logl, blobs

    # Discard subconformal EoSs
    if flagSubConformal:
        if 3.0 * struc.speed2max > 1.0:
            logl = -linf
            return logl, blobs

    ################################################## 
    # measurements & constraints

    # Chirp mass (GW170817) [Msun]
    if flag_GW and flag_TOV:
        # Masses GW170817
        mass1_GW170817 = (1.0 + cube[ci+1])**0.2 / (cube[ci+1])**0.6 * cube[ci]
        mass2_GW170817 = mass1_GW170817 * cube[ci+1]

    ci += 2

    # solve structure 
    if debug:
        mpi_print("params", cube)
        mpi_print("TOV...")

    if flag_TOV:
        if flag_GW:  # with tidal deformabilities
            struc.tov(l=2, m1=mass1_GW170817*cgs.Msun, m2=mass2_GW170817*cgs.Msun, rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)
        else:  # without
            struc.tov(rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)

        # Maximum mass [Msun]
        mmax = struc.maxmass

        # Are the objects in event GW170817 too heavy?
        if flag_GW:
            if mass1_GW170817 > mmax or mass2_GW170817 > mmax:
                logl = -linf
                return logl, blobs

    # Mass measurement of PSR J0348+0432 from Antoniadis et al 2013 arXiv:1304.6875
    # and PSR J0740+6620 from Cromartie et al 2019 arXiv:1904.06759
    if flag_Mobs and flag_TOV:
        m0432 = cube[ci]
        m6620 = cube[ci+1]

        if m0432 > mmax or m6620 > mmax:
            logl = -linf
            return logl, blobs

    ci += 2

    # Mass-radius measurements
    if flag_MRobs and flag_TOV:
        # Masses [Msun]
        mass_1702 = cube[ci]
        mass_6304 = cube[ci+1]
        mass_6397 = cube[ci+2]
        mass_M28  = cube[ci+3]
        mass_M30  = cube[ci+4]
        mass_X7   = cube[ci+5]
        mass_wCen = cube[ci+6]
        mass_M13  = cube[ci+7]
        mass_1724 = cube[ci+8]
        mass_1810 = cube[ci+9]
        mass_0030 = cube[ci+10]

        # All stars have to be lighter than the max mass limit
        masses = [ mass_1702, mass_6304, mass_6397, mass_M28,
                   mass_M30, mass_X7, mass_wCen, mass_M13,
                   mass_1724, mass_1810, mass_0030, ]

        if any(m > mmax for m in masses):
            logl = -linf
            return logl, blobs

        # 4U 1702-429 from Nattila et al 2017, arXiv:1709.09120
        rad_1702 = struc.radius_at(mass_1702)
        logl += measurement_MR(mass_1702, rad_1702, NSK17)
        #logl += gaussian_MR(mass_1702, rad_1702, NSK17)

        # NGC 6304 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_6304 = struc.radius_at(mass_6304)
        logl += measurement_MR(mass_6304, rad_6304, SHB18_6304_He)

        # NGC 6397 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_6397 = struc.radius_at(mass_6397)
        logl += measurement_MR(mass_6397, rad_6397, SHB18_6397_He)

        # M28 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_M28 = struc.radius_at(mass_M28)
        logl += measurement_MR(mass_M28, rad_M28, SHB18_M28_He)

        # M30 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_M30 = struc.radius_at(mass_M30)
        logl += measurement_MR(mass_M30, rad_M30, SHB18_M30_H)

        # X7 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_X7 = struc.radius_at(mass_X7)
        logl += measurement_MR(mass_X7, rad_X7, SHB18_X7_H)

        # wCen with H atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_wCen = struc.radius_at(mass_wCen)
        logl += measurement_MR(mass_wCen, rad_wCen, SHB18_wCen_H)

        # M13 with H atmosphere from Shaw et al 2018, arXiv:1803.00029
        rad_M13 = struc.radius_at(mass_M13)
        logl += measurement_MR(mass_M13, rad_M13, SHS18_M13_H)

        # 4U 1724-307 from Natiila et al 2016, arXiv:1509.06561
        rad_1724 = struc.radius_at(mass_1724)
        logl += measurement_MR(mass_1724, rad_1724, NKS15_1724)

        # SAX J1810.8-260 from Natiila et al 2016, arXiv:1509.06561
        rad_1810 = struc.radius_at(mass_1810)
        logl += measurement_MR(mass_1810, rad_1810, NKS15_1810)

        # PSR J0030+0451 (NICER), arXiv:TODO
        rad_0030 = struc.radius_at(mass_0030)
        logl += measurement_MR(mass_0030, rad_0030, NICER_0030)

        # PSR J0740+6620 (NICER), arXiv:TODO
        rad_6620 = struc.radius_at(m6620)
        #logl += measurement_MR(m6620, rad_6620, NICER_6620) #TODO add measurement data

        if np.isneginf(logl):
            return logl, blobs

    # GW170817, tidal deformability
    if flag_GW and flag_TOV:
        if struc.TD > 1600.0 or struc.TD2 > 1600.0 or struc.TD < 0.0 or struc.TD2 < 0.0:
            logl = -linf
            return logl, blobs
  
        logl += measurement_TD(struc.TD, struc.TD2, GW170817)

    ic = 0

    # build M-R curve
    if debug:
        ic = param_indices['rad_0']  # starting index
        mpi_print("building M-R curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['rad_' + str(im)]  # this is the index pointing to correct position in cube

        if flag_TOV:
            blobs[ic] = struc.radius_at(mass) 
        else:
            blobs[ic] = 0.0

        if debug:
            mpi_print("im = {}, mass = {}, rad = {}, ic = {}".format(im, mass, blobs[ic], ic))

    # build eps-P curve
    if debug:
        ic = param_indices['Peps_0']  # starting index
        mpi_print("building eps-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, eps in enumerate(param_indices['eps_grid']):
        ic = param_indices['Peps_'+str(ir)]  # this is the index pointing to correct position in cube
        blobs[ic] = struc.eos.pressure_edens( eps * confacC2inv ) * confacinv

        if debug:
            mpi_print("ir = {}, eps = {}, P = {}, ic = {}".format(ir, eps, blobs[ic], ic))

    ############################################
    # build nsat long

    n_pieces = len(struc.eos.pieces)
    if n_pieces == 4:
        eos_interp = struc.eos.pieces[2]  # one cc-trans point
    elif n_pieces == 5:
        eos_interp = struc.eos.pieces[3]  # two cc-trans points

    def psb_cgs_new(mub):
        return 0.75 * pi2inv * (mub * inv3)**4 * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm

    # Calculating certain blobs can be done faster if the c2-interpolation is used
    if isinstance(eos_interp, c2AGKNV):
        interp_index1 = np.where(param_indices['nsat_long_grid'] <= eos_interp.listRhoLong[-1]*rhoS_inv)[0]
        interp_index2 = np.where( (param_indices['nsat_long_grid'] <= trans_qcd*rhoS_inv) & (param_indices['nsat_long_grid'] > eos_interp.listRhoLong[-1]*rhoS_inv) )[0]
        pqcd_index = np.where(param_indices['nsat_long_grid'] > trans_qcd*rhoS_inv)
        interp_rhoB1 = param_indices['nsat_long_grid'][interp_index1] * cgs.rhoS
        interp_rhoB2 = param_indices['nsat_long_grid'][interp_index2] * cgs.rhoS
        pqcd_rhoB = param_indices['nsat_long_grid'][pqcd_index]

        # cEFT part XXX NB the crust part has NOT been separeted from cEFT (i.e. keep n_B > n_s/2)!
        ir = interp_index1[0]
        ii1_len = len(interp_index1)
        icP = param_indices['nsat_p_'+str(ir)]
        icE = param_indices['nsat_eps_'+str(ir)]
        icG = param_indices['nsat_gamma_'+str(ir)]
        icC2 = param_indices['nsat_c2_'+str(ir)]
        icPfd = param_indices['nsat_press_'+str(ir)]

        press = np.interp(interp_rhoB1, eos_interp.listRhoLong, eos_interp.listPLong)
        edens = cgs.c**2 * np.interp(interp_rhoB1, eos_interp.listRhoLong, eos_interp.listELong)
        speed2 = np.reciprocal( np.interp(interp_rhoB1, eos_interp.listRhoLong, eos_interp.listC2invLong) )
        muB = ( press + edens ) * mB_nu / interp_rhoB1

        blobs[np.arange(icP,icP+ii1_len)] = press * confacinv
        blobs[np.arange(icE,icE+ii1_len)] = edens * confacinv
        blobs[np.arange(icG,icG+ii1_len)] = edens * speed2 / press
        blobs[np.arange(icC2,icC2+ii1_len)] = speed2
        blobs[np.arange(icPfd,icPfd+ii1_len)] = press / ( 0.75 * pi2inv * (muB * inv3)**4 * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm )

        # c2-interpolated part
        ir = interp_index2[0]
        ii2_len = len(interp_index2)
        icP = param_indices['nsat_p_'+str(ir)]
        icE = param_indices['nsat_eps_'+str(ir)]
        icG = param_indices['nsat_gamma_'+str(ir)]
        icC2 = param_indices['nsat_c2_'+str(ir)]
        icPfd = param_indices['nsat_press_'+str(ir)]

        press = np.interp(interp_rhoB2, eos_interp.listRhoLongHigh, eos_interp.listPLongHigh)
        edens = cgs.c**2 * np.interp(interp_rhoB2, eos_interp.listRhoLongHigh, eos_interp.listELongHigh)
        speed2 = np.reciprocal( np.interp(interp_rhoB2, eos_interp.listRhoLongHigh, eos_interp.listC2invLongHigh) )
        muB = ( press + edens ) * mB_nu / interp_rhoB2

        blobs[np.arange(icP,icP+ii2_len)] = press * confacinv
        blobs[np.arange(icE,icE+ii2_len)] = edens * confacinv
        blobs[np.arange(icG,icG+ii2_len)] = edens * speed2 / press
        blobs[np.arange(icC2,icC2+ii2_len)] = speed2
        blobs[np.arange(icPfd,icPfd+ii2_len)] = press / ( 0.75 * pi2inv * (muB * inv3)**4 * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm )

        # pQCD part
        ir0 = interp_index2[-1] + 1
        for ir, nsat in enumerate(pqcd_rhoB):#param_indices['nsat_long_grid']):
            #these are the indecies pointing to correct positions in cube
            icP = param_indices['nsat_p_'+str(ir0+ir)]
            icE = param_indices['nsat_eps_'+str(ir0+ir)]
            icG = param_indices['nsat_gamma_'+str(ir0+ir)]
            icC2 = param_indices['nsat_c2_'+str(ir0+ir)]
            icPfd = param_indices['nsat_press_'+str(ir0+ir)]

            rhoB = cgs.rhoS * nsat
            press, edens, speed2 = struc.eos.press_edens_c2( rhoB )
            edens = edens * cgs.c**2
            muB = ( press + edens ) * mB_nu / rhoB

            blobs[icP] = press * confacinv
            blobs[icE] = edens * confacinv
            blobs[icG] = edens * speed2 / press
            blobs[icC2] = speed2
            blobs[icPfd] = press / psb_cgs_new( muB )
    else:
        for ir, nsat in enumerate(param_indices['nsat_long_grid']):
            #these are the indecies pointing to correct positions in cube
            icP = param_indices['nsat_p_'+str(ir)]
            icE = param_indices['nsat_eps_'+str(ir)]
            icG = param_indices['nsat_gamma_'+str(ir)]
            icC2 = param_indices['nsat_c2_'+str(ir)]
            icPfd = param_indices['nsat_press_'+str(ir)]

            rhoB = cgs.rhoS * nsat
            press, edens, speed2 = struc.eos.press_edens_c2( rhoB )
            edens = edens * cgs.c**2
            muB = ( press + edens ) * mB_nu / rhoB

            blobs[icP] = press * confacinv
            blobs[icE] = edens * confacinv
            blobs[icG] = edens * speed2 / press
            blobs[icC2] = speed2
            blobs[icPfd] = press / psb_cgs_new( muB )
    ############################################

    #build nsat short
    for ir, nsat in enumerate(param_indices['nsat_short_grid']):
        #these are the indecies pointing to correct positions in cube
        icM = param_indices['nsat_mass_'+str(ir)]
        icR = param_indices['nsat_radius_'+str(ir)]
        icT = param_indices['nsat_TD_'+str(ir)]

        if ir <= struc.indexM:
            blobs[icM] = struc.mass[ir]
            blobs[icR] = struc.rad[ir]
            blobs[icT] = struc.TDlist[ir]
        else:
            blobs[icM] = 0.0
            blobs[icR] = 0.0
            blobs[icT] = 0.0

    #build M-TD curve
    if debug:
        ic = param_indices['TD_0'] #starting index
        mpi_print("building M-TD curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['TD_' + str(im)] #this is the index pointing to correct position in cube

        if flag_TOV:
            blobs[ic] = struc.TD_at(mass)
        else:
            blobs[ic] = 0.0

        if debug:
            mpi_print("im = {}, mass = {}, TD = {}, ic = {}".format(im, mass, blobs[ic], ic))

    # Variables at M = M_max
    ic = param_indices['mmax'] #this is the index pointing to correct position in cube
    rhoM = struc.maxmassrho # central number density
    press = struc.eos.pressure( rhoM ) # central pressure
    edens = struc.eos.edens ( rhoM ) * cgs.c**2 # central energy density
    muB = ( press + edens ) * mB_nu / rhoM # central chemical potential
    blobs[ic] = mmax # max mass [M_sun]
    blobs[ic+1] = struc.maxmassrad # rad(Mmax) [km]
    blobs[ic+2] = rhoM * rhoS_inv # rho(Mmax) [rhoS]
    blobs[ic+3] = press * confacinv # pressure [MeV/fm^3]
    blobs[ic+4] = edens * confacinv  # edens [MeV/fm^3]
    blobs[ic+5] = press / pSB_csg( muB ) # normalized press [1]
    blobs[ic+6] = struc.eos.speed2_rho ( rhoM ) # speed of sound [1]
    blobs[ic+7] = struc.eos.gammaFunction ( rhoM ) # gamma [1]

    # Crust-core transition (mass) density (g/cm^3)
    blobs[ic+8] = struc.rho_cc

    # Max speed of sound square
    ic = param_indices['c2max'] #this is the index pointing to correct position in cube
    blobs[ic] = struc.speed2max

    if eos_model == 0:
        # First and second gamma
        blobs[ic+1] = struc.gammasSolved[0]
        blobs[ic+2] = struc.gammasSolved[1]
    elif eos_model == 1:
        # Solved mu and c2 (starting point of the last segment)
        blobs[ic+1] = struc.muSolved
        blobs[ic+2] = struc.c2Solved

    #radii
    ic = param_indices['r1702'] #this is the index pointing to correct position in cube
    blobs[ic]    = rad_1702
    blobs[ic+1]  = rad_6304
    blobs[ic+2]  = rad_6397
    blobs[ic+3]  = rad_M28
    blobs[ic+4]  = rad_M30
    blobs[ic+5]  = rad_X7
    blobs[ic+6]  = rad_wCen
    blobs[ic+7]  = rad_M13
    blobs[ic+8]  = rad_1724
    blobs[ic+9]  = rad_1810
    blobs[ic+10] = rad_0030
    blobs[ic+11] = rad_6620

    return logl, blobs

# combine likelihood and prior ~ posterior
def lnprob(cube):
    lp = myprior(cube)
    if not np.isfinite(lp):
        return -np.inf, np.zeros(n_blobs)
    else:
        ll, blobs = myloglike(cube)
        return lp + ll, blobs

##################################################
# initial point(s) for MCMC

# Finding a (semi-)random staring point
# To be able to cut the length of the burn-in period, this point is close the max probability denisty
def initial_point(nwalkers, ndim):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    pi_vec = [None] * nwalkers

    # constants
    radius_cut = 17.  # max radius at M_TOV (km)
    mass_cut = 2.01  # smallest acceptable TOV mass (M_sun)

    # correct cEFT function
    if ceft_model == 'HLPS' or ceft_model == 'HLPS3':
        from polytropes import cEFT
    elif ceft_model == 'HLPS+':
        from polytropes import cEFT_r4

    # Splits vector (vec) into n subvectors
    def splitter(vec, n):
        q, r = divmod(len(vec), n)

        res = [None] * n
        for i in range(n):
            a = i*q + min(r, i)
            b = (i+1)*q + min(r, i+1)
            res[i] = vec[a:b]

        return res

    # creates a randomized set of ndim points around the starting point (pinit)
    def init_rand(pinit, ndim):
        pi = pinit
        while not np.isfinite( lnprob(pi)[0] ):
            pi = pinit + 0.001 * np.random.randn(ndim)

        return pi

    # generates a set of HLPS(3) params
    def ceft_params_hlps():
        aL = np.random.uniform(1.17, 1.61)
        eL = np.random.uniform(0.6,  1.15)

        return aL, eL

    # generates a set of HLPS+ params and the corresponding probability density
    def ceft_prob_hlpsp():
        aL    = np.random.uniform(0.88, 0.92)
        eL    = np.random.uniform(0.38, 0.42)
        gamma = np.random.uniform(1.88, 1.98)
        zL = np.random.uniform(0.075, 0.085)
        rr0  = np.random.uniform(0.90, 0.94)

        params_ceft = [aL, eL, gamma, zL, rr0]

        # probability denisty
        prob = sum( map( lambda x, y: x * y.pdf(params_ceft), cEFT_weights, cEFT_mvn ) )

        return prob, aL, eL, gamma, zL, rr0

    again = True
    while again:
        # using only the main core
        if rank == 0:
            again = False

            ##################################################
            # staring point of the interpolation section
            trans = trans_points[:]

            ##################################################
            # cEFT params
            if ceft_model == 'HLPS':
                gamma = 4.0 * inv3
                aL, eL = ceft_params_hlps()

                while not check_cEFT(aL, eL):
                    aL, eL = ceft_params_hlps()
            elif ceft_model == 'HLPS3':
                gamma = np.random.uniform(1.2, 2.5)
                aL, eL = ceft_params_hlps()

                while not check_cEFT(aL, eL):
                    aL, eL = ceft_params_hlps()
            elif ceft_model == 'HLPS+':
                prob_ceft, aL, eL, gamma, zL, rr0 = ceft_prob_hlpsp()

                while not prob_ceft > 0:
                    prob_ceft, aL, eL, gamma, zL, rr0 = ceft_prob_hlpsp()


            # pQCD param
            X = np.random.uniform(1., 4.)

            ##################################################
            # cEFT and pQCD params
            if ceft_model == 'HLPS':
                list1 = [aL, eL, X]
            elif ceft_model == 'HLPS3':
                list1 = [aL, eL, gamma, X]
            elif ceft_model == 'HLPS+':
                list1 = [aL, eL, gamma, zL, rr0, X]

            ##################################################
            # low-density cEFT EoS
            if ceft_model == 'HLPS' or ceft_model == 'HLPS3':
                lowDensity = [gamma, aL, eL]
            elif ceft_model == 'HLPS+':
                lowDensity = [gamma, aL, eL, zL, rr0]

            ##################################################
            # interpolation-model parameters
            list2 = []

            if eos_model == 0:  # polytropic model
                if phaseTransition > 0:
                    PTrans = 1
                else:
                    PTrans = 0

                for i in range(eos_Nsegment - 2 - PTrans):  # gamma
                    if i == 0:
                        list2.append(np.random.uniform(1.0, 3.5))
                    elif i == 1:
                        list2.append(np.random.uniform(1.0, 2.5))
                    else:
                        list2.append(np.random.uniform(1.0, 2.))

                trans_qcd = nQCD(muQCD, X) * cgs.mB / cgs.rhoS
                trans_interp = trans_points[0] / cgs.rhoS

                for i in range(eos_Nsegment - 1):  # delta n_B
                    summa = 0. if i == 0 else sum(list2[eos_Nsegment-2-PTrans:])
                    nb_max = min(43., trans_qcd - trans_interp - summa)

                    if i == 0:
                        list2.append(np.random.uniform(0.0, 4.))
                    elif i == 1:
                        list2.append(np.random.uniform(20.0, nb_max))
                    else:
                        list2.append(np.random.uniform(0.0, nb_max))

            elif eos_model == 1:  # c_s^2 model
                if ceft_model == 'HLPS' or ceft_model == 'HLPS3':
                    ceftEoS = cEFT(lowDensity)
                elif ceft_model == 'HLPS+':
                    ceftEoS = cEFT_r4(lowDensity)

                    if not ceftEoS.realistic:
                        again = True
                        continue

                pr0 = ceftEoS.pressure(trans[0])
                e0 = ceftEoS.edens(trans[0])
                n0 = trans[0]
                mu0 = cgs.mB * ( e0 * cgs.c**2 + pr0 ) / ( n0 * cgs.eV ) * 1.0e-9

                for i in range(eos_Nsegment - 2):  # delta mu_B
                    summa = 0.0 if i == 0 else sum(list2[-i:])
                    muDmax = min( 1.8, muQCD - mu0 - summa )

                    if i == 0 and flagSubConformal:
                        list2.append(np.random.uniform(.04, .1))
                    elif i == 0:
                        list2.append(np.random.uniform(.1, .3))
                    else:
                        list2.append(np.random.uniform(0.0, muDmax))

                for i in range(eos_Nsegment - 2):  # c_s^2
                    if flagSubConformal:
                        list2.append(np.random.uniform(0.3, inv3))
                    else:
                        if i == 0:
                            list2.append(np.random.uniform(.5, .7))
                        else:
                            list2.append(np.random.uniform(0.0, 1.))

            cube = list1 + list2

            ##################################################
            # interpolated EoS

            # general running index that maps cube array to real EoS parameters
            if ceft_model == 'HLPS':
                ci = 3
            elif ceft_model == 'HLPS3':
                ci = 4
            elif ceft_model == 'HLPS+':
                ci = 6

            if eos_model == 0:
                # Polytropic exponents excluding the first two ones
                gammas = []
                for itrope in range(eos_Nsegment-2):
                    if itrope + 1 != phaseTransition:
                        gammas.append(cube[ci])
                        ci += 1
                    else:
                        gammas.append(0.0)

                # Transition ("matching") densities (g/cm^3)
                for itrope in range(eos_Nsegment-1):
                    if debug:
                        mpi_print("loading trans from cube #{}".format(ci))
                    trans.append(trans[-1] + cgs.rhoS * cube[ci])
                    ci += 1
            elif eos_model == 1:
                # Matching chemical potentials (GeV)
                mu_deltas = []
                for itrope in range(eos_Nsegment-2):
                    if debug:
                        mpi_print("loading mu_deltas from cube #{}".format(ci))
                    mu_deltas.append(cube[ci])
                    ci += 1

                speed2 = []
                # Speed of sound squareds (unitless)
                for itrope in range(eos_Nsegment-2):
                    if debug:
                        mpi_print("loading speed2 from cube #{}".format(ci))
                    speed2.append(cube[ci])
                    ci += 1

            ##################################################
            # high-density pQCD EoS

            # Perturbative QCD parameters, see e.g. Fraga et al. (2014, arXiv:1311.5154) for details
            highDensity = [muQCD, X]

            # Check that last transition (matching) point is large enough
            if nQCD(muQCD, X) * cgs.mB <= trans[-1]:
                again = True
                continue

            ##################################################
            # build neutron star structure

            # Construct the EoS
            if eos_model == 0:
                struc = structure_poly(gammas, trans, lowDensity, highDensity, CEFT_model = ceft_model)
            elif eos_model == 1:
                struc = structure_c2(mu_deltas, speed2, trans, lowDensity, highDensity, approximation = True, CEFT_model = ceft_model)

            # Is the obtained EoS realistic, e.g. causal?
            if not struc.realistic:
                again = True
                continue

            # Discard subconformal EoSs
            if flagSubConformal:
                if 3.0 * struc.speed2max > 1.0:
                    again = True
                    continue

            mmax = 10.0
            Mc   = 1.186
            q    = 1.0
            if flag_TOV:
                if flag_GW: # with tidal deformabilities
                    #GW170817
                    Mc = np.random.uniform(1.1855, 1.1865)

                    mass_GW170817 = 2.0**0.2 * Mc * cgs.Msun
                    struc.tov(l=2, m1=mass_GW170817, m2=mass_GW170817, rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)

                    # Realistic max mass and radius
                    if struc.maxmass < mass_cut or struc.maxmassrad >= radius_cut:
                        again = True
                        continue

                    if struc.TD > 1600.0 or struc.TD2 > 1600.0 or struc.TD <= 0.0 or struc.TD2 <= 0.0:
                        again = True
                        continue

                    q  = np.random.uniform(0.8, 1.0)
                    mass1_GW170817 = (1.0 + q)**0.2 / q**0.6 * Mc
                    mass2_GW170817 = mass1_GW170817 * q

                    struc.tov(l=2, m1=mass1_GW170817*cgs.Msun, m2=mass2_GW170817*cgs.Msun, rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)
                else: # without
                    struc.tov(rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)

                    # Realistic max mass and radius
                    if struc.maxmass < mass_cut or struc.maxmassrad >= radius_cut:
                        again = True
                        continue

                # Maximum mass [Msun]
                mmax = struc.maxmass

                if flag_GW:
                    if mmax < 2.0**0.2 * 1.17:
                        again = True
                        continue

                    while mmax < ( (1.0 + q)**0.2 / q**0.6 * Mc ) or struc.TD > 1600.0 or struc.TD2 > 1600.0 or struc.TD <= 0.0 or struc.TD2 <= 0.0:
                        #GW170817
                        q  = np.random.uniform(0.8, 1.0)
                        mass1_GW170817 = (1.0 + q)**0.2 / q**0.6 * Mc
                        mass2_GW170817 = mass1_GW170817 * q

                        struc.tov(l=2, m1=mass1_GW170817*cgs.Msun, m2=mass2_GW170817*cgs.Msun, rhocs = param_indices['nsat_short_grid'] * cgs.rhoS)

            # mass measurements
            Mm1 = np.random.uniform(1.97, min(mmax, 2.05))

            #################################################
            # M-R measurements

            try_count = 1000

            # 4U 1702-429, arXiv:1709.09120
            for i in range(try_count):
                # mass [Msun]
                MR01  = np.random.uniform(1.6, min(mmax, 2.0)) #M_1702 [Msun]
                # radius [km]
                rad_MR01 = struc.radius_at(MR01)

                if np.isfinite( measurement_MR(MR01, rad_MR01, NSK17) ) and rad_MR01 < radius_cut:
                    break

            # NGC 6304, helium atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR02  = np.random.uniform(0.7, min(mmax, 2.6))
                rad_MR02 = struc.radius_at(MR02)
                if np.isfinite( measurement_MR(MR02, rad_MR02, SHB18_6304_He) ) and rad_MR02 < radius_cut:
                    break

            # NGC 6397, helium atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR03  = np.random.uniform(1.5, min(mmax, 2.0))
                rad_MR03 = struc.radius_at(MR03)
                if np.isfinite( measurement_MR(MR03, rad_MR03, SHB18_6397_He) ) and rad_MR03 < radius_cut:
                    break

            # M28, helium atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR04  = np.random.uniform(1.5, min(mmax, 2.4))
                rad_MR04 = struc.radius_at(MR04)
                if np.isfinite( measurement_MR(MR04, rad_MR04, SHB18_M28_He) ) and rad_MR04 < radius_cut:
                    break

            # M30, hydrogen atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR05  = np.random.uniform(0.7, min(mmax, 2.0))
                rad_MR05 = struc.radius_at(MR05)
                if np.isfinite( measurement_MR(MR05, rad_MR05, SHB18_M30_H) ) and rad_MR05 < radius_cut:
                    break

            # 47 Tuc X7, hydrogen atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR06  = np.random.uniform(1.2, min(mmax, 1.9))
                rad_MR06 = struc.radius_at(MR06)
                if np.isfinite( measurement_MR(MR06, rad_MR06, SHB18_X7_H) ) and rad_MR06 < radius_cut:
                    break

            # wCen, hydrogen atmosphere, arXiv:1709.05013
            for i in range(try_count):
                MR07  = np.random.uniform(0.7, min(mmax, 2.0))
                rad_MR07 = struc.radius_at(MR07)
                if np.isfinite( measurement_MR(MR07, rad_MR07, SHB18_wCen_H) ) and rad_MR07 < radius_cut:
                    break

            # M13, arXiv:1803.00029
            for i in range(try_count):
                MR08  = np.random.uniform(1.3, min(mmax, 1.9))
                rad_MR08 = struc.radius_at(MR08)
                if np.isfinite( measurement_MR(MR08, rad_MR08, SHS18_M13_H) ) and rad_MR08 < radius_cut:
                    break

            # 4U 1724-307, arXiv:1509.06561
            for i in range(try_count):
                MR09  = np.random.uniform(0.8, min(mmax, 1.9))
                rad_MR09 = struc.radius_at(MR09)
                if np.isfinite( measurement_MR(MR09, rad_MR09, NKS15_1724) ) and rad_MR09 < radius_cut:
                    break

            # SAX J1810.8-260, arXiv:1509.06561
            for i in range(try_count):
                MR10  = np.random.uniform(0.8, min(mmax, 1.9))
                rad_MR10 = struc.radius_at(MR10)
                if np.isfinite( measurement_MR(MR10, rad_MR10, NKS15_1810) ) and rad_MR10 < radius_cut:
                    break

            # J0030+0451, NICER, arXiv:TODO
            for i in range(try_count):
                # TODO optimal range
                MR11  = np.random.uniform(1.0129494423462766, min(mmax, 2.2792157996697235))
                rad_MR11 = struc.radius_at(MR11)
                if np.isfinite( measurement_MR(MR11, rad_MR11, NICER_0030) ) and rad_MR11 < radius_cut:
                    break

            # PSR J0740+6620
            for i in range(try_count):
                Mm2   = np.random.uniform(2.01, min(mmax, 2.15)) # M_6620
                rad_Mm2 = struc.radius_at(Mm2)
                # TODO remove comments when we have the new dataset
                #if np.isfinite( measurement_MR(Mm2, rad_Mm2, NICER_6620) ) and rad_Mm2 < radius_cut:
                #    break
                break #TODO remove this line too

            list3 = [Mc, q, Mm1, Mm2, MR01, MR02, MR03, MR04, MR05, MR06, MR07, MR08, MR09, MR10, MR11]

            #initial point
            pinit = cube + list3

            # final check that the point is realsitic
            prob_density = lnprob(pinit)[0]

            if not np.isfinite( prob_density ) or prob_density < -100:
                again = True
                continue

            # generate a set of points around pinit
            pi_vec = [pinit] * nwalkers
            pi_vec = list( map(lambda z: z + 0.001 * np.random.randn(ndim), pi_vec) )

            # split the set to all usable cores
            pi_vec = splitter(pi_vec, size)

        #parallellization
        pi_vec = comm.scatter(pi_vec, root=0)

        #res = list( map( init_rand, pi_vec) )
        res = [init_rand(item, ndim) for item in pi_vec]
        res = splitter(res, size)
        res = comm.allgather(res)

        res = [i for sl in res for ssl in sl for i in ssl]

        return res

if __name__ == "__main__":
    # MCMC sample params
    ndim = len(parameters)
    nwalkers = args.walkers * ndim

    # initial point(s)
    p0 = initial_point(nwalkers, ndim)

    # number of steps
    Nsteps = args.nsteps

    ##################################################
    #serial v3.0
    if False:

        #output
        filename = prefix+'run.h5'

        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim) #no restart
        
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)

        result = sampler.run_mcmc(p0, Nsteps)

        import h5py
        hf = h5py.File(filename, 'a')
        group = hf.get('mcmc')
        group.create_dataset('mass_grid', data=param_indices['mass_grid'])
        group.create_dataset('eps_grid', data=param_indices['eps_grid'])
        group.create_dataset('nsat_long_grid', data=param_indices['nsat_long_grid'])
        group.create_dataset('nsat_short_grid', data=param_indices['nsat_short_grid'])
        group.create_dataset('const_params', data=const_params)
        hf.close()

    #parallel v3.0
    if True:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        from schwimmbad import MPIPool
        #from emcee.utils import MPIPool

        #even out all workers
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            #output
            filename = prefix+'run.h5'

            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim) #no restart

            # initialize sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)
            result = sampler.run_mcmc(p0, Nsteps, progress=True)

        import h5py
        hf = h5py.File(filename, 'a')
        group = hf.get('mcmc')
        group.create_dataset('mass_grid', data=param_indices['mass_grid'])
        group.create_dataset('eps_grid', data=param_indices['eps_grid'])
        group.create_dataset('nsat_long_grid', data=param_indices['nsat_long_grid'])
        group.create_dataset('nsat_short_grid', data=param_indices['nsat_short_grid'])
        group.create_dataset('const_params', data=const_params)
        hf.close()

    # serial version emcee v2.2
    if False:

        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #, pool=pool)

        #output
        f = open("chains2/chain.dat", "w")
        f.close()

        result = sampler.run_mcmc(p0, 20)
        mpi_print(result)
        position = result[0]
        mpi_print(position)

        #loop & sample
        for result in sampler.sample(p0, iterations=1, storechain=False):
            print(result)
            position = result[0]
            print(position)
        
            f = open("chain.dat", "a")
            for k in range(position.shape[0]):
               f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
            f.close()

    ##################################################
    # parallel version emcee v2.2
    if False:
        from emcee.utils import MPIPool

        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        #sampler.run_mcmc(p0, 20)

        for result in sampler.sample(p0, iterations=2, storechain=False):
            if pool.is_master():
                #print(result) #pos, lnprob, rstate, [blobs]
                position = result[0]
                mpi_print("position:")
                mpi_print(position)

                f = open("chain.dat", "a")
                for k in range(position.shape[0]):
                   f.write("{0:4d} {1:s}\n".format(k, " ".join(str(position[k]))))
                f.close()

        pool.close()

