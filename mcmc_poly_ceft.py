#--------------------------------------------------
# system packages
import numpy as np
import os
import argparse
from scipy.stats import norm

#  problem physics modules
#--------------------------------------------------
from priors import check_uniform, check_cEFT
from structure import structurePolytropeWithCEFT as structure
import units as cgs
from pQCD import nQCD, pSB_csg

# measurements
#--------------------------------------------------
from measurements import gaussian_MR
from measurements import NSK17 #1702 measurement

from measurements import measurement_MR
from measurements import SHB18_6304_He #6304 measurement
from measurements import SHB18_6397_He #6397 measurement
from measurements import SHB18_M28_He  #M28 measurement
from measurements import SHB18_M30_H   #M30 measurement
from measurements import SHB18_X7_H    #X7 measurement
from measurements import SHB18_X5_H    #X5 measurement
from measurements import SHB18_wCen_H  #wCen measurement
from measurements import SHS18_M13_H   #M13 measurement
from measurements import NKS15_1724    #1724 measurement
from measurements import NKS15_1810    #1810 measurement


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

np.random.seed(args.seed) #for reproducibility

if not os.path.exists(args.outputdir): os.mkdir(args.outputdir)



##################################################
# global flags for different run modes
eos_Ntrope = args.eos_nseg #polytrope order
debug = args.debug  #flag for additional debug printing
phaseTransition = args.ptrans #position of the 1st order transition


#after first two monotropes, 0: no phase transition
#in other words, the first two monotrope do not behave
#like a latent heat (ie. gamma != 0)

##################################################
flag_TOV   = True # calculating MR curve
flag_GW    = True # using GW170817 event as a constrain (NB usable if flag_TOV = True)
flag_Mobs  = True # using mass measurement data (NB usable if flag_TOV = True)
flag_MRobs = True # using mass-radius observations (NB usable if flag_TOV = True)


##################################################
#auto-generated parameter names for polytropes 

#cEFT + pQCD parameters
parameters = ["alphaL", "etaL", "X"]

#append gammas (start from 3 since two first ones are given by QMC)
for itrope in range(eos_Ntrope-2):
    if itrope + 1 != phaseTransition:
        parameters.append("gamma"+str(3+itrope))

#append transition depths
for itrope in range(eos_Ntrope-1):
    parameters.append("trans_delta"+str(1+itrope))

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
parameters.append("mass_X5")
parameters.append("mass_wCen")
parameters.append("mass_M13")
parameters.append("mass_1724")
parameters.append("mass_1810")


mpi_print("Parameters to be sampled are:")
mpi_print(parameters)


n_params = len(parameters)
prefix = "chains/PC{}_{}-s{}T1".format(eos_Ntrope, phaseTransition, args.seed)


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

parameters2 = []

Ngrid = args.ngrid
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0,   Ngrid),
        'eps_grid':  np.logspace(2.0, 4.3, Ngrid),
        'nsat_gamma_grid': np.linspace(1.1, 45.0, Ngrid), #TODO limits
        'nsat_c2_grid': np.linspace(1.1, 45.0, Ngrid),    #TODO
        'nsat_press_grid': np.linspace(1.1, 45.0, Ngrid), #TODO
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

#add nsat - gamma grid
for ir, nsat  in enumerate(param_indices['nsat_gamma_grid']):
    parameters2.append('nsat_gamma_'+str(ir))
    param_indices['nsat_gamma_'+str(ir)] = ci
    ci += 1

#add nsat - c^2 grid
for ir, nsat  in enumerate(param_indices['nsat_c2_grid']):
    parameters2.append('nsat_c2_'+str(ir))
    param_indices['nsat_c2_'+str(ir)] = ci
    ci += 1

#add nsat - press grid
for ir, nsat  in enumerate(param_indices['nsat_press_grid']):
    parameters2.append('nsat_press_'+str(ir))
    param_indices['nsat_press_'+str(ir)] = ci
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

# max squared speed of sound
parameters2.append('c2max')
param_indices['c2max'] = ci+8

mpi_print("Parameters to be only stored (blobs):")
mpi_print(len(parameters2))
n_blobs = len(parameters2)



##################################################
# Prior function; changes from [0,1] to whatever physical lims
#def myprior(cube, ndim, nparams):
def myprior(cube):
    lps = np.empty_like(cube)

    # Parameters of the cEFT EoS
    lps[0] = check_uniform(cube[0], 1.17, 1.61) #alphaL [unitless] TODO check 2D region
    lps[1] = check_uniform(cube[1], 0.6,  1.15) #etaL [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    lps[2] = check_uniform(cube[2], 1.0,  4.0 ) #X [unitless]

    # Polytropic exponents excluding the first two ones
    ci = 3
    for itrope in range(eos_Ntrope-2):
        if itrope + 1 != phaseTransition:
            if debug:
                print("prior for gamma from cube #{}".format(ci))
            lps[ci] = check_uniform(cube[ci], 0.0, 10.0)  #gamma_i [unitless]
            ci += 1


    # Lengths of the first N-1 monotropes (N = # of polytropes)
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("prior for trans from cube #{}".format(ci))
        lps[ci] = check_uniform(cube[ci], 0.0, 43.0)  #delta_ni [rhoS]
        ci += 1

    # TD measurements
    lps[ci] = check_uniform(cube[ci],   1.000, 1.372) #TODO limits ok?
    lps[ci+1] = check_uniform(cube[ci+1], 0.0, 1.0)

    ci += 2

    # M measurements
    lps[ci]   =  check_uniform(cube[ci], 1.0, 4.0)   # 2.01 NS TODO limits
    lps[ci+1] =  check_uniform(cube[ci+1], 1.0, 4.0) # 2.14 NS TODO limits

    ci += 2

    # M-R measurements
    lps[ci+0]  = check_uniform(cube[ci],    1.0, 2.5)  #M_1702 [Msun]
    lps[ci+1]  = check_uniform(cube[ci+1],  0.5, 2.7)  #M_6304 [Msun]
    lps[ci+2]  = check_uniform(cube[ci+2],  0.5, 2.0)  #M_6397 [Msun]
    lps[ci+3]  = check_uniform(cube[ci+3],  0.5, 2.8)  #M_M28 [Msun]
    lps[ci+4]  = check_uniform(cube[ci+4],  0.5, 2.5)  #M_M30 [Msun]
    lps[ci+5]  = check_uniform(cube[ci+5],  0.5, 2.7)  #M_X7 [Msun]
    lps[ci+6]  = check_uniform(cube[ci+6],  0.5, 2.7)  #M_X5 [Msun]
    lps[ci+7]  = check_uniform(cube[ci+7],  0.5, 2.5)  #M_wCen [Msun]
    lps[ci+8]  = check_uniform(cube[ci+8],  0.8, 2.4)  #M_M13 [Msun]
    lps[ci+9]  = check_uniform(cube[ci+9],  0.8, 2.5)  #M_1724 [Msun]
    lps[ci+10] = check_uniform(cube[ci+10], 0.8, 2.5)  #M_1810 [Msun]

    return np.sum(lps)





# probability function
linf = np.inf


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

        #pQCD parameters:
        3 X

        #nuclear EoS parameters:
        gammas
        transition points

        # Measurement parameters:
        mass of individual objects


    """

    ##################################################
    # initialization
    #print(cube)
    blobs = np.zeros(n_blobs)
    logl = 0.0 #total likelihood

    if debug:
        global icalls
        icalls += 1
        mpi_print(icalls, cube)

    ################################################## 
    # cEFT low-density EOS parameters
    if debug:
        mpi_print("Checking cEFT")
    if not(check_cEFT(cube[0], cube[1])):
        logl = -linf
        return logl, blobs

    ################################################## 
    # interpolated EoS
 
    # general running index that maps cube array to real EoS parameters
    ci = 3 
 
    # Polytropic exponents excluding the first two ones
    gammas = []  
    for itrope in range(eos_Ntrope-2):
        if itrope + 1 != phaseTransition:
            if debug:
                mpi_print("loading gamma from cube #{}".format(ci))
            gammas.append(cube[ci])
            ci += 1
        else:
            gammas.append(0.0)

    # Transition ("matching") densities (g/cm^3)
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #[0.9e14, 1.1 * cgs.rhoS] #starting point BTW 1.0e14 ~ 0.4*rhoS #TODO is the crust-core transtion density ok?
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("loading trans from cube #{}".format(ci))
        trans.append(trans[-1] + cgs.rhoS * cube[ci]) 
        ci += 1


    ################################################## 
    # low-density cEFT EoS

    # Parameters of the cEFT EoS
    gamma  = 4.0 / 3.0                    # unitless
    alphaL = cube[0]                      # untiless
    etaL   = cube[1]                      # untiless
    lowDensity = [gamma, alphaL, etaL]


    ################################################## 
    # high-density pQCD EoS

    # Perturbative QCD parameters, see Fraga et al. (2014, arXiv:1311.5154) for details
    X = cube[2]
    muQCD = 2.6 # Transition (matching) chemical potential where pQCD starts (GeV)
    highDensity = [muQCD, X]


    # Check that last transition (matching) point is large enough
    if debug:
        mpi_print("Checking nQCD")
    if nQCD(muQCD, X) * cgs.mB <= trans[-1]:
        logl = -linf
        return logl, blobs

    ##################################################
    # build neutron star structure 

    # Construct the EoS
    if debug:
        mpi_print("Structure...")
    struc = structure(gammas, trans, lowDensity, highDensity)

    # Is the obtained EoS realistic, e.g. causal?
    if debug:
        mpi_print("Checking structure")
    if not struc.realistic:
        logl = -linf
        return logl, blobs

    ################################################## 
    # measurements & constraints

    #Chirp mass (GW170817) [Msun]
    if flag_GW and flag_TOV:
        logl += norm.logpdf(cube[ci], 1.186, 0.0006079568319312625) #TODO varianssi?

        # Masses GW170817
        mass1_GW170817 = (1.0 + cube[ci+1])**0.2 / (cube[ci+1])**0.6 * cube[ci]
        mass2_GW170817 = mass1_GW170817 * cube[ci+1]

    ci += 2

    # solve structure 
    if debug:
        mpi_print("TOV...")
        
    mpi_print("params", cube)


    if flag_TOV:
        if flag_GW: # with tidal deformabilities
            struc.tov(l=2, m1=mass1_GW170817*cgs.Msun, m2=mass2_GW170817*cgs.Msun)
        else: # without
            struc.tov()

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

        logl += measurement_M(cube[ci],   J0348) #m0432 [Msun]
        logl += measurement_M(cube[ci+1], J0740) #m6620 [Msun]

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
        mass_X5   = cube[ci+6]
        mass_wCen = cube[ci+7]
        mass_M13  = cube[ci+8]
        mass_1724 = cube[ci+9]
        mass_1810 = cube[ci+10]

        # All stars have to be lighter than the max mass limit
        masses = [ mass_1702, mass_6304, mass_6397, mass_M28,
                   mass_M30, mass_X7, mass_X5, mass_wCen, mass_M13,
                   mass_1724, mass_1810, ]

        if any(m > mmax for m in masses):
            logl = -linf
            return logl, blobs

        # 4U 1702-429 from Nattila et al 2017, arXiv:1709.09120
        rad_1702 = struc.radius_at(mass_1702)
        logl += gaussian_MR(mass_1702, rad_1702, NSK17)

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

        # X5 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
        rad_X5 = struc.radius_at(mass_X5)
        logl += measurement_MR(mass_X5, rad_X5, SHB18_X5_H)

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

    # GW170817, tidal deformability
    if flag_GW and flag_TOV:
        if struc.TD > 1600.0 or struc.TD2 > 1600.0 or struc.TD < 0.0 or struc.TD2 < 0.0:
            logl = -linf
            return logl, blobs
  
        logl += measurement_TD(struc.TD, struc.TD2, GW170817)

    ic = 0

    #build M-R curve
    if debug:
        ic = param_indices['rad_0'] #starting index
        print("building M-R curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['rad_' + str(im)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.radius_at(mass) 

        if flag_TOV:
            blobs[ic] = struc.radius_at(mass) 
        else:
            blobs[ic] = 0.0

        if debug:
            print("im = {}, mass = {}, rad = {}, ic = {}".format(im, mass, blobs[ic], ic))

    #build eps-P curve
    if debug:
        ic = param_indices['Peps_0'] #starting index
        mpi_print("building eps-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, eps in enumerate(param_indices['eps_grid']):
        ic = param_indices['Peps_'+str(ir)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.eos.pressure_edens( eps * 0.001 * cgs.GeVfm_per_dynecm / (cgs.c**2) ) * 1000.0 / cgs.GeVfm_per_dynecm

        if debug:
            mpi_print("ir = {}, eps = {}, P = {}, ic = {}".format(ir, eps, blobs[ic], ic))

    #build nsat-gamma curve
    if debug:
        ic = param_indices['nsat_gamma_0'] #starting index
        mpi_print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_gamma_grid']):
        ic = param_indices['nsat_gamma_'+str(ir)] #this is the index pointing to correct position in cube

        blobs[ic] = struc.eos.gammaFunction( cgs.rhoS*nsat, flag = 0 )

        if debug:
            mpi_print("ir = {}, nsat = {}, gamma = {}, ic = {}".format(ir, nsat, blobs[ic], ic))

    #build nsat-c^2 curve
    if debug:
        ic = param_indices['nsat_c2_0'] #starting index
        mpi_print("building nsat-c^2 curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_c2_grid']):
        ic = param_indices['nsat_c2_'+str(ir)] #this is the index pointing to correct position in cube

        blobs[ic] = struc.eos.speed2( struc.eos.pressure( cgs.rhoS * nsat ) )

        if debug:
            mpi_print("ir = {}, nsat = {}, c^2 = {}, ic = {}".format(ir, nsat, blobs[ic], ic))

    #build nsat-press/press_SB curve
    if debug:
        ic = param_indices['nsat_press_0'] #starting index
        mpi_print("building nsat-press curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_press_grid']):
        ic = param_indices['nsat_press_'+str(ir)] #this is the index pointing to correct position in cube
        rhooB = cgs.rhoS * nsat
        pB = struc.eos.pressure( rhooB )
        muB = ( pB + struc.eos.edens_inv(pB) * cgs.c**2 ) * cgs.mB * cgs.erg_per_kev * 1.0e-6 / rhooB
        blobs[ic] = pB / pSB_csg( muB )

        if debug:
            mpi_print("ir = {}, nsat = {}, press = {}, ic = {}".format(ir, nsat, cube[ic], ic))

    # Variables at M = M_max
    ic = param_indices['mmax'] #this is the index pointing to correct position in cube
    rhoM = struc.maxmassrho
    press = struc.eos.pressure( rhoM )
    edens = struc.eos.edens ( rhoM )
    muB = ( press + edens * cgs.c**2 ) * cgs.mB * cgs.erg_per_kev * 1.0e-6 / rhoM
    blobs[ic] = mmax # max mass
    blobs[ic+1] = struc.maxmassrad # rad(Mmax)
    blobs[ic+2] = rhoM / cgs.rhoS # rho(Mmax) [rhoS]
    blobs[ic+3] = press * 1000.0 / cgs.GeVfm_per_dynecm # pressure [MeV/fm^3]
    blobs[ic+4] = edens * cgs.c**2 * 1000.0 / cgs.GeVfm_per_dynecm  # edens [MeV/fm^3]
    blobs[ic+5] = press / pSB_csg( muB ) # normalized press
    blobs[ic+6] = struc.eos.speed2_rho ( rhoM ) # speed of sound
    blobs[ic+7] = struc.eos.gammaFunction ( rhoM ) # gamma

    # Max speed of sound square
    ic = param_indices['c2max'] #this is the index pointing to correct position in cube
    blobs[ic] = struc.speed2max

    return logl, blobs



# combine likelihood and prior
def lnprob(cube):
    lp = myprior(cube)

    if not np.isfinite(lp):
        return -np.inf, np.zeros(n_blobs)
    else:
        ll, blobs = myloglike(cube)
        return lp + ll, blobs


##################################################
##################################################
##################################################
# MCMC sample


ndim = len(parameters)
nwalkers = 2 * ndim # XXX This isn't ideal, I guess

#initial guess

if eos_Ntrope == 2: #(trope = 2)
    if phaseTransition == 0:
        pinit = [1.4, 0.9, 1.2,
        3.3, 1.186, 0.85, 2.01, 2.14, 1.49127976, 2.0, 1.6, 1.6, 
        1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.18M_sun, no PT, Lambda_1.4 ~ 365

elif eos_Ntrope == 3: #(trope = 3)
    if phaseTransition == 0:
        pinit = [1.4, 0.9, 2.0,
        2.7, 3.57470224, 26.0, 1.186, 0.85, 2.01, 2.14, 1.49127976,
        2.0, 1.6, 1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.18M_sun, no PT, Lambda_1.4 ~ 360
    elif phaseTransition == 1:
        pinit = [1.4, 0.9, 1.16,
        3.57470224, 26.0, 1.186, 0.85, 2.01, 2.14, 1.49127976, 2.0, 1.6,
        1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.22M_sun, PT1, Lambda_1.4 ~ 365

elif eos_Ntrope == 4: #(trope = 4)
    if phaseTransition == 0:
        pinit = [1.4, 0.9, 2.0, 5.0,
        2.4, 3.57470224, 26.40907385, 1.31246422, 1.186, 0.85, 2.01, 2.14,
        1.49127976, 2.0, 1.6, 1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.17M_sun, no PT, Lambda_1.4 ~ 355
    elif phaseTransition == 1:
        pinit = [1.4, 0.9, 2.0,
        3.2, 3.57470224, 26.40907385, 1.31246422, 1.186, 0.85, 2.01, 2.14, 1.49127976,
        2.0, 1.6, 1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.18M_sun, PT1, Lambda_1.4 ~ 355
    elif phaseTransition == 2:
        pinit = [1.4, 0.9, 2.0,
        2.4, 3.57470224, 8.30907385, 19.41246422, 1.186, 0.85, 2.01, 2.14, 1.49127976,
        2.0, 1.6, 1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.22M_sun, PT2, Lambda_1.4 ~ 380

elif eos_Ntrope == 5: #(trope = 5)
    if phaseTransition == 1:
        pinit = [1.4, 0.9, 2.0, 5.0,
        2.6, 3.57470224, 25.00907385, 1.4, 1.31246422, 1.186, 0.85, 2.01, 2.14, 1.49127976,
        2.0, 1.6, 1.6, 1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
        ,]
        # ~2.18M_sun, PT1, Lambda_1.4 ~ 355


#initialize small Gaussian ball around the initial point
p0 = [pinit + 0.01*np.random.randn(ndim) for i in range(nwalkers)]

Nsteps = args.nsteps

##################################################
#serial v3.0-dev
if False:
    #output
    filename = prefix+'run.h5'

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim) #no restart
    
    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)

    result = sampler.run_mcmc(p0, Nsteps)

    #print(result)
    #position = result[0]
    #print(position)
    #loop & sample
    #for result in sampler.sample(p0, iterations=1, storechain=False):
    #    print(result)
    #    position = result[0]
    #    print(position)
    

#parallel v3.0-dev
if True:
    import os
    os.environ["OMP_NUM_THREADS"] = "1"    
    from schwimmbad import MPIPool

    #even out all workers
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        #output
        #filename = "chains2/chain190527P2PT0.h5"
        filename = prefix+'run.h5'

        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim) #no restart
        
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)
        result = sampler.run_mcmc(p0, Nsteps, progress=True)
        #result = sampler.run_mcmc(None, 1, progress=True)


# serial version emcee v2.2
if False:

    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #, pool=pool)
    
    #output
    f = open("chains2/chain.dat", "w")
    f.close()
    
    
    result = sampler.run_mcmc(p0, 20)
    print(result)
    position = result[0]
    print(position)
    
    
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
            print("position:")
            print(position)

            f = open("chain.dat", "a")
            for k in range(position.shape[0]):
               f.write("{0:4d} {1:s}\n".format(k, " ".join(str(position[k]))))
            f.close()

    pool.close()
