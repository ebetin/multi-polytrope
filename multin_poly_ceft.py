from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os
import argparse
from input_parser import parse_cli

from priors import transform_uniform, check_cEFT
from structure import structurePolytropeWithCEFT as structure
import units as cgs
from pQCD import nQCD, pSB_csg


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

from scipy.stats import norm

# emcee stuff
import sys
from pymultinest.solve import solve as pymlsolve
from pymultinest.run import run as pymlrun
import json

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


print("Parameters to be sampled are:")
print(parameters)


n_params = len(parameters)
prefix = "chains/PC{}_{}-s{}".format(eos_Ntrope, phaseTransition, args.seed)


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

# NOTE: for multinest everything is stored in the same array; combining params and params2
parameters2 = []

Ngrid = args.ngrid
param_indices = {
        'mass_grid': np.linspace(0.5, 3.6,   Ngrid),
        'eps_grid':  np.logspace(2.0, 4.3, Ngrid),
        'nsat_gamma_grid': np.linspace(1.1, 45.0, Ngrid), #TODO limits
        'nsat_c2_grid': np.linspace(1.1, 45.0, Ngrid), #TODO
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

print("Parameters to be only stored:")
print(len(parameters2))
n_blobs = len(parameters2)


# NOTE: for multinest everything is stored ifn the same array; combining params and params2
parameters = parameters + parameters2


##################################################
# Prior function; changes from [0,1] to physical limits
def myprior(cube):
    # Parameters of the cEFT EoS, see Hebeler et al. (2013, arXiv:1303.4662)
    cube[0] = transform_uniform(cube[0], 1.17, 1.61) #alphaL [unitless] TODO check 2D region
    cube[1] = transform_uniform(cube[1], 0.6,  1.15) #etaL [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    cube[2] = transform_uniform(cube[2], 1.0,  4.0 ) #X [unitless]

    # Polytropic exponents excluding the first two ones
    ci = 3
    for itrope in range(eos_Ntrope-2):
        if itrope + 1 != phaseTransition:
            if debug:
                print("prior for gamma from cube #{}".format(ci))
            cube[ci] = transform_uniform(cube[ci], 0.0, 10.0)  #gamma_i [unitless]
            ci += 1

    # Lengths of the first N-1 monotropes (N = # of polytropes)
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("prior for trans from cube #{}".format(ci))
        cube[ci] = transform_uniform(cube[ci], 0.0, 43.0)  #delta_ni [rhoS]
        ci += 1

    # TD measurements
    #mu, sig = 1.186, 0.0006 #Chirp mass (GW170817) [Msun]
    cube[ci]   = transform_uniform(cube[ci],   1.0, 1.3) #TODO limits ok?
    #cube[ci+1] = transform_uniform(cube[ci+1], 0.0, 1.0)  #Mass ratio (GW170817), NOTE no need to renormalize
    ci += 2


    # pulsar maximum M measurements; used as an upper limit constraint for EOS
    cube[ci]   =  transform_uniform(cube[ci], 1.0, 4.0)   # 2.01 NS TODO limits
    cube[ci+1] =  transform_uniform(cube[ci+1], 1.0, 4.0) # 2.14 NS TODO limits
    ci += 2

    # M-R measurements
    cube[ci+0]  = transform_uniform(cube[ci],    1.0, 2.5)  #M_1702 [Msun]
    cube[ci+1]  = transform_uniform(cube[ci+1],  0.5, 2.7)  #M_6304 [Msun]
    cube[ci+2]  = transform_uniform(cube[ci+2],  0.5, 2.0)  #M_6397 [Msun]
    cube[ci+3]  = transform_uniform(cube[ci+3],  0.5, 2.8)  #M_M28 [Msun]
    cube[ci+4]  = transform_uniform(cube[ci+4],  0.5, 2.5)  #M_M30 [Msun]
    cube[ci+5]  = transform_uniform(cube[ci+5],  0.5, 2.7)  #M_X7 [Msun]
    cube[ci+6]  = transform_uniform(cube[ci+6],  0.5, 2.7)  #M_X5 [Msun]
    cube[ci+7]  = transform_uniform(cube[ci+7],  0.5, 2.5)  #M_wCen [Msun]
    cube[ci+8]  = transform_uniform(cube[ci+8],  0.8, 2.4)  #M_M13 [Msun]
    cube[ci+9]  = transform_uniform(cube[ci+9],  0.8, 2.5)  #M_1724 [Msun]
    cube[ci+10] = transform_uniform(cube[ci+10], 0.8, 2.5)  #M_1810 [Msun]

    return cube





# probability function
linf = 1e100 # numerical infinity


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
    if debug:
        global icalls
        icalls += 1
        print(icalls, cube)

    logl = 0.0 #total likelihood


    ################################################## 
    # cEFT low-density EOS parameters
    if debug:
        print("Checking cEFT")
    if not(check_cEFT(cube[0], cube[1])):
        logl = -linf
        return logl 


    ################################################## 
    # interpolated EoS
 
    # general running index that maps cube array to real EoS parameters
    ci = 3 

    # Polytropic exponents excluding the first two ones
    gammas = []  
    for itrope in range(eos_Ntrope-2):
        if itrope + 1 != phaseTransition:
            if debug:
                print("loading gamma from cube #{}".format(ci))
            gammas.append(cube[ci])
            ci += 1
        else:
            gammas.append(0.0)

    # Transition ("matching") densities (g/cm^3)
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #starting points #TODO is the crust-core transtion density ok?
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
        print("Checking nQCD")
    if nQCD(muQCD, X) * cgs.mB <= trans[-1]:
        logl = -linf
        return logl 

    ##################################################
    # build neutron star structure 

    # Construct the EoS
    if debug:
        print("Structure...")
    struc = structure(gammas, trans, lowDensity, highDensity)

    # Is the obtained EoS realistic, e.g. causal?
    if debug:
        print("Checking structure")
    if not struc.realistic:
        logl = -linf
        return logl

    ################################################## 
    # measurements & constraints

    #Chirp mass (GW170817) [Msun]
    if flag_GW and flag_TOV:
        logl += norm.logpdf(cube[ci], 1.186, 0.0006079568319312625)  

        # Masses GW170817
        mass1_GW170817 = (1.0 + cube[ci+1])**0.2 / (cube[ci+1])**0.6 * cube[ci]
        mass2_GW170817 = mass1_GW170817 * cube[ci+1]

    ci += 2

    # solve structure 
    if debug:
        print("TOV...")
    
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
                return logl

    # Mass measurement of PSR J0348+0432 from Antoniadis et al 2013 arXiv:1304.6875
    # and PSR J0740+6620 from Cromartie et al 2019 arXiv:1904.06759
    if flag_Mobs and flag_TOV:
        m0432 = cube[ci]
        m6620 = cube[ci+1]

        if m0432 > mmax or m6620 > mmax:
            logl = -linf
            return logl 

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
            return logl 

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
            return logl

        logl += measurement_TD(struc.TD, struc.TD2, GW170817)

    ci += 11
    
    #build M-R curve
    if debug:
        ic = param_indices['rad_0'] #starting index
        print("building M-R curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['rad_' + str(im)] #this is the index pointing to correct position in cube
        cube[ci+ic] = struc.radius_at(mass)
    
        if debug:
            print("im = {}, mass = {}, rad = {}, ic = {}".format(im, mass, cube[ic], ic))

    #build eps-P curve
    if debug:
        ic = param_indices['Peps_0'] #starting index
        print("building eps-P curve from EoS... (starts from ic = {}".format(ic))
    
    for ir, eps in enumerate(param_indices['eps_grid']):
        ic = param_indices['Peps_'+str(ir)] #this is the index pointing to correct position in cube
        cube[ci+ic] = struc.eos.pressure_edens( eps * 0.001 * cgs.GeVfm_per_dynecm / (cgs.c**2) ) * 1000.0 / cgs.GeVfm_per_dynecm
    
        if debug:
            print("ir = {}, eps = {}, P = {}, ic = {}".format(ir, eps, cube[ic], ic))

    #build nsat-gamma curve
    if debug:
        ic = param_indices['nsat_gamma_0'] #starting index
        print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_gamma_grid']):
        ic = param_indices['nsat_gamma_'+str(ir)] #this is the index pointing to correct position in cube

        cube[ci+ic] = struc.eos.gammaFunction( cgs.rhoS*nsat, flag = 0 )

        if debug:
            print("ir = {}, nsat = {}, gamma = {}, ic = {}".format(ir, nsat, cube[ic], ic))

    #build nsat-c^2 curve
    if debug:
        ic = param_indices['nsat_c2_0'] #starting index
        print("building nsat-c^2 curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_c2_grid']):
        ic = param_indices['nsat_c2_'+str(ir)] #this is the index pointing to correct position in cube

        cube[ci+ic] = struc.eos.speed2( struc.eos.pressure( cgs.rhoS * nsat ) )

        if debug:
            print("ir = {}, nsat = {}, c^2 = {}, ic = {}".format(ir, nsat, cube[ic], ic))

    #build nsat-press/press_SB curve
    if debug:
        ic = param_indices['nsat_press_0'] #starting index
        print("building nsat-press curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_press_grid']):
        ic = param_indices['nsat_press_'+str(ir)] #this is the index pointing to correct position in cube
        rhooB = cgs.rhoS * nsat
        pB = struc.eos.pressure( rhooB )
        muB = ( pB + struc.eos.edens_inv(pB) * cgs.c**2 ) * cgs.mB * cgs.erg_per_kev * 1.0e-6 / rhooB
        cube[ci+ic] = pB / pSB_csg( muB )

        if debug:
            print("ir = {}, nsat = {}, press = {}, ic = {}".format(ir, nsat, cube[ic], ic))

    return logl

##################################################
# run MultiNest
if flag_MN:
    result = pymlsolve( 
                LogLikelihood        = myloglike, 
                Prior                = myprior, 
    	        n_dims               = n_params + n_blobs,
    	        n_params             = n_params,
                outputfiles_basename = prefix,
                resume               = False,
                verbose              = True,
                seed                 = args.seed,
                sampling_efficiency  = 0.8,
                evidence_tolerance   = 0.5,
                n_live_points        = 400,
                log_zero             = -linf
                )

    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    
    for name, col in zip(parameters, result['samples'].transpose()):
    	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
    with open('%sparams.json' % prefix, 'w') as f:
        json.dump(parameters, f, indent=2)
    

