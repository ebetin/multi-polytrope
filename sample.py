from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve

from priors import transform_uniform
from structure import structure
import units as cgs
from pQCD import nQCD


from measurements import gaussian_MR
from measurements import NSK17 #1702 measurement





if not os.path.exists("chains"): os.mkdir("chains")


##################################################
# global flags for different run modes
eos_Ntrope = 4 #polytrope order
debug = False  #flag for additional debug printing


##################################################
#parameter names

#4-trope
#parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "gamma4", "trans_delta1", "trans_delta2", "trans_delta3"]

#3-trope
#parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "trans_delta1", "trans_delta2"]

#2-trope
#parameters = ["a", "alpha", "b", "beta", "X", "trans_delta1"]



##################################################
#auto-generated parameter names for polytropes 

#QMC + pQCD parameters
parameters = ["a", "alpha", "b", "beta", "X"]

#append gammas (start from 3 since two first ones are given by QMC)
for itrope in range(eos_Ntrope-2):
    parameters.append("gamma"+str(3+itrope))

#append transition depths
for itrope in range(eos_Ntrope-1):
    parameters.append("trans_delta"+str(1+itrope))

#finally add individual object masses (needed for measurements)
parameters.append("mass_1702")


print("Parameters to be sampled are:")
print(parameters)



n_params = len(parameters)
prefix = "chains/1-"


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

Ngrid = 20
param_indices = {
        'mass_grid' :np.linspace(0.5, 2.8,   Ngrid),
        'rho_grid':  np.logspace(14.3, 16.0, Ngrid),
        'nsat_grid': np.linspace(1.0, 20.0, Ngrid),
               }

#add M-R grid
ci = n_params #current running index of the parameters list
for im, mass  in enumerate(param_indices['mass_grid']):
    parameters.append('rad_'+str(im))
    param_indices['rad_'+str(im)] = ci
    ci += 1

#add rho-P grid
for ir, rho  in enumerate(param_indices['rho_grid']):
    parameters.append('P_'+str(ir))
    param_indices['P_'+str(ir)] = ci
    ci += 1

#add nsat - gamma grid
for ir, nsat  in enumerate(param_indices['nsat_grid']):
    parameters.append('nsat_'+str(ir))
    param_indices['nsat_'+str(ir)] = ci
    ci += 1

print(param_indices)


##################################################
# Prior function; changes from [0,1] to whatever physical lims
#def myprior(cube, ndim, nparams):
def myprior(cube):
    # TODO check these!
    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    cube[0] = transform_uniform(cube[0], 10.0, 14.0 ) #a [Mev]
    cube[1] = transform_uniform(cube[1],  0.4,  0.55) #alpha [unitless]
    cube[2] = transform_uniform(cube[2],  1.5,  7.5 ) #b [MeV]
    cube[3] = transform_uniform(cube[3],  1.8,  2.7 ) #beta [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    cube[4] = transform_uniform(cube[4],  1.0,  4.0 ) #X [unitless]


    # Polytropic exponents excluding the first two ones
    ci = 5
    for itrope in range(eos_Ntrope-2):
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

    # M-R measurements
    cube[ci] = transform_uniform(cube[ci], 1.0, 2.5)  #M_1702 [Msun]



    return cube



# probability function
linf = 100.0

icalls = 0
def myloglike(cube):
    """
        General likelihood function that builds the EoS and solves TOV-structure
        equations for it.

        Parameters from 0:nDim are sampled, everything beyond that are just 
        carried along for analysis purposes.

        ## QMC parameters:
        0 a
        1 alpha
        2 b
        3 beta

        #pQCD parameters:
        4 X

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

    # general running index that maps cube array to real EoS parameters
    # NOTE: parameters from 0 to 4 are reserved for QMC and pQCD. However,
    # we first build the nuclear Eos (>=5) and only then the low- and high-
    # density regions (<5).
    ci = 5  


    ################################################## 
    # nuclear EoS
 
    # Polytropic exponents excluding the first two ones
    gammas = []  
    for itrope in range(eos_Ntrope-2):
        if debug:
            print("loading gamma from cube #{}".format(ci))
        gammas.append(cube[ci])
        ci += 1


    # Transition ("matching") densities (g/cm^3)
    trans  = [1.0e14, 1.1 * cgs.rhoS] #starting point #TODO QMC-polytrope boundary (1-1.3(-2) * rhoS)? BTW 1.0e14 ~ 0.4*rhoS
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("loading trans from cube #{}".format(ci))
        trans.append(trans[-1] + cgs.rhoS * cube[ci]) 
        ci += 1


    ################################################## 
    # low-density QMC EoS

    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    a     = cube[0] * 1.0e6 * cgs.eV     # (erg)
    alpha = cube[1]                      # untiless
    b     = cube[2] * 1.0e6 * cgs.eV     # (erg)
    beta  = cube[3]                      # unitless
    S     = cgs.Enuc + a + b      # (erg)
    L     = 3.0 * (a * alpha + b * beta) # (erg)
    lowDensity = [a, alpha, b, beta]


    ################################################## 
    # high-density pQCD EoS

    # Perturbative QCD parameters, see Fraga et al. (2014, arXiv:1311.5154) for details
    X = cube[4]
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
    if not struc.realistic:
        logl = -linf

        return logl


    # solve structure 
    if debug:
        print("TOV...")
    #struc.tov(m1 = 1.4 * cgs.Msun)
    struc.tov()



    ################################################## 
    # measurements & constraints

    # strict two-solar-mass constraint
    if struc.maxmass < 1.97:
        logl = -linf

    # strict tidal deformablity constrain
    # TODO: implement correct likelihood distribution instead
    #if struc.TD >= 1000.0:
    #    logl = -linf


    # 4U 1702-429 from Nattila et al 2017
    #mass_1702 = cube[ci] # first measurement
    #rad_1702 = struc.radius_at(mass_1702)
    #logl = gaussian_MR(mass_1702, rad_1702, NSK17)


    
    #build M-R curve
    if debug:
        ic = param_indices['rad_0'] #starting index
        print("building M-R curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['rad_' + str(im)] #this is the index pointing to correct position in cube
        cube[ic] = struc.radius_at(mass)

        if debug:
            print("im = {}, mass = {}, rad = {}, ic = {}".format(im, mass, cube[ic], ic))


    #build rho-P curve
    if debug:
        ic = param_indices['P_0'] #starting index
        print("building rho-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, rho in enumerate(param_indices['rho_grid']):
        ic = param_indices['P_'+str(ir)] #this is the index pointing to correct position in cube
        cube[ic] = struc.eos.pressure(rho)

        if debug:
            print("ir = {}, rho = {}, P = {}, ic = {}".format(ir, rho, cube[ic], ic))


    #build nsat-gamma curve
    if debug:
        ic = param_indices['nsat_0'] #starting index
        print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_grid']):
        ic = param_indices['nsat_'+str(ir)] #this is the index pointing to correct position in cube
        try:
            cube[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat ).G
        except:
            cube[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat )._find_interval_given_density( cgs.rhoS*nsat ).G


        if debug:
            print("ir = {}, nsat = {}, gamma = {}, ic = {}".format(ir, nsat, cube[ic], ic))


    
    print(cube[0:12])
    print(cube[ param_indices['rad_8'] ])

    return logl



##################################################
# run MultiNest

result = pymlsolve( 
            LogLikelihood=myloglike, 
            Prior=myprior, 
	    n_params=n_params,  
	    n_dims=len(parameters),
            n_live_points=50,
            log_zero = linf,
            outputfiles_basename=prefix,
            resume=False,
            verbose=True,
                 )


print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')


for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

import json
with open('%sparams.json' % prefix, 'w') as f:
    json.dump(parameters, f, indent=2)
