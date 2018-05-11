from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve

from priors import transform_uniform


from structure import structure
import units as cgs
from pQCD import nQCD


if not os.path.exists("chains"): os.mkdir("chains")


##################################################
# global flags for different run modes
eos_Ntrope = 3 #polytrope order
debug = True  #flag for additional debug printing


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

#append gammas (start from 3 since two first one are given by QMC)
for itrope in range(eos_Ntrope-2):
    parameters.append("gamma"+str(3+itrope))

#append transition depths
for itrope in range(eos_Ntrope-1):
    parameters.append("trans_delta"+str(1+itrope))

print("Parameters to be sampled are:")
print(parameters)




##################################################

n_params = len(parameters)
prefix = "chains/20-"

# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.






##################################################
# Prior function; changes from [0,1] to whatever physical lims
#def myprior(cube, ndim, nparams):
def myprior(cube):

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
        cube[ci] = transform_uniform(cube[ci], 0.0, 30.0)  #delta_ni [rhoS]
        ci += 1


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

        ## QMC parameters
        0 a
        1 alpha
        2 b
        3 beta

        #pQCD parameters
        4 X

        #nuclear EoS parameters
        gammas
        transition points


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
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #starting point
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
    S     = Enuc + a + b      # (erg)
    L     = 3.0 * (a * alpha + b * beta) # (erg)
    lowDensity = [a, alpha, b, beta]


    ################################################## 
    # high-density pQCD EoS

    # Perturbative QCD parameters, see Frage et al. (2014, arXiv:1311.5154) for details
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
    struc.tov()



    ################################################## 
    # measurements & constraints

    # strict two-solar-mass constrain
    if struc.maxmass < 1.97:
        logl = -linf





    return logl




##################################################
# run MultiNest

result = pymlsolve( 
            LogLikelihood=myloglike, 
            Prior=myprior, 
	    n_dims=n_params,  
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
