from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve



from structure import structure
import units as cgs
from pQCD import nQCD


if not os.path.exists("chains"): os.mkdir("chains")


##################################################
# global flags for different run modes
#eos_mode = "twop"
eos_Ntrope = 3

debug = True



##################################################
# Prior function; changes from [0,1] to whatever physical lims
def myprior(cube, ndim, nparams):

    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    cube[0] = cube[0] * (14.0 - 10.0) + 10.0 #a [Mev]
    cube[1] = cube[1] * (0.55 - 0.40) + 0.40 #alpha [unitless]
    cube[2] = cube[2] * (7.5 - 1.5) + 1.5 #b [MeV]
    cube[3] = cube[3] * (2.7 - 1.8) + 1.8 #beta [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) for details
    cube[4] = cube[4] * (4.0 - 1.0) + 1.0 #X [unitless]

    # Polytropic exponents excluding the first two ones
    #cube[5] *= 15.0 #gamma3 [unitless]
    #cube[6] *= 15.0 #gamma4 [unitless]

    # Lengths of the first N-1 monotropes (N = # of polytropes)
    #cube[7] *= 15.0 #delta_n1 [rhoS]
    #cube[8] *= 15.0 #delta_n2 [rhoS]
    #cube[9] *= 15.0 #delta_n3 [rhoS]
    
    cube[5] *= 50.0 #delta_n1 [rhoS]

    return cube



# probability function
linf = 100.0
def myloglike(cube):
    """
        General likelihood function that builds the EoS and solves TOV-structure
        equations for it.

        Parameters from 0:nDim are sampled, everything beyond that are just 
        carried along.

        ## QMC parameters
        0 a
        1 alpha
        2 b
        3 beta

        #pQCD parameters
        4 X

        #nuclear EoS parameters


    """

    logl = 0.0 #total likelihood
    cubei = 5  #general running index that maps cube array to real EoS parameters


    # Polytropic exponents excluding the first two ones
    #if   eos_mode == "twop":
    #    gammas = [] # 2-trope
    #elif eos_mode == "threep"
    #    gammas = [cube[5]]
    #elif eos_mode == "quoadp"
    #    gammas = [cube[5], cube[6]] # 4-trope

 
    gammas = []  # Polytropic exponents excluding the first two ones
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("loading gamma from cube #{}".format(cubei))
        gammas.append(cube[cubei])
        cubei += 1


    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] # Transition ("matching") densities (g/cm^3)
    for itrope in range(eos_Ntrope-1):
        if debug:
            print("loading trans from cube #{}".format(cubei))
        trans.append(trans[-1] + cgs.rhoS * cube[cubei]) 


    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS]
    #trans.append(trans[-1] + cgs.rhoS * cube[7])
    #trans.append(trans[-1] + cgs.rhoS * cube[8])
    #trans.append(trans[-1] + cgs.rhoS * cube[9])
     trans.append(trans[-1] + cgs.rhoS * cube[5]) # 2-trope


    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    a = cube[0] * 1.0e6 * cgs.eV # (erg)
    alpha = cube[1] # untiless
    b = cube[2] * 1.0e6 * cgs.eV # (erg)
    beta = cube[3] # unitless
    S = 16.0e6 * cgs.eV + a + b # (erg)
    L = 3.0 * (a * alpha + b * beta) # (erg)
    lowDensity = [a, alpha, b, beta]


    # Perturbative QCD parameters, see Frage et al. (2014, arXiv:1311.5154) for details
    X = cube[4]
    muQCD = 2.6 # Transition (matching) chemical potential where pQCD starts (GeV)
    highDensity = [muQCD, X]


    # Check that last transition (matching) point is large enough
    if nQCD(muQCD, X) * cgs.mB <= trans[-1]:
        logl = -linf

        return logl


    # Construct the EoS
    struc = structure(gammas, trans, lowDensity, highDensity)

    # Is the obtained EoS realistic, e.g. causal?
    if not struc.realistic:
        logl = -linf

        return logl


    # solve structure 
    struc.tov()

    ################################################## 
    # measurements & constraints

    # strict two-solar-mass constrain
    if struc.maxmass < 1.97:
        logl = -linf





    return logl



##################################################
# number of dimensions our problem has

#4-trope
#parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "gamma4", "trans_delta1", "trans_delta2", "trans_delta3"]

#3-trope
parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "trans_delta1", "trans_delta2"]

#2-trope
#parameters = ["a", "alpha", "b", "beta", "X", "trans_delta1"]

n_params = len(parameters)
prefix = "chains/14-"




##################################################
# run MultiNest

result = pymlsolve( 
        LogLikelihood=myloglike, 
        Prior=myprior, 
	n_dims=n_params,  
        n_live_points=100,
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
