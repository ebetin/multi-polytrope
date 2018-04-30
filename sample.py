from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from pymultinest.solve import solve
import os
from structure import structure
import units as cgs
from pQCD import nQCD


if not os.path.exists("chains"): os.mkdir("chains")





##################################################
# Prior function; changes from [0,1] to whatever physical lims
def myprior(cube):

    # XXX rajojen tarkistaminen

    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    cube[0] = cube[0] * (14.0 - 10.0) + 10.0 #a [Mev]
    cube[1] = cube[1] * (0.55 - 0.40) + 0.40 #alpha [unitless]
    cube[2] = cube[2] * (7.5 - 1.5) + 1.5 #b [MeV]
    cube[3] = cube[3] * (2.7 - 1.8) + 1.8 #beta [unitless]

    # Scale parameter of the perturbative QCD, see Frage et al. (2014, arXiv:1311.5154) for details
    cube[4] = cube[4] * (4.0 - 1.0) + 1.0 #X [unitless]

    # Polytropic exponents excluding the first two ones
    cube[5] *= 10.0 #gamma3 [unitless]
    cube[6] *= 10.0 #gamma4 [unitless]

    # Lengths of the first N-1 monotropes (N = # of polytropes)
    cube[7] *= 15.0 #delta_n1 [rhoS]
    cube[8] *= 43.0 #delta_n2 [rhoS]
    cube[9] *= 40.0 #delta_n3 [rhoS]
    
    #cube[5] *= 50.0 #delta_n1 [rhoS] #2-trope

    #cube[5] *= 15.0 #gamma3 [unitless] #3-trope
    #cube[6] *= 50.0 #delta_n1 [rhoS]
    #cube[7] *= 50.0 #delta_n2 [rhoS]

    return cube



# probability function
linf = 100.0
def myloglike(cube):

    logl = 0.0


    # Polytropic exponents excluding the first two ones
    gammas = [cube[5], cube[6]] # 4-trope
    #gammas = [cube[5]] # 3-trope
    #gammas = [] # 2-trope


    # Transition ("matching") densities (g/cm^3)
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS]
    trans.append(trans[-1] + cgs.rhoS * cube[7])
    trans.append(trans[-1] + cgs.rhoS * cube[8])
    trans.append(trans[-1] + cgs.rhoS * cube[9])
    #trans.append(trans[-1] + cgs.rhoS * cube[6]) # 3-trope
    #trans.append(trans[-1] + cgs.rhoS * cube[7])
    #trans.append(trans[-1] + cgs.rhoS * cube[5]) # 2-trope



    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    a = cube[0] * 1.0e6 * cgs.eV        # (erg)
    alpha = cube[1]                     # (untiless)
    b = cube[2] * 1.0e6 * cgs.eV        # (erg)
    beta = cube[3]                      # (unitless)
    #S = 16.0e6 * cgs.eV + a + b         # (erg)
    #L = 3.0 * (a * alpha + b * beta)    # (erg)
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


    struc.tov()

    # two-solar-mass constrain
    if struc.maxmass < 1.97:
        logl = -linf


    return logl



##################################################
# number of dimensions our problem has

#4-trope
parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "gamma4", "trans_delta1", "trans_delta2", "trans_delta3"]

#3-trope
#parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "trans_delta1", "trans_delta2"]

#2-trope
#parameters = ["a", "alpha", "b", "beta", "X", "trans_delta1"]

n_params = len(parameters)

prefix = "chains/18-"




##################################################
# run MultiNest
result = solve( LogLikelihood=myloglike, 
                Prior=myprior, 
	        n_dims=n_params,  
            n_live_points=400,
                outputfiles_basename=prefix,
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
