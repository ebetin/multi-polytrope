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

    cube[0] = cube[0] * (14.0 - 10.0) + 10.0 #a
    cube[1] = cube[1] * (0.55 - 0.40) + 0.40 #alpha
    cube[2] = cube[2] * (7.5 - 1.5) + 1.5 #b
    cube[3] = cube[3] * (8.0 - 1.7) + 1.7 #beta
    cube[4] = cube[4] * (4.0 - 1.0) + 1.0 #X

    cube[5] *= 15.0 #gamma3
    cube[6] *= 15.0 #gamma4

    cube[7] *= 15.0 #delta_n1
    cube[8] *= 15.0 #delta_n2
    cube[9] *= 15.0 #delta_n3
    
    #cube[0] = cube[0] * (30.0-1.0) + 1.0

    return cube



# probability function
linf = 100.0
def myloglike(cube):

    logl = 0.0

    #unpack EoS

    gammas = [cube[5], cube[6]]


    #trans  = [0.2 * cgs.rhoS, 1.1 * cgs.rhoS, cube[0] * cgs.rhoS]
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS]
    trans.append(trans[-1] + cgs.rhoS * cube[7])
    trans.append(trans[-1] + cgs.rhoS * cube[8])
    trans.append(trans[-1] + cgs.rhoS * cube[9])


    #a = 13.4e6 * cgs.eV
    #alpha = 0.514
    #b = 5.62e6 * cgs.eV
    #beta = 2.436

    a = cube[0] * 1.0e6 * cgs.eV
    alpha = cube[1]
    b = cube[2] * 1.0e6 * cgs.eV
    beta = cube[3]
    lowDensity = [a, alpha, b, beta]

    #X = 1.2
    X = cube[4]
    muQCD = 2.6 # (GeV)
    highDensity = [muQCD, X]


    if nQCD(muQCD, X) * cgs.mB <= trans[-1]:
        logl = -linf

        return logl

    #construct it
    struc = structure(gammas, trans, lowDensity, highDensity)

    if struc.realistic == False:
        logl = -linf

        return logl

    struc.tov()

    #check validity
    if struc.maxmass < 1.97:
        logl = -linf


    return logl



##################################################
# number of dimensions our problem has

#twop
#parameters = ["gamma1", "K1", "gamma2", "K2"]

#quadrutrope
parameters = ["a", "alpha", "b", "beta", "X", "gamma3", "gamma4", "trans_delta1", "trans_delta2", "trans_delta3"]

n_params = len(parameters)






##################################################
# run MultiNest
result = solve( LogLikelihood=myloglike, 
                Prior=myprior, 
	        n_dims=n_params,  
            n_live_points=400,
                outputfiles_basename="chains/7-",
                )


print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')


for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
