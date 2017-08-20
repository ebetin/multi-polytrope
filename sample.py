from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from pymultinest.solve import solve
import os
from structure import structure


if not os.path.exists("chains"): os.mkdir("chains")





##################################################
# Prior function; changes from [0,1] to whatever physical lims
def myprior(cube):

    cube[0] *=  5.0
    cube[1] *=  5.0e3
    cube[2] *=  5.0
    cube[3] *=  2.0e3

    return cube



# probability function
linf = 100.0
def myloglike(cube):

    logl = 0.0

    #unpack EoS
    gammas = []
    Ks     = []
    trans  = []

    gammas.append( cube[0] )
    Ks.append(     cube[1] )
    gammas.append( cube[2] )
    Ks.append(     cube[3] )

    #construct it
    struc = structure(gammas, Ks, trans)
    struc.tov()

    #check validity
    if struc.maxmass < 1.97:
        logl = -linf


    return logl



##################################################
# number of dimensions our problem has

#twop
parameters = ["gamma1", "K1", "gamma2", "K2"]

n_params = len(parameters)






##################################################
# run MultiNest
result = solve( LogLikelihood=myloglike, 
                Prior=myprior, 
	        n_dims=n_params, 
                outputfiles_basename="chains/3-",
                )


print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')


for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
