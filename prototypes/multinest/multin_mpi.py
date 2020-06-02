# test multinest fitting on a small scale
#from __future__ import absolute_import, unicode_literals, print_function

import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os

import mpi4py #loading mpi4py should automatically switch state to parallel 


try: os.mkdir('chains')
except OSError: pass



# probability function, taken from the eggbox problem.
def myprior(cube):
	return cube * 10 * pi

def myloglike(cube):
	chi = (cos(cube / 2.)).prod()
	return (2. + chi)**5


# number of dimensions our problem has
parameters = ["x", "y"]
n_params = len(parameters)

# name of the output files
prefix = "chains/2-"


# run MultiNest
result = solve(
        LogLikelihood=myloglike, 
        Prior=myprior, 
	n_dims=n_params, 
        outputfiles_basename=prefix, 
        verbose=True)



if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    print()
    print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
    print()
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
    	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
    
    # make marginal plots by running:
    # $ python multinest_marginals.py chains/3-
    # For that, we need to store the parameter names:
    
    
    import json
    with open('%sparams.json' % prefix, 'w') as f:
    	json.dump(parameters, f, indent=2)
    
