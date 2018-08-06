from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve

from priors import check_uniform
from structure import structure
import units as cgs
from pQCD import nQCD


from measurements import gaussian_MR
from measurements import NSK17 #1702 measurement

# emcee stuff
import sys
import emcee

np.random.seed(1) #for reproducibility

if not os.path.exists("chains2"): os.mkdir("chains2")



##################################################
# global flags for different run modes
eos_Ntrope = 4 #polytrope order
debug = False  #flag for additional debug printing


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

parameters2 = []

Ngrid = 20
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.1,   Ngrid),
        'rho_grid':  np.logspace(14.3, 16.0, Ngrid),
        'nsat_grid': np.linspace(1.0, 20.0, Ngrid),
               }

#add M-R grid
#ci = n_params #current running index of the parameters list
ci = 0
for im, mass  in enumerate(param_indices['mass_grid']):
    parameters2.append('rad_'+str(im))
    param_indices['rad_'+str(im)] = ci
    ci += 1

#add rho-P grid
for ir, rho  in enumerate(param_indices['rho_grid']):
    parameters2.append('P_'+str(ir))
    param_indices['P_'+str(ir)] = ci
    ci += 1

#add nsat - gamma grid
for ir, nsat  in enumerate(param_indices['nsat_grid']):
    parameters2.append('nsat_'+str(ir))
    param_indices['nsat_'+str(ir)] = ci
    ci += 1

print("Parameters to be only stored (blobs):")
print(len(parameters2))
n_blobs = len(parameters2)



##################################################
# Prior function; changes from [0,1] to whatever physical lims
#def myprior(cube, ndim, nparams):
def myprior(cube):

    #print(cube)

    # Parameters of the QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
    lps = np.empty_like(cube)
    lps[0] = check_uniform(cube[0], 10.0, 14.0 ) #a [Mev]
    lps[1] = check_uniform(cube[1],  0.4,  0.55) #alpha [unitless]
    lps[2] = check_uniform(cube[2],  1.5,  7.5 ) #b [MeV]
    lps[3] = check_uniform(cube[3],  1.8,  2.7 ) #beta [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    lps[4] = check_uniform(cube[4],  1.0,  4.0 ) #X [unitless]


    # Polytropic exponents excluding the first two ones
    ci = 5
    for itrope in range(eos_Ntrope-2):
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

    # M-R measurements
    lps[ci] = check_uniform(cube[ci], 1.0, 2.5)  #M_1702 [Msun]

    return np.sum(lps)





# probability function
linf = np.inf


icalls = 0
def myloglike(cube, m2=False):
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
    #print(cube)
    blobs = np.zeros(n_blobs)

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
        return logl, blobs

    ##################################################
    # build neutron star structure 

    # Construct the EoS
    if debug:
        print("Structure...")
    struc = structure(gammas, trans, lowDensity, highDensity)

    # Is the obtained EoS realistic, e.g. causal?
    if not struc.realistic:
        logl = -linf

        return logl, blobs


    # solve structure 
    if debug:
        print("TOV...")
    #struc.tov(m1 = 1.4 * cgs.Msun)
    struc.tov()
    #print("params ", cube)



    ################################################## 
    # measurements & constraints

    # strict two-solar-mass constraint
    if struc.maxmass < 1.97 and m2:
        logl = -linf

        return logl, blobs

    # strict tidal deformablity constrain
    # TODO: implement correct likelihood distribution instead
    #if struc.TD >= 1000.0:
    #    logl = -linf

    # 4U 1702-429 from Nattila et al 2017
    mass_1702 = cube[ci] # first measurement
    rad_1702 = struc.radius_at(mass_1702)
    logl = gaussian_MR(mass_1702, rad_1702, NSK17)


    ic = 0
    
    #build M-R curve
    if debug:
        ic = param_indices['rad_0'] #starting index
        print("building M-R curve from EoS... (starts from ic = {}".format(ic))

    for im, mass in enumerate(param_indices['mass_grid']):
        ic = param_indices['rad_' + str(im)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.radius_at(mass) 

        if debug:
            print("im = {}, mass = {}, rad = {}, ic = {}".format(im, mass, blobs[ic], ic))


    #build rho-P curve
    if debug:
        ic = param_indices['P_0'] #starting index
        print("building rho-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, rho in enumerate(param_indices['rho_grid']):
        ic = param_indices['P_'+str(ir)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.eos.pressure(rho) 

        if debug:
            print("ir = {}, rho = {}, P = {}, ic = {}".format(ir, rho, blobs[ic], ic))


    #build nsat-gamma curve
    if debug:
        ic = param_indices['nsat_0'] #starting index
        print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_grid']):
        ic = param_indices['nsat_'+str(ir)] #this is the index pointing to correct position in cube
        try:
            blobs[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat ).G
        except:
            blobs[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat )._find_interval_given_density( cgs.rhoS*nsat ).G 


        if debug:
            print("ir = {}, nsat = {}, gamma = {}, ic = {}".format(ir, nsat, blobs[ic], ic))

    return logl, blobs



# combine likelihood and prior
def lnprob(cube):
    lp = myprior(cube)

    if not np.isfinite(lp):
        return -np.inf, np.zeros(n_blobs)
    else:
        ll, blobs = myloglike(cube)
        return lp + ll, blobs

def lnprob2M(cube):
    lp = myprior(cube)

    if not np.isfinite(lp):
        return -np.inf, np.zeros(n_blobs)
    else:
        ll, blobs = myloglike(cube,m2=True)
        return lp + ll, blobs


##################################################
##################################################
##################################################
# MCMC sample


ndim = len(parameters)
nwalkers = 30

#initial guess (trope = 4)

if eos_Ntrope == 4:
    #pinit = [1.30010135e+01, 4.09984639e-01, 2.39018583e+00, 2.61439725e+00,
    #         3.13400608e+00, 1.78222716e+00, 9.40358043e-01, 5.34923160e+00,
    #         2.13772426e+01, 6.69159502e+00, 1.34026217e+00 ]
    #pinit = [11.09019685,    0.51512932,     2.98627603,     2.32939703,
    #         2.8060472,      2.10362911,     0.66582859,     0.70016283,
    #         25.95583397,    2.50928181,     1.07738423] #~1.68M_sun

    pinit = [11.58537289, 0.44584499, 3.3476275,  2.52521747,
             2.29671933,  2.19735491, 1.31372968, 1.02766687,       
             25.51001808, 2.36631282, 1.63793851] # ~2.1M_sun


#initialize small Gaussian ball around the initial point
p0 = [pinit + 0.01*np.random.rand(ndim) for i in range(nwalkers)]

##################################################
#serial v3.0-dev
if False:
    #output
    filename = "chain.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim) #no restart
    
    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)

    result = sampler.run_mcmc(p0, 20)

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
        filename = "chains2/chain180803+.h5"
        backend = emcee.backends.HDFBackend(filename)
        #backend.reset(nwalkers, ndim) #no restart
        
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2M, backend=backend, pool=pool)

        result = sampler.run_mcmc(None, 100, progress=True)


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
