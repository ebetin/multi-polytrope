from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve

from priors import check_uniform
from structure import structurePolytrope as structure
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
eos_Ntrope = 5 #polytrope order
debug = False  #flag for additional debug printing
phaseTransition = 0 #position of the 1st order transition
#after first two monotropes, 0: no phase transition
#in other words, the first two monotrope do not behave
#like a latent heat (ie. gamma != 0)


##################################################
#auto-generated parameter names for polytropes 

#QMC + pQCD parameters
parameters = ["a", "alpha", "b", "beta", "X"]

#append gammas (start from 3 since two first ones are given by QMC)
for itrope in range(eos_Ntrope-2):
    if itrope + 1 != phaseTransition:
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

Ngrid = 200
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0,   Ngrid),
        'eps_grid':  np.logspace(2.0, 4.3, Ngrid),
        'nsat_gamma_grid': np.linspace(1.1, 15.0, Ngrid),
        'nsat_c2_grid': np.linspace(1.1, 15.0, Ngrid),
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
    lps[0] = check_uniform(cube[0], 12.4, 13.6 ) #a [Mev]
    lps[1] = check_uniform(cube[1], 0.46,  0.53) #alpha [unitless]
    lps[2] = check_uniform(cube[2],  1.6,  5.8 ) #b [MeV]
    lps[3] = check_uniform(cube[3],  2.0,  2.7 ) #beta [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    lps[4] = check_uniform(cube[4],  1.0,  4.0 ) #X [unitless]


    # Polytropic exponents excluding the first two ones
    ci = 5
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
        if itrope + 1 != phaseTransition:
            if debug:
                print("loading gamma from cube #{}".format(ci))
            gammas.append(cube[ci])
            ci += 1
        else:
            gammas.append(0.0)


    # Transition ("matching") densities (g/cm^3)
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #[0.9e14, 1.1 * cgs.rhoS] #starting point BTW 1.0e14 ~ 0.4*rhoS
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
    #struc.tov()
    struc.tov(l=2, m1=1.4 * cgs.Msun) # tidal deformability
    print("params", cube)


    ################################################## 
    # measurements & constraints

    # strict two-solar-mass constraint
    if struc.maxmass < 1.97 and m2:
        logl = -linf

        return logl, blobs

    # strict tidal deformablity constrain
    # LIGO/Virgo Lambda(1.4 M_sun) 90 % credibility limits
    if 70.0 > struc.TD or struc.TD > 580.0:
        logl = -linf

        return logl, blobs

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

    #build eps-P curve
    if debug:
        ic = param_indices['Peps_0'] #starting index
        print("building eps-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, eps in enumerate(param_indices['eps_grid']):
        ic = param_indices['Peps_'+str(ir)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.eos.pressure_edens( eps * 0.001 * cgs.GeVfm_per_dynecm / (cgs.c**2) ) * 1000.0 / cgs.GeVfm_per_dynecm

        if debug:
            print("ir = {}, eps = {}, P = {}, ic = {}".format(ir, eps, blobs[ic], ic))

    #build nsat-gamma curve
    if debug:
        ic = param_indices['nsat_gamma_0'] #starting index
        print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_gamma_grid']):
        ic = param_indices['nsat_gamma_'+str(ir)] #this is the index pointing to correct position in cube

        blobs[ic] = struc.eos.gammaFunction( cgs.rhoS*nsat, flag = 0 )

        if debug:
            print("ir = {}, nsat = {}, gamma = {}, ic = {}".format(ir, nsat, blobs[ic], ic))

    #build nsat-c^2 curve
    if debug:
        ic = param_indices['nsat_c2_0'] #starting index
        print("building nsat-c^2 curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_c2_grid']):
        ic = param_indices['nsat_c2_'+str(ir)] #this is the index pointing to correct position in cube

        blobs[ic] = struc.eos.speed2( struc.eos.pressure( cgs.rhoS * nsat ) )

        if debug:
            print("ir = {}, nsat = {}, c^2 = {}, ic = {}".format(ir, nsat, blobs[ic], ic))

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

#initial guess

if eos_Ntrope == 2: #(trope = 2)
    if phaseTransition == 0:
        pinit = [12.7,  0.475,  3.2,  2.49,  1.2,
        5.0, 1.49127976]
        # ~2M_sun, no PT, Lambda_1.4 ~ 250

elif eos_Ntrope == 3: #(trope = 3)
    if phaseTransition == 0:
        pinit = [12.7,  0.475,  3.2,  2.49,  2.98691193,
        2.095,   3.57470224, 26.0, 1.49127976]
        # ~2M_sun, no PT, Lambda_1.4 ~ 300
    elif phaseTransition == 1:
        pinit = [12.7,  0.475,  3.2,  2.49,  1.16,
        3.57470224, 26.0, 1.49127976]
        # ~2M_sun, PT1, Lambda_1.4 ~ 300

elif eos_Ntrope == 4: #(trope = 4)
    if phaseTransition == 0:
        pinit = [12.7,  0.475,  3.2,  2.49,  2.98691193,  2.75428241,
        2.0493058,   3.57470224, 26.40907385,  1.31246422,  1.49127976]
        # ~2M_sun, no PT, Lambda_1.4 ~ 300
    elif phaseTransition == 1:
        pinit = [12.7,  0.475,  3.2,  2.49,  2.98691193,
        2.41,   3.57470224, 26.40907385,  1.31246422,  1.49127976]
        # ~2M_sun, PT1, Lambda_1.4 ~ 300
    elif phaseTransition == 2:
        pinit = [12.7,  0.475,  3.2,  2.49,  2.98691193,
        2.4,   3.57470224, 8.30907385,  19.41246422,  1.49127976]
        # ~2M_sun, PT2, Lambda_1.4 ~ 300

elif eos_Ntrope == 5: #(trope = 5)
    if phaseTransition == 1:
        pinit = [12.7,  0.475,  3.2,  2.49,  2.98691193,  2.75428241,
        2.2, 3.57470224, 25.00907385, 1.4, 1.31246422,  1.49127976]
        # ~2M_sun, PT1, Lambda_1.4 ~ 300


#initialize small Gaussian ball around the initial point
p0 = [pinit + 0.01*np.random.randn(ndim) for i in range(nwalkers)]

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
        filename = "chains2/chain190527P2PT0.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim) #no restart
        
        # initialize sampler
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2M, backend=backend, pool=pool)

        result = sampler.run_mcmc(p0, 10, progress=True)
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
