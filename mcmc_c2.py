from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.solve import solve as pymlsolve

from priors import check_uniform
from structure import structureC2AGKNV as structure
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
eos_Nsegment = 5 #polytrope order
debug = True  #flag for additional debug printing


##################################################
#auto-generated parameter names for c2 interpolation

#QMC + pQCD parameters
parameters = ["a", "alpha", "b", "beta", "X"]

#append chemical potential depths (NB last one will be determined)#XXX
for itrope in range(eos_Nsegment-2):
    parameters.append("mu_delta"+str(1+itrope))

#append speed of sound squared (NB last one will be determined)#XXX
for itrope in range(eos_Nsegment-2):
    parameters.append("speed"+str(1+itrope))

#finally add individual object masses (needed for measurements)
parameters.append("mass_1702")


print("Parameters to be sampled are:")
print(parameters)


n_params = len(parameters)
prefix = "chains/C1-"


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

parameters2 = []

Ngrid = 20
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0,   Ngrid),
        'rho_grid':  np.logspace(14.3, 16.0, Ngrid),
        'nsat_grid': np.linspace(1.1, 40.0, Ngrid),
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

#add nsat - gamma grid #XXX what to do with this?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    lps[0] = check_uniform(cube[0], 12.4, 13.6 ) #a [Mev]
    lps[1] = check_uniform(cube[1], 0.46,  0.53) #alpha [unitless]
    lps[2] = check_uniform(cube[2],  1.6,  5.8 ) #b [MeV]
    lps[3] = check_uniform(cube[3],  2.0,  2.7 ) #beta [unitless]

    # Scale parameter of the perturbative QCD, see Fraga et al. (2014, arXiv:1311.5154) 
    # for details
    lps[4] = check_uniform(cube[4],  1.0,  4.0 ) #X [unitless]


    # Chemical potential depths #XXX
    ci = 5
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("prior for mu_delta from cube #{}".format(ci))
        lps[ci] = check_uniform(cube[ci], 0.0, 1.8)  #delta_mui [GeV]
        ci += 1


    # Matching speed of sound squared excluding the last one#XXX
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("prior for c^2 from cube #{}".format(ci))
        lps[ci] = check_uniform(cube[ci], 0.0, 1.0)  #c_i^2 [unitless]
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

        #interpolation parameters:
        matching chemical potentials
        matching speed of sound squared

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

    # Transition ("matching") densities (g/cm^3)
    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #[0.9e14, 1.1 * cgs.rhoS] #starting point BTW 1.0e14 ~ 0.4*rhoS
 
    # Matching chemical potentials (GeV)#XXX
    mu_deltas = []  
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("loading mu_deltas from cube #{}".format(ci))
        mu_deltas.append(cube[ci])
        ci += 1

    speed2 = []
    # Speed of sound squareds (unitless)#XXX
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("loading speed2 from cube #{}".format(ci))
        speed2.append(cube[ci]) 
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
    struc = structure(mu_deltas, speed2, trans, lowDensity, highDensity)#XXX

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


    #build rho-P curve
    if debug:
        ic = param_indices['P_0'] #starting index
        print("building rho-P curve from EoS... (starts from ic = {}".format(ic))

    for ir, rho in enumerate(param_indices['rho_grid']):
        ic = param_indices['P_'+str(ir)] #this is the index pointing to correct position in cube
        blobs[ic] = struc.eos.pressure(rho) 

        if debug:
            print("ir = {}, rho = {}, P = {}, ic = {}".format(ir, rho, blobs[ic], ic))


    #build nsat-gamma curve#XXX what to do with this?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if debug:
        ic = param_indices['nsat_0'] #starting index
        print("building nsat-gamma curve from EoS... (starts from ic = {}".format(ic))

    for ir, nsat in enumerate(param_indices['nsat_grid']):
        ic = param_indices['nsat_'+str(ir)] #this is the index pointing to correct position in cube
        #try:
        print("KKKKKK", struc.eos._find_interval_given_density( cgs.rhoS*nsat ))
        blobs[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat ).gammaFunction( cgs.rhoS*nsat )
        #except:
        #    blobs[ic] = struc.eos._find_interval_given_density( cgs.rhoS*nsat )._find_interval_given_density( cgs.rhoS*nsat ).gammaFunction( cgs.rhoS*nsat ) 


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

#initial guess

if eos_Nsegment == 5: #(segments = 5)
    pinit = [12.7,  0.475,  3.2,  2.49,  1.6,
    0.0794198894924294, 0.48, 0.36, 0.315, 0.36, 0.27, 1.49127976]
    # ~1.97M_sun, Lambda_1.4 ~ 465


#initialize small Gaussian ball around the initial point
p0 = [pinit + 0.01*np.random.randn(ndim) for i in range(nwalkers)]#XXX rand vs randn

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
        filename = "chains2/chain190516C.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim) #no restart
        
        # initialize sampler
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2M, backend=backend, pool=pool)

        result = sampler.run_mcmc(p0, 5, progress=True)
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
