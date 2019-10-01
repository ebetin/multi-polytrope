#from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
import os

#from pymultinest.solve import solve as pymlsolve

from priors import check_uniform
from structure import structureC2AGKNV as structure
import units as cgs
from pQCD import nQCD

from measurements import gaussian_MR
from measurements import NSK17 #1702 measurement

from measurements import measurement_MR
from measurements import SHB18_6304_He #6304 measurement
from measurements import SHB18_6397_He #6397 measurement
from measurements import SHB18_M28_He  #M28 measurement
from measurements import SHB18_M30_H   #M30 measurement
from measurements import SHB18_X7_H    #X7 measurement
from measurements import SHB18_X5_H    #X5 measurement
from measurements import SHB18_wCen_H  #wCen measurement
from measurements import SHS18_M13_H   #M13 measurement
from measurements import NKS15_1724    #1724 measurement
from measurements import NKS15_1810    #1810 measurement


from measurements import measurement_M
from measurements import J0348
from measurements import J0740

from measurements import measurement_TD
from measurements import GW170817

from scipy.stats import norm

# emcee stuff
import sys
import emcee

np.random.seed(1) #for reproducibility

if not os.path.exists("chains"): os.mkdir("chains")


##################################################
# global flags for different run modes
eos_Nsegment = 5 #polytrope order
debug = True  #flag for additional debug printing


##################################################
#auto-generated parameter names for c2 interpolation

#QMC + pQCD parameters
parameters = ["a", "alpha", "b", "beta", "X"]

#append chemical potential depths (NB last one will be determined)
for itrope in range(eos_Nsegment-2):
    parameters.append("mu_delta"+str(1+itrope))

#append speed of sound squared (NB last one will be determined)
for itrope in range(eos_Nsegment-2):
    parameters.append("speed"+str(1+itrope))

#GW170817
parameters.append("chrip_mass_GW170817")
parameters.append("mass_ratio_GW170817")

#finally add individual object masses (needed for measurements)
parameters.append("mass_0432")
parameters.append("mass_6620")

parameters.append("mass_1702")

parameters.append("mass_6304")
parameters.append("mass_6397")
parameters.append("mass_M28")
parameters.append("mass_M30")
parameters.append("mass_X7")
parameters.append("mass_X5")
parameters.append("mass_wCen")
parameters.append("mass_M13")
parameters.append("mass_1724")
parameters.append("mass_1810")


print("Parameters to be sampled are:")
print(parameters)


n_params = len(parameters)
prefix = "chains/C1-"


##################################################
# next follows the parameters that we save but do not sample
# NOTE: For convenience, the correct index is stored in the dictionary,
#       this way they can be easily changed or expanded later on.

parameters2 = []

Ngrid = 200
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0, Ngrid),
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


    # Chemical potential depths
    ci = 5
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("prior for mu_delta from cube #{}".format(ci))
        lps[ci] = check_uniform(cube[ci], 0.0, 1.8)  #delta_mui [GeV]
        ci += 1


    # Matching speed of sound squared excluding the last one
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("prior for c^2 from cube #{}".format(ci))
        lps[ci] = check_uniform(cube[ci], 0.0, 1.0)  #c_i^2 [unitless]
        ci += 1

    # TD measurements
    lps[ci] = norm.logpdf(cube[ci], 1.186, 0.0006079568319312625)  #Chirp mass (GW170817) [Msun]
    lps[ci+1] = check_uniform(cube[ci+1], 0.0, 1.0)  #Mass ratio (GW170817) #XXX Is this ok?

    ci += 2

    # M measurements
    lps[ci]   = measurement_M(cube[ci], J0348)   #m0432 [Msun]
    lps[ci+1] = measurement_M(cube[ci+1], J0740) #m6620 [Msun]

    ci += 2

    # M-R measurements
    lps[ci] = check_uniform(cube[ci], 1.0, 2.5)  #M_1702 [Msun]

    lps[ci+1] = check_uniform(cube[ci+1], 0.5, 2.7)  #M_6304 [Msun]
    lps[ci+2] = check_uniform(cube[ci+2], 0.5, 2.0)  #M_6397 [Msun]
    lps[ci+3] = check_uniform(cube[ci+3], 0.5, 2.8)  #M_M28 [Msun]
    lps[ci+4] = check_uniform(cube[ci+4], 0.5, 2.5)  #M_M30 [Msun]
    lps[ci+5] = check_uniform(cube[ci+5], 0.5, 2.7)  #M_X7 [Msun]
    lps[ci+6] = check_uniform(cube[ci+6], 0.5, 2.7)  #M_X5 [Msun]
    lps[ci+7] = check_uniform(cube[ci+7], 0.5, 2.5)  #M_wCen [Msun]
    lps[ci+8] = check_uniform(cube[ci+8], 0.8, 2.4)  #M_M13 [Msun]
    lps[ci+9] = check_uniform(cube[ci+9], 0.8, 2.5)  #M_1724 [Msun]
    lps[ci+10] = check_uniform(cube[ci+10], 0.8, 2.5)  #M_1810 [Msun]

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
 
    # Matching chemical potentials (GeV)
    mu_deltas = []  
    for itrope in range(eos_Nsegment-2):
        if debug:
            print("loading mu_deltas from cube #{}".format(ci))
        mu_deltas.append(cube[ci])
        ci += 1

    speed2 = []
    # Speed of sound squareds (unitless)
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
    struc = structure(mu_deltas, speed2, trans, lowDensity, highDensity)

    # Is the obtained EoS realistic, e.g. causal?
    if not struc.realistic:
        logl = -linf

        return logl, blobs

    # Masses GW170817
    mass1_GW170817 = (1.0 + cube[ci+1])**0.2 / (cube[ci+1])**0.6 * cube[ci]
    mass2_GW170817 = mass1_GW170817 * cube[ci+1]

    ci += 2

    # solve structure 
    if debug:
        print("TOV...")
    #struc.tov()
    #struc.tov(l=2, m1=1.4 * cgs.Msun) # tidal deformability
    struc.tov(l=2, m1=mass1_GW170817*cgs.Msun, m2=mass2_GW170817*cgs.Msun) # tidal deformabilities
    print("params", cube)

    ################################################## 
    # measurements & constraints

    # Mass measurement of PSR J0348+0432 from Antoniadis et al 2013 arXiv:1304.6875
    # and PSR J0740+6620 from Cromartie et al 2019 arXiv:1904.06759
    if m2:
        mmax = struc.maxmass

        m0432 = cube[ci]
        m6620 = cube[ci+1]

        if m0432 > mmax or m6620 > mmax:
            logl = -linf

            return logl, blobs

    ci += 2

    # masses, MR measurements
    mass_1702 = cube[ci]
    mass_6304 = cube[ci+1]
    mass_6397 = cube[ci+2]
    mass_M28 = cube[ci+3]
    mass_M30 = cube[ci+4]
    mass_X7 = cube[ci+5]
    mass_X5 = cube[ci+6]
    mass_wCen = cube[ci+7]
    mass_M13 = cube[ci+8]
    mass_1724 = cube[ci+9]
    mass_1810 = cube[ci+10]

    masses = [ mass_1702, mass_6304, mass_6397, mass_M28,
    mass_M30, mass_X7, mass_X5, mass_wCen, mass_M13,
    mass_1724, mass_1810, ]

    # All stars have to be lighter than the max mass limit
    if any(m > struc.maxmass for m in masses):
        logl = -linf

        return logl, blobs

    # 4U 1702-429 from Nattila et al 2017, arXiv:1709.09120
    rad_1702 = struc.radius_at(mass_1702)
    logl = logl + gaussian_MR(mass_1702, rad_1702, NSK17)

    # NGC 6304 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_6304 = struc.radius_at(mass_6304)
    logl = logl + measurement_MR(mass_6304, rad_6304, SHB18_6304_He)

    # NGC 6397 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_6397 = struc.radius_at(mass_6397)
    logl = logl + measurement_MR(mass_6397, rad_6397, SHB18_6397_He)

    # M28 with He atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_M28 = struc.radius_at(mass_M28)
    logl = logl + measurement_MR(mass_M28, rad_M28, SHB18_M28_He)

    # M30 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_M30 = struc.radius_at(mass_M30)
    logl = logl + measurement_MR(mass_M30, rad_M30, SHB18_M30_H)

    # X7 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_X7 = struc.radius_at(mass_X7)
    logl = logl + measurement_MR(mass_X7, rad_X7, SHB18_X7_H)

    # X5 with H atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_X5 = struc.radius_at(mass_X5)
    logl = logl + measurement_MR(mass_X5, rad_X5, SHB18_X5_H)

    # wCen with H atmosphere from Steiner et al 2018, arXiv:1709.05013
    rad_wCen = struc.radius_at(mass_wCen)
    logl = logl + measurement_MR(mass_wCen, rad_wCen, SHB18_wCen_H)

    # M13 with H atmosphere from Shaw et al 2018, arXiv:1803.00029
    rad_M13 = struc.radius_at(mass_M13)
    logl = logl + measurement_MR(mass_M13, rad_M13, SHS18_M13_H)

    # 4U 1724-307 from Natiila et al 2016, arXiv:1509.06561
    rad_1724 = struc.radius_at(mass_1724)
    logl = logl + measurement_MR(mass_1724, rad_1724, NKS15_1724)

    # SAX J1810.8-260 from Natiila et al 2016, arXiv:1509.06561
    rad_1810 = struc.radius_at(mass_1810)
    logl = logl + measurement_MR(mass_1810, rad_1810, NKS15_1810) 


    # GW170817, tidal deformability
    if struc.TD > 1600.0 or struc.TD2 > 1600.0:
        logl = -linf

        return logl, blobs

    logl = logl + measurement_TD(struc.TD, struc.TD2, GW170817)   

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
nwalkers = 2 * ndim # XXX This isn't ideal, I guess

#initial guess

if eos_Nsegment == 5: #(segments = 5)
    #pinit = [12.7,  0.475,  3.2,  2.49,  1.6,
    #0.0069, 0.102, 0.193, 0.054, 0.32, 0.66,
    #2.01, 2.14, 1.49127976, 2.0, 1.6
    #,]
    # ~2.1M_sun, Lambda_1.4 ~ 310

    pinit = [12.6869347, 0.475763805, 3.20367232, 2.50232899,
    1.59577143, 0.00776464407, 0.0805753327, 0.184698311,
    0.0585161595, 0.331041743, 0.657182637,
    1.186, 0.85, 2.01, 2.14, 1.49127976, 2.0, 1.6, 1.6,
    1.6, 1.4, 1.0, 1.6, 1.8, 1.4, 1.4
    ,]
    # ~2.2M_sun, Lambda_1.4 ~ 410


#initialize small Gaussian ball around the initial point
p0 = [pinit + 0.001*np.random.randn(ndim) for i in range(nwalkers)]

##################################################
#serial v3.0-dev
if False:

    #output
    filename = prefix+'run.h5'

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim) #no restart
    
    # initialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2M, backend=backend)

    result = sampler.run_mcmc(p0, 10000)

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
    #from emcee.utils import MPIPool

    #even out all workers
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        #output
        filename = prefix+'run.h5'

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
