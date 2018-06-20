import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt



##################################################
#metadata of the run
eos_Ntrope = 4
Ngrid = 20

parameters2 = []

Ngrid = 20
param_indices = {
        'mass_grid' :np.linspace(0.5, 2.8,   Ngrid),
        'rho_grid':  np.logspace(14.3, 16.0, Ngrid),
        'nsat_grid': np.linspace(1.0, 20.0,  Ngrid),
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

##################################################
# read chain

filename = '../chains2/chain2.h5'
reader = emcee.backends.HDFBackend(filename)


tau = reader.get_autocorr_time()
print(tau)
#burnin = int(2*np.max(tau))
#thin = int(0.5*np.min(tau))

burnin = 0
thin = 1

samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
blob_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat blob shape: {0}".format(blob_samples.shape))

#all_samples = np.concatenate((
#    samples, 
#    log_prob_samples[:, None], 
#    blob_samples[:, None]
#), axis=1)

all_samples = samples

##################################################
#create labels
#labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
#labels += ["log prob", "log prior"]

labels = [r"$a$", r"$\alpha$", r"$b$", r"$X$"]

for itrope in range(eos_Ntrope-2):
    labels.append(r"$\gamma_{{{0}}}$".format((3+itrope)))

for itrope in range(eos_Ntrope-1):
    labels.append(r"$\n_{{{0}}}$".format((1+itrope)))

labels.append(r"$M_{1702}$")

##################################################
# triangle/corner plot

if True:
    fig = corner.corner(all_samples, 
            #quantiles=[0.16, 0.5, 0.84],
            #show_titles=True, 
            #title_kwargs={"fontsize": 12})
            labels=labels)
    
    plt.savefig("triangle.pdf")


##################################################
# ensemble histograms





