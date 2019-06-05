import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



##################################################
#metadata of the run
eos_Nsegment = 5
Ngrid = 20

parameters2 = []

Ngrid = 200
param_indices = {
        'mass_grid' :np.linspace(0.5, 3.0,   Ngrid),
        #'rho_grid':  np.logspace(14.3, 16.0, Ngrid),
        'rho_grid':  np.logspace(-0.79588, 0.85, Ngrid),
        #'P_grid':  np.logspace(0.0, 4.0, Ngrid)
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

#add rho-P grid
for ir, rho  in enumerate(param_indices['rho_grid']):
    parameters2.append('Prho_'+str(ir))
    param_indices['Prho_'+str(ir)] = ci
    ci += 1

#add P-eps grid
#for ir, press  in enumerate(param_indices['P_grid']):
#    parameters2.append('eps_'+str(ir))
#    param_indices['eps_'+str(ir)] = ci
#    ci += 1

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

##################################################
# read chain

filename = '../chains2/chain190605C.h5'
reader = emcee.backends.HDFBackend(filename)#, name='initialization')


#tau = reader.get_autocorr_time()
#print(tau)
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

labels = [r"$a$", r"$\alpha$", r"$b$", r"$\beta$", r"$X$"]

for itrope in range(eos_Nsegment-2):
    labels.append(r"$\Delta\mu_{{{0}}}$".format((1+itrope)))

for itrope in range(eos_Nsegment-2):
    labels.append(r"$c^2_{{{0}}}$".format(1+itrope))

labels.append(r"$M_{1702}$")

#labels.append(r"$\log L$")


##################################################
# triangle/corner plot

if False:
    fig = corner.corner(all_samples, 
            #quantiles=[0.16, 0.5, 0.84],
            #show_titles=True, 
            #title_kwargs={"fontsize": 12})
            labels=labels)
    
    plt.savefig("triangle_190531C1000B0T1.pdf")


##################################################
# ensemble histograms

# trick to make nice colorbars
# see http://joseph-long.com/writing/colorbars/
def colorbar(mappable, 
        loc="right", 
        orientation="vertical", 
        size="5%", 
        pad=0.05, 
        ticklocation='auto'):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, orientation=orientation, ticklocation=ticklocation)


#--------------------------------------------------
#set up figure
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('axes', labelsize=7)

fig = plt.figure(figsize=(3.54, 3.0)) #single column fig
#fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
gs = plt.GridSpec(1, 1) #, wspace=0.0, hspace=0.35)

axs = []
axs.append( plt.subplot(gs[0,0]) )

for ax in axs:
    ax.minorticks_on()


#M-R
if False:
    nsamples, nblobs = blob_samples.shape
    Nr = 50 #number of radius histogram bins

    mass_grid = param_indices["mass_grid"]
    rad = np.zeros((nsamples, Ngrid))

    rad_grid = np.linspace(9.0, 16.0, Nr)

    #get rad from blobs
    for im, mass  in enumerate(param_indices['mass_grid']):
        ci = param_indices['rad_'+str(im)]
        rad[:, im] = blob_samples[:, ci]


    rad_hist = np.zeros((Nr-1, Ngrid))
    for im, mass  in enumerate(param_indices['mass_grid']):

        radslice = rad[:,im]
        rads = radslice[ radslice > 0.0 ]
        radm = np.mean(rads)
        print("M= {} R= {}".format(mass,radm))

        hist, bin_edges = np.histogram(rads, bins=rad_grid)

        rad_hist[:,im] = (1.0*hist[:])/hist.max()

    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(rad_hist)

    hdata_masked = np.ma.masked_where(rad_hist <= 0.0, rad_hist)

    axs[0].set_xlim((9.0, 16.0))
    axs[0].set_ylim((0.5,  3.0))
    axs[0].set_ylabel("Mass $M$ (M$_{\odot}$)")
    axs[0].set_xlabel("Radius $R$ (km)")

    im = axs[0].imshow(
            hdata_masked.T,
            extent=[rad_grid[0], rad_grid[-1], mass_grid[0], mass_grid[-1] ],
            origin='lower',
            interpolation='nearest',
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect='auto',
            )

    #causality
    rad0 = rad_grid[0]
    mass0 = rad0/(2.95*1.52)
    mass1 = mass_grid[-1]
    rad1 = mass1*2.95*1.52
    axs[0].fill_between([rad0, rad1], [mass0, mass1], [3.0, 3.0], color='orange', visible=True)
    #text(5.0, 2.3, "GR")

    #GR limits
    x_gr = [0, 17.5]
    y_gr = [0, 6.0]
    axs[0].fill_between(x_gr, y_gr, [3.0, 3.0], color='darkorange')
    #ax.text(7.4, 2.3, "Causality")
    #ax.text(5.5, 1.8, "Causality", rotation=39, size=10)

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("mass_radius_190531C1000B0T1.pdf")




##################################################
# rho - P
if False:
    nsamples, nblobs = blob_samples.shape
    Nr = 50 #number of radius histogram bins

    rho_grid = param_indices["rho_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.logspace(-0.3, 4.0, Nr)

    #get P from blobs
    for ir, rho  in enumerate(param_indices['rho_grid']):
        ci = param_indices['Prho_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    press_hist = np.zeros((Nr-1, Ngrid))
    for ir, rho  in enumerate(param_indices['rho_grid']):
        pressslice = press[:,ir]
        press_s = pressslice[ pressslice > 0.0 ]
        pressm = np.mean(press_s)
        print("rho= {} P= {}".format(rho,pressm))

        hist, bin_edges = np.histogram(press_s, bins=press_grid)
        press_hist[:,ir] = (1.0*hist[:])/hist.max()


    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(press_hist)

    hdata_masked = np.ma.masked_where(press_hist <= 0.0, press_hist)

    #axs[0].set_xlim((1.0e14, 1.0e16))
    axs[0].set_xlim((0.16, 7))
    #axs[0].set_ylim((1.0e33,  1.0e37))
    axs[0].set_ylim((0.5,  1.e4))
    axs[0].set_ylabel(r"Pressure $P$ (MeV/fm$^3$)")
    axs[0].set_xlabel(r"Density $\rho$ (1/fm$^3$)")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    press_grid = press_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(rho_grid, press_grid)


    im = axs[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("rho_P_190603C340B0T1.pdf")


##################################################
# eps - P
if False:
    nsamples, nblobs = blob_samples.shape
    Nr = 100 #number of radius histogram bins

    rho_grid = param_indices["eps_grid"]
    press = np.zeros((nsamples, Ngrid))
    press_grid = np.logspace(-0.3, 4.0, Nr)

    #get P from blobs
    for ir, eps  in enumerate(param_indices['eps_grid']):
        ci = param_indices['Peps_'+str(ir)]
        press[:, ir] = blob_samples[:, ci]

    press_hist = np.zeros((Nr-1, Ngrid))
    for ir, eps  in enumerate(param_indices['eps_grid']):
        pressslice = press[:,ir]
        press_s = pressslice[ pressslice > 0.0 ]
        pressm = np.mean(press_s)
        print("eps= {} P= {}".format(eps,pressm))

        hist, bin_edges = np.histogram(press_s, bins=press_grid)
        press_hist[:,ir] = (1.0*hist[:])/hist.max()


    #print(rad.shape)
    #print(blob_samples[:,0])
    #print(press_hist)

    hdata_masked = np.ma.masked_where(press_hist <= 0.0, press_hist)

    axs[0].set_xlim((1.0e2, 2.0e4))
    #axs[0].set_ylim((1.0e33,  1.0e37))
    axs[0].set_ylim((0.5,  1.e4))
    axs[0].set_ylabel(r"Pressure $P$ (MeV/fm$^3$)")
    axs[0].set_xlabel(r"Energy density $\epsilon$ (MeV/fm$^3$)")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    press_grid = press_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(rho_grid, press_grid)


    im = axs[0].pcolormesh(
            X,Y,
            hdata_masked,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("eps_P_190605C10B0T1.pdf")

##################################################
# n - gamma
if False:
    nsamples, nblobs = blob_samples.shape
    Ng = 100

    nsat_grid = param_indices["nsat_gamma_grid"]
    gamma = np.zeros((nsamples, Ngrid))
    gamma_grid = np.linspace(0.0, 5.0, Ng)

    #get gamma from blobs
    for ir, nsat  in enumerate(param_indices['nsat_gamma_grid']):
        ci = param_indices['nsat_gamma_'+str(ir)]
        gamma[:, ir] = blob_samples[:, ci]

    gamma_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_gamma_grid']):
        gammaslice = gamma[:,ir]
        gammas = gammaslice[ gammaslice > 0.0 ]
        gammam = np.mean( gammas )
        print("n= {} G= {}".format(nsat, gammam))

        hist_ng, bin_edges = np.histogram(gammas, bins=gamma_grid)
        gamma_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()


    hdata_masked_ng = np.ma.masked_where(gamma_hist <= 0.0, gamma_hist)

    axs[0].set_xlim((1.0, 15.0))
    axs[0].set_ylim((0.0, 5.0))
    axs[0].set_ylabel(r"$\Gamma$")
    axs[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    gamma_grid = gamma_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, gamma_grid)

    im_ng = axs[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("n_gamma_190605C10B0T1.pdf")

##################################################
# n - c^2
if True:
    nsamples, nblobs = blob_samples.shape
    Ng = 100

    nsat_grid = param_indices["nsat_c2_grid"]
    c2 = np.zeros((nsamples, Ngrid))
    c2_grid = np.linspace(0.0, 1.0, Ng)

    #get c2 from blobs
    for ir, nsat  in enumerate(param_indices['nsat_c2_grid']):
        ci = param_indices['nsat_c2_'+str(ir)]
        c2[:, ir] = blob_samples[:, ci]

    c2_hist = np.zeros((Ng-1, Ngrid))
    for ir, nsat  in enumerate(param_indices['nsat_c2_grid']):
        c2slice = c2[:,ir]
        c2s = c2slice[ c2slice > 0.0 ]
        c2m = np.mean( c2s )
        print("n= {} G= {}".format(nsat, c2m))

        hist_ng, bin_edges = np.histogram(c2s, bins=c2_grid)
        c2_hist[:,ir] = (1.0*hist_ng[:])/hist_ng.max()


    hdata_masked_ng = np.ma.masked_where(c2_hist <= 0.0, c2_hist)

    axs[0].set_xlim((1.0, 15.0))
    axs[0].set_ylim((0.0, 1.0))
    axs[0].set_ylabel(r"$c^2$")
    axs[0].set_xlabel(r"Density $n$ ($n_{\mathrm{sat}})$")

    c2_grid = c2_grid[0:-1] #scale grids because histogram does not save last bin
    X,Y=np.meshgrid(nsat_grid, c2_grid)

    im_ng = axs[0].pcolormesh(
            X,Y,
            hdata_masked_ng,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            )

    cb = colorbar(im_ng, loc="top", orientation="horizontal", size="3%", pad=0.03, ticklocation="top")

    plt.savefig("n_c2_190605C10B0T1.pdf")

