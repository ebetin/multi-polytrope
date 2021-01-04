import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

##################################
# Parses
def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--plots', 
            dest='flag_plot', 
            default=0,
            type=int,
            help='Ploted fig(s): 1 (fig 1), 2 (fig 2), 3 (both) or 0 (none, default)')

    parser.add_argument('-f', '--file', 
            dest='file', 
            default="",
            type=str,
            help='File name (e.g. M1_S3_PT0-s587198-w5-g200-n20000)')

    parser.add_argument('-m', '--ml', 
            dest='flag_ml', 
            default=True,
            type=bool,
            help='Usage of the maximum likelihood estimate (default: True)')

    parser.add_argument('--debug', 
            dest='debug', 
            default=False,
            type=bool,
            help='Debug mode (default: False)')

    parser.add_argument('--lhparam', 
            dest='lhparams', 
            default=3,
            type=int,
            help='Number of low- and high-density EoS parameters (default: 3)')

    parser.add_argument('--mparam', 
            dest='mparams', 
            default=15,
            type=int,
            help='Mass-measurement parameters (default: 15)')

    args = parser.parse_args()

    return args

##################################
# Probability-related functions

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

##################################
#Main code

args = parse_cli()

flag_plot = args.flag_plot
flag_ml = args.flag_ml
debug = args.debug

prefix = args.file#'M1_S3_PT0-s587198-w5-g200-n20000'
filename = 'chains/csc/'+prefix+'run.h5'        
backend = emcee.backends.HDFBackend(filename,read_only=True)

if flag_plot == 1 or flag_plot == 3:
    fig1 = plt.figure()
    chain = backend.get_chain(discard=0)[:, :, 0].T

    # Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 15)).astype(int)
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain[:, :n])
        new[i] = autocorr_new(chain[:, :n])

    if flag_ml:
        import celerite
        from celerite import terms
        from scipy.optimize import minimize

        def autocorr_ml(y, thin=1, c=5.0):
            # Compute the initial estimate of tau using the standard method
            init = autocorr_new(y, c=c)
            z = y[:, ::thin]
            N = z.shape[1]

            # Build the GP model
            tau = max(1.0, init / thin)
            if debug:
                print(z.min(), z.max(), tau)
                print(np.log(0.9 * np.var(z)).min(), np.log(0.9 * np.var(z)).max(), -np.log(tau) )
            kernel = terms.RealTerm(
                    np.log(0.9 * np.var(z)), -np.log(tau), bounds=[(-11.0, 11.0), (-np.log(N), 0.0)]
                )
            if debug:
                print( np.log(0.1 * np.var(z)).min(),  np.log(0.1 * np.var(z)).max(), -np.log(0.5 * tau))
            kernel += terms.RealTerm(
                np.log(0.1 * np.var(z)),
                -np.log(0.5 * tau),
                bounds=[(-11.0, 11.0), (-np.log(N), 0.0)],
            )
            gp = celerite.GP(kernel, mean=np.mean(z))
            gp.compute(np.arange(z.shape[1]))

            # Define the objective
            def nll(p):
                # Update the GP model
                gp.set_parameter_vector(p)

                # Loop over the chains and compute likelihoods
                v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

                # Combine the datasets
                return -np.sum(v), -np.sum(g, axis=0)

            # Optimize the model
            p0 = gp.get_parameter_vector()
            bounds = gp.get_parameter_bounds()
            soln = minimize(nll, p0, jac=True, bounds=bounds)
            gp.set_parameter_vector(soln.x)

            # Compute the maximum likelihood tau
            a, c = kernel.coefficients[:2]
            tau = thin * 2 * np.sum(a / c) / np.sum(a)
            return tau


        # Calculate the estimate for a set of different chain lengths
        ml = np.empty(len(N))
        ml[:] = np.nan
        for j, n in enumerate(N[1:-1]):
            i = j + 1
            thin = max(1, int(0.05 * new[i]))
            ml[i] = autocorr_ml(chain[:, :n], thin=thin)

if flag_plot == 1 or flag_plot == 3:
    # Plot the comparisons
    plt.loglog(N, gw2010, "o-", label="G&W 2010")
    plt.loglog(N, new, "o-", label="new")
    if flag_ml:
        plt.loglog(N, ml, "o-", label="ML")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14);

    fig1.savefig('chains/csc/post/'+prefix+'run_act1.pdf')


if flag_plot == 2 or flag_plot == 3:
    fig2 = plt.figure()
    samples = backend.get_chain()

    #EoS model
    eos_model_pos1 = prefix.find("M")
    eos_model_pos2 = prefix.find("_", eos_model_pos1)
    eos_model = int(prefix[eos_model_pos1+1:eos_model_pos2])

    # # of segments
    eos_Nsegment_pos1 = prefix.find("S")
    eos_Nsegment_pos2 = prefix.find("_", eos_Nsegment_pos1)
    eos_Nsegment = int(prefix[eos_Nsegment_pos1+1:eos_Nsegment_pos2])

    # dimension of the parameter space
    mparams  = args.mparams  #mass-measurement params
    lhparams = args.lhparams #low- and high-density params
    if eos_model == 0: #poly
        eos_dim = 2 * eos_Nsegment - 3
    elif eos_model == 1: #c_s^2
        eos_dim = 2 * eos_Nsegment - 4
    ndim = lhparams + eos_dim + mparams

    labels = [r"$\alpha_L$", r"$\eta_L$", r"$X$"]

    if eos_model == 0:
        #append gammas
        for itrope in range(eos_Nsegment-2):
            if itrope + 1 != phaseTransition:
                #parameters.append("gamma"+str(3+itrope))
                labels.append(r"$\gamma_{{{0}}}$".format((3+itrope)))

        #append transition depths
        for itrope in range(eos_Nsegment-1):
            #parameters.append("trans_delta"+str(1+itrope))
            labels.append(r"$\Delta n_{{{0}}}$".format(1+itrope))

    elif eos_model == 1:
        #append chemical potential depths (NB last one will be determined)
        for itrope in range(eos_Nsegment-2):
            #parameters.append("mu_delta"+str(1+itrope))
            labels.append(r"$\Delta\mu_{{{0}}}$".format((1+itrope)))

        #append speed of sound squared (NB last one will be determined)
        for itrope in range(eos_Nsegment-2):
            #parameters.append("speed"+str(1+itrope))
            labels.append(r"$c^2_{{{0}}}$".format(1+itrope))

    labels.append(r"$\mathcal{M}_{GW170817}$")
    labels.append(r"$q_{GW170817}$")

    labels.append(r"$M_{0432}$")
    labels.append(r"$M_{6620}$")

    labels.append(r"$M_{1702}$")
    labels.append(r"$M_{6304}$")
    labels.append(r"$M_{6397}$")
    labels.append(r"$M_{M28}$")
    labels.append(r"$M_{M30}$")
    labels.append(r"$M_{X7}$")
    labels.append(r"$M_{\omega Cen}$")
    labels.append(r"$M_{M13}$")
    labels.append(r"$M_{1724}$")
    labels.append(r"$M_{1810}$")
    labels.append(r"$M_{0437}$")

    fig2, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");

    fig2.savefig('chains/csc/post/'+prefix+'run_act2.pdf')

act = backend.get_autocorr_time(discard=0, thin=1, quiet=True)
np.savetxt('chains/csc/post/'+prefix+'run_autocorr.txt', act)
