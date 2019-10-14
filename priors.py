import numpy as np
from scipy.stats import norm,beta,truncnorm

# Originating from https://github.com/nespinoza/spotnest/blob/master/spotnest.py


def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def transform_truncated_normal(x,mu,sigma,a=0.,b=1.):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)


# for MCMC sampler
def check_uniform(x,a,b):
    if a < x < b:
        return 0.0
    else:
        return -np.inf

def prior_cEFT(x, y):
    xmin = 1.17937
    xmax = 1.59097

    if xmin < x < xmax:
        a1 = 1.31859
        b1 = -0.917676

        a2 = 0.912723
        b2 = -0.336348

        a3 = 0.84288
        b3 = -0.358904

        a4 = 1.29349
        b4 = -0.948088

        if (x < 1.43656 and y <= a1 * x + b1) or (x >= 1.43656 and y <= a2 * x + b2):
            if (x < 1.30753 and y >= a3 * x + b3) or (x >= 1.30753 and y >= a4 * x + b4):
                return 0.0

    return -np.inf
