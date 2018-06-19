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
