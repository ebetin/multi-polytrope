from scipy.optimize import fmin
from pQCD import pQCD
import units as cgs


# Checks that the polytropes are causal
#   Inputs:
#     gammas: polytropic exponents
#     transitions: transition densities [n0,n1,...nN,nQCD] (A.U.)
#     lowDensity = [p, e]
#       p: pressure (Ba)
#       e: energy density (g/cm^3)
def causalityPolytropes(gammas, transitions, lowDensity):
    # Low-density limit
    p = [lowDensity[0]] # Pressure
    e = [lowDensity[1] * cgs.c**2] # Energy density
    speedLow = gammas[0] * p[0] / (p[0] + e[0]) # Speed of sound

    if speedLow > 1.0:
        return False

    # Densities 
    rhoo = transitions

    for k in range(1, len(rhoo)):
        # Matching ("transition") pressures
        p.append(p[-1] * (rhoo[k] / rhoo[k-1])**gammas[k-1])


        # Matching ("transition") energy densities
        if gammas[k-1] == 1:
            e.append(p[k] * log(rhoo[k] / rhoo[k-1]) + e[k-1] * rhoo[k] / rhoo[k-1] )

        else:
            e.append(p[k] / (gammas[k-1] - 1.0) + (e[k-1] - p[k-1] / (gammas[k-1] - 1.0)) * (rhoo[k] / rhoo[k-1]))

        
        # Speed of sound before a transition
        coeff = p[-1] / (p[-1] + e[-1])
        speedHigh = gammas[k-1] * coeff
        
        # Speed of sound after a transition
        if k < len(rhoo) - 1:
            speedLow = gammas[k] * coeff

        if speedLow > 1.0 or speedHigh > 1.0:
            return False

    return True


# Function which tests is a polytropic EoS hydrodynamically stable
#   In other words, the polytropic exponent has to be non-negative.
def hydrodynamicalStabilityPolytropes(gammas):
    for gamma in gammas:
        if gamma < 0.0:
            return False

    return True


# Test for the causality of the pQCD EoS
# NB expects that the EoS is causal for all 1<=X<=4 and mu >= muLow
# NB this also means that the EoS is hydrodynamically stable
# Input:
#   EOS: qcd EOS
#   muMatching: chemical potential (GeV) of the pQCD EoS at the matching point
def causalityPerturbativeQCD(EOS, muMatching):
    press = pQCD(muMatching, EOS.X)

    if EOS.speed2(press) < 0.0:
        return False

    return True


# Tests that latent heats are positive quantities 
def positiveLatentHeat(pieces, transitions):
    for q in range( len(transitions) - 1 ):
        pressureLow = pieces[q].pressure(transitions[q+1])
        pressureHigh = pieces[q+1].pressure(transitions[q+1])

        # The pressure of the low density site should be greater or equal to the pressure of the high desity EoS. Nevertheless, due to possible rounding errors etc. some extra buffer is also included.
        if pressureLow < (1.0 - 1.49013e-08) * pressureHigh:
            return False

    return True


# NB The double monotropic EoS by Gandolfi et al. (2012, arXiv:1101.1921) is hydrodynamically stable because the mass of the baryon (neutron) is ~1000 MeV >> a, b ~ 1 MeV. However, this statement may not work with values of a, b, alpha and beta smaller than zero or they are larger than O(10 [MeV]). Besides, the derivative of the speed of sound is monotonically increasing due to the same effect.

# EOS = doubleMonotropic EoS
# rhoMax = highest used value of the baryon mass density with this EoS
def causalityDoubleMonotrope(EOS, rhoMax):
    pressMax = EOS.pressure(rhoMax)

    if EOS.speed2(pressMax) > 1.0:
        return False
    
    return True
