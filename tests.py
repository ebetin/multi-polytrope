from scipy.optimize import fmin
import units as cgs

# NB The double monotropic EoS by Gandolfi et al. (2012, arXiv:XXX) is causal and hydrodynamically stable because the mass of the baryon (neutron) is ~1000 MeV >> a, b ~ 1 MeV. However, this statement may not work with values of a, b, alpha and beta smaller than zero or they are larger than O(10 MeV)!

# Function which checks is a polytrope causal
# NB Used after the polytrope is formed! #XXX tarkista onko toimiva nain!
# Inputs:
#   tropes:       collection of monotropes
#   transtitions: transtion densities (g/cm^3)
#   highDensity:  [press, edens]
#     press: pressure (Ba) at the high end of the polytrope
#     edens: corresponding energy density (g/cm^3)
def causalityPolytropes(tropes, transitions, highDensity):
    for j in range(len(tropes)):
        trope = tropes[j]
        transitionLower = transitions[j]   
        energyDensityLower = trope.edens(transitionLower)
        pressureLower = trope.pressure(transitionLower)
            
        if j == len(tropes) - 1:
            energyDensityUpper = highDensity[1]
            pressureUpper = highDensity[0]
        else:
            transitionUpper = transitions[j+1]
            energyDensityUpper = trope.edens(transitionUpper)
            pressureUpper = trope.pressure(transitionUpper)

        #Squared adiabatic speed of sound
        speedUpper = trope.G * pressureUpper / (pressureUpper + energyDensityUpper)
        speedLower = trope.G * pressureLower / (pressureLower + energyDensityLower)

        #Is the monotope acausal?
        if speedUpper > 1.0 and speedUpper < 0.0 and speedLower > 1.0 and speedLower < 0.0:
            return False

    return True


# Function which tests is a polytropic EoS hydrodynamically stable
#def hydrodynamicalStabilityPolytropes(tropes):
#    for trope in tropes:
#        if trope.G < 0.0:
#            return False
#
#    return True
def hydrodynamicalStabilityPolytropes(gammas):
    for gamma in gammas:
        if gamma < 0.0:
            return False

    return True

# Test for the causality of the pQCD EoS
# NB expects that the EoS is causal for all 1<=X<=4 and mu >= muLow
# NB this also means that the EoS is hydrodynamically stable
# Input:
#   muMatching: chemical potential of the 
def causalityPerturbativeQCD(muMatching):
    if muMatching < 2.009005:
        return False

    return True

def positiveLatentHeat(pieces, transitions):
    for q in range( len(transitions) - 1 ):
        pressureLow = pieces[q].pressure(transitions[q+1])
        pressureHigh = pieces[q+1].pressure(transitions[q+1])

        # The pressure of the low density site should be greater or equal to the pressure of the high desity EoS. Nevertheless, due to possible rounding errors etc. some extra buffer is also included.
        if pressureLow < (1.0 - 1.0e-11) * pressureHigh:
            return False

    return True
