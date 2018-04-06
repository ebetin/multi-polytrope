from scipy.optimize import fsolve
from math import log
from math import pi
import numpy

import units as cgs

class Error(Exception):
    """Base class for other exceptions"""
    pass

class IncorrectNumberOfParametersError(Error):
    """Raised when the EoSLow or EoSHigh contain incorrect number of parameters"""
    pass

class NegativeQuantitativeError(Error):
    """Raised when the EoSLow or EoSHigh contain negative values"""
    pass

class NumberDensityDecreasingError(Error):
    """Raised when number density inputs are not in monotonically increasing order"""
    pass

class GammaError(Error):
    """Raised when the unknown polytropic indexes could not be determined"""
    pass

class IncorrectFlag(Error):
    """Raised when variable 'flag' has an incorrext value in function ab(x, flag)"""
    pass

class NonpositiveChemicalPotential(Error):
    """Raised when the chemical potential is nonpositive"""
    pass

class NonpositivePressure(Error):
    """Raised when the pressure is nonpositive"""
    pass

class NonpositiveNumberDensity(Error):
    """Raised when the baryon number density is nonpositive"""
    pass

class NonpositiveEnergyDensity(Error):
    """Raised when the energy density is nonpositive"""
    pass


class matchPolytopesWithLimits:

    # Input parameters:
    #   EoSLow: Low-density EoS parameters [p0, e0, n0]
    #     p0, e0, n0: Pressure (Ba), energy density (g/cm^3) and baryon density (rhoS) at the starting point
    #   EoSHigh: High-density EoS parameters [X, muFin]
    #     pN, eN: Pressure (Ba), energy density (g/cm^3) at the end point
    #   rhooEOS: Matching (transition) baryon densities (rhoS): [n0,n1,...,nN]
    #   gammaEOS: Known polytropic indexes [gamma_2,gamma_3,---]
    #     NB The two first polytropic indexes are assumed to be unknown
    def __init__(self, EoSLow, EoSHigh, rhooTransition, gammaKnown):
        self.pN = EoSHigh[0]
        self.eN = EoSHigh[1]
        self.p0 = EoSLow[0]
        self.e0 = EoSLow[1]
        self.rhooT = rhooTransition
        self.gamma = gammaKnown
        self.lenEoSLow = len(EoSLow)
        self.lenEoSHigh = len(EoSHigh)
        self.lenGamma = len(gammaKnown)
        self.lenRhooT = len(rhooTransition)

    # Gamma solver's test function: Checks pressure and energy density conditions for interpolated polytropes
    #   gammaUnknown: Two (still) unknown gammas,  [gamma_0,gamma_1]
    def solveGamma(self, gammaUnknown):
        # Normalized pressure and energy density at the starting density
        p = [self.p0 / (self.e0 * cgs.c**2)]
        e = [1.0]

        # Number densities [n0,n1,...nN,nQCD] (1/cm^3)
        rhoo = self.rhooT[:]

        # Number of mathing (transition) points
        n=len(rhoo)

        # Polytropic exponents
        g = self.gamma[:]
        g.insert(0, gammaUnknown[0])
        g.insert(1, gammaUnknown[1])

        for k in range(1, n):
            # Matching ("transition") pressures
            p.append(p[-1] * (rhoo[k] / rhoo[k-1])**g[k-1])

            # Matching ("transition") energy densities
            if g[k-1] == 1:
                e.append(p[k] * log(rhoo[k] / rhoo[k-1]) + e[k-1] * rhoo[k] / rhoo[k-1] )
            else:
                e.append(p[k] / (g[k-1] - 1.0) + (e[k-1] - p[k-1] / (g[k-1] - 1.0)) * (rhoo[k] / rhoo[k-1]))
    
        # Pressure condition at the end point
        out = [p[-1] - self.pN / (self.e0 * cgs.c**2)] 

        # Energy density condition at the end point
        out.append(e[-1] - self.eN / (self.e0 * cgs.c**2))

        return out

    # Returns all polytropic indexes (list) for given data set in case of interpolation between
    # nuclear physics EoS and perturbative QCD calculations
    #   GammaGuess: starting point for unknown indexes [Gamma_0, Gamma_1], default [4.0, 1.4]
    def GammaValues(self, GammaGuess=[4.0,1.4]):
        try:
            # Incorrect number of parameters
            if self.lenEoSLow != 2 or self.lenEoSHigh != 2 or self.lenGamma != (self.lenRhooT - 3):
                raise IncorrectNumberOfParametersError

            # Nonpositive thermodynamical quantitatives
            if self.p0 <= 0.0 or self.e0 <= 0.0 or self.pN <= 0.0 or self.eN <= 0.0 or self.rhooT[0] <= 0.0:
                raise NegativeQuantitativeError

            for k in range(1,len(self.rhooT)):
                if self.rhooT[k-1] >= self.rhooT[k]:
                    return NumberDensityDecreasingError
        
            # Calculate unknown polytropic indexes
            [Gammas,infoG,flagG,mesgG]=fsolve(self.solveGamma,GammaGuess,full_output=1,xtol=1.0e-10)

            GammaAll = Gammas.tolist() + self.gamma

            # Could not determine the polytropic indexes
            if flagG != 1:
                #print "?", self.solveGamma(GammaAll), infoG, flagG
                raise GammaError

            test = all(y < 1.0e-7 for y in self.solveGamma(GammaGuess)) 
            #print self.solveGamma(GammaAll) XXX
            if Gammas[0] == GammaGuess[0] and Gammas[1] == GammaGuess[1] and not test:

                raise GammaError

            return GammaAll

        except IncorrectNumberOfParametersError:
            print("Incorrect number of EoS parameters!")
            print()
        except NegativeQuantitativeError:
            print("Some EoS parameters are nonpositive!")
            print()
        except NumberDensityDecreasingError:
            print("Number density is not monotonically increasing!")
            print()
        except GammaError:
            print("Cannot solve polytropic indexes!")
            print()


# Perturbative QCD results, see Fraga et al. (2014) arXiv:1311.5154
# NB One should use values mu > 2.0 Gev and 1 <= X <= 4!
#   mu: Baryon chemical potential
#   X: Perturbatice QCD parameter related to renormalization scale

# Fitting functions
#   x: Perturbatice QCD parameter related to renormalization scale
#   flag: 
#     1: Function a(x), see Fraga et al. (2014) arXiv:1311.5154
#     2: Function b(x)
#     default: 0
def ab(x, flag):
    try:
        if flag == 1: # Function a(x)
            d = 0.5034
            v = 0.3553
        elif flag == 2: # Function b(x)
            d = 1.452
            v = 0.9101
        else:
            d = 0.0
            v = 0.0

        if flag != 1 and flag != 2:
            raise IncorrectFlag

        return d * pow(x,-v)

    except IncorrectFlag:
        print("In function ab(x, flag), the parameter 'flag' has an incorrect entry!")


# Stefan-Boltzmann limit (GeV^4) as a function of the baryon chemical potential (GeV)
def pSB(mu):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        return 0.75 / pi**2 * (mu / 3.0)**4

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")


# Pressure (difference) (Ba) as a function of the baryon chemical potential (GeV)
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def pQCD(mu, x, p=0):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        c = 0.9008
        pressure = pSB(mu) * (c - ab(x, 1) / (mu - ab(x, 2)))

        if pressure <= 0.0 and p == 0:
            raise NonpositivePressure

        return pressure * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm - p

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositivePressure:
        print("Pressure is nonpositive!")


# Number density (difference) (1/cm^3) as a function of the baryon chemical potential (GeV) 
# NB Number density = dP/d{mu}
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def nQCD(mu, x, rhoo=0):
    try:   
        if mu <= 0.0:
            raise NonpositiveChemicalPotential
 
        c = 0.9008

        density1 = (mu / 3.0)**3 / pi**2 * (c - ab(x, 1) / (mu - ab(x, 2))) 
        density2 = pSB(mu) * ab(x, 1) / (mu - ab(x, 2))**2
        density = density1 + density2

        if density <= 0.0 and rhoo==0:
            raise NonpositiveNumberDensity

        return density * cgs.GeV3_to_fm3 * 1.0e39 - rhoo

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")


# "Energy" density (difference) (NB g/cm^3, not erg/cm^3) as a function of the baryon chemical potential (GeV)
# NB energy density = mu * n - p for baryonic matter
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def eQCD(mu, x, e=0):
    try:   
        if mu <= 0.0:
            raise NonpositiveChemicalPotential
    
        energy = (mu * 1.0e9 * cgs.eV) * nQCD(mu, x) - pQCD(mu, x)

        if energy <= 0.0 and e==0:
            raise NonpositiveEnergyDensity

        return energy / cgs.c**2 - e

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositiveEnergyDensity:
        print("Energy density is nonpositive!")


# Pressure (Ba) as a function of function of the "energy" density (g/cm^3)
#   e: Energy density
#   x: Perturbatice QCD parameter related to renormalization scale
def pQCD_energy(e, x):
    try:
        if e <= 0.0:
            raise NonpositiveEnergyDensity

        mu = fsolve(eQCD, 2.6, args = (x, e))[0]

        return pQCD(mu, x)

    except NonpositiveEnergyDensity:
        print("Energy density is nonpositive!")


# Pressure (Ba) as a function of function of the baryon number density (1/cm^3)
#   rhoo: Baryon number density
#   x: Perturbatice QCD parameter related to renormalization scale
def pQCD_density(rhoo, x):
    try:
        if rhoo <= 0.0:
            raise NonpositiveNumberDensity

        mu = fsolve(nQCD, 2.6, args = (x, rhoo))[0]

        return pQCD(mu, x)

    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")

# Number density (1/cm^3) as a function of the pressure (Ba)
#   p: Pressure
#   x: Perturbatice QCD parameter related to renormalization scale
def nQCD_pressure(p, x):
    try:
        if p <= 0.0:
            raise NonpositivePressure

        mu = fsolve(pQCD, 2.6, args = (x, p))[0]

        return nQCD(mu, x)

    except NonpositivePressure:
        print("Pressure is nonpositive!")


# "Energy" density (g/cm^3) as a function of the baryon number density (1/cm^3)
#   rhoo: Baryon number density
#   x: Perturbatice QCD parameter related to renormalization scale
def eQCD_density(rhoo, x):
    try:
        if rhoo <= 0.0:
            raise NonpositiveNumberDensity

        mu = fsolve(nQCD, 2.6, args = (x, rhoo))[0]

        return eQCD(mu, x)

    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")


# "Energy" density (g/cm^3) as a function of the pressure (Ba)
#   p: Pressure
#   x: Perturbatice QCD parameter related to renormalization scale
def eQCD_pressure(p, x):
    try:
        if p <= 0.0:
            raise NonpositivePressure

        mu = fsolve(pQCD, 2.6, args = (x, p))[0]

        return eQCD(mu, x)

    except NonpositivePressure:
        print("Pressure is nonpositive!")


# Quark matter EoS from perturbative QCD
class qcd:

    def __init__(self, x):
        # x: pQCD scale parameter, see arXiv:1311.5154 for details
        self.X = x
        self.a = ab(x, 1)
        self.b = ab(x, 2)
        self.c1 = 0.9008

    # Pressure (Ba) as a function of mass density (g/cm^3)
    def pressure(self, rhoo):
        try:
            if rhoo <= 0.0:
                raise NonpositiveNumberDensity

            mu = fsolve(nQCD, 2.6, args = (self.X, rhoo / cgs.mB))[0]

            return pQCD(mu, self.X)

        except NonpositiveNumberDensity:
            print("Baryon number density is nonpositive!")

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho / cgs.mB)
            press.append( pr )
        return press

    # Energy density (g/cm^3) as a function of pressure (Ba)
    def edens_inv(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD, 2.6, args = (self.X, press))[0]

            return eQCD(mu, self.X)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    # Baryon mass density (g/cm^3) as a function of pressure (Ba)
    def rho(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD, 2.6, args = (self.X, press))[0]

            return nQCD(mu, self.X) * cgs.mB

        except NonpositivePressure:
            print("Pressure is nonpositive!")

