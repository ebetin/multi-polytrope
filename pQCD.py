import numpy
from scipy.optimize import fsolve

from math import log, pi

import units as cgs

#######################################
#constants
muNNNLOstop = 2.739512831322249
pi2inv = 1.0 / pi**2
inv3 = 0.3333333333333333333333333 #1.0 / 3.0
c2inv = 1.0 / cgs.c**2
#######################################

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
        # Scaling factor
        denominator = c2inv / self.e0

        # Scaled pressure and energy density at the starting density
        p = [self.p0 * denominator]
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
            ratio = rhoo[k] / rhoo[k-1]

            if p[-1] == 0.0 or g[k-1] < -10.0 or g[k-1] > 335.0:
                # Breaking out if the situation is unrealistic
                p.append(0.0)
                return  [-10.0, -10.0]

            else:
                # Matching ("transition") pressures
                p.append( ratio**g[k-1] * p[-1] )

            # Matching ("transition") energy densities
            if g[k-1] == 1:
                e.append(p[k] * log(ratio) + e[k-1] * ratio)
            else:
                division = 1.0 / (g[k-1] - 1.0)
                e.append(p[k] * division + (e[k-1] - p[k-1] * division) * ratio)

    
        # Pressure condition at the end point
        out = [p[-1] - self.pN * denominator] 

        # Energy density condition at the end point
        out.append(e[-1] - self.eN * denominator)

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
                    raise NumberDensityDecreasingError

            # Calculate unknown polytropic indexes
            [Gammas,infoG,flagG,mesgG]=fsolve(self.solveGamma,GammaGuess,full_output=1,xtol=1.0e-9)

            # Could not determine the polytropic indexes (flagG != 1)
            if flagG != 1:
                # Recalculate with a bit different starting point
                [Gammas,infoG,flagG,mesgG]=fsolve(self.solveGamma,[10.0,10.0],full_output=1,xtol=1.0e-9)

                # No realistic solutions found
                if flagG != 1:
                    Gammas = numpy.array([-1.0, -1.0])

            self.gammasSolved = Gammas.tolist()

            GammaAll = Gammas.tolist() + self.gamma

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


# Stefan-Boltzmann limit (GeV^4) as a function of the baryon chemical potential (GeV)
def pSB(mu):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        return 0.75 * pi2inv * (mu * inv3)**4

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")

#TODO maybe one need to fix this?
# Stefan-Boltzmann limit (Ba) as a function of the baryon chemical potential (GeV)
def pSB_csg(mu):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        return pSB(mu) * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

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


# Pressure (difference) (Ba) as a function of the baryon chemical potential (GeV)
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def pQCD_pocket(mu, x, p=0):
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
def nQCD_pocket(mu, x, rhoo=0):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        c = 0.9008

        density1 = (mu * inv3)**3 * pi2inv * (c - ab(x, 1) / (mu - ab(x, 2)))
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
def eQCD_pocket(mu, x, e=0):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        energy = (mu * 1.0e9 * cgs.eV) * nQCD_pocket(mu, x) - pQCD_pocket(mu, x)

        if energy < 0.0 and e==0:
            raise NonpositiveEnergyDensity

        return energy * c2inv - e

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositiveEnergyDensity:
        print("Energy density is nonpositive!")


# Pressure (Ba) as a function of function of the "energy" density (g/cm^3)
#   e: Energy density
#   x: Perturbatice QCD parameter related to renormalization scale
def pQCD_energy_pocket(e, x):
    try:
        if e <= 0.0:
            raise NonpositiveEnergyDensity

        mu = fsolve(eQCD_pocket, 2.6, args = (x, e))[0]

        return pQCD_pocket(mu, x)

    except NonpositiveEnergyDensity:
        print("Energy density is nonpositive!")


# Pressure (Ba) as a function of function of the baryon number density (1/cm^3)
#   rhoo: Baryon number density
#   x: Perturbatice QCD parameter related to renormalization scale
def pQCD_density_pocket(rhoo, x):
    try:
        if rhoo <= 0.0:
            raise NonpositiveNumberDensity

        mu = fsolve(nQCD_pocket, 2.6, args = (x, rhoo))[0]

        return pQCD_pocket(mu, x)

    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")

# Number density (1/cm^3) as a function of the pressure (Ba)
#   p: Pressure
#   x: Perturbatice QCD parameter related to renormalization scale
def nQCD_pressure_pocket(p, x):
    try:
        if p <= 0.0:
            raise NonpositivePressure

        mu = fsolve(pQCD_pocket, 2.6, args = (x, p))[0]

        return nQCD_pocket(mu, x)

    except NonpositivePressure:
        print("Pressure is nonpositive!")


# "Energy" density (g/cm^3) as a function of the baryon number density (1/cm^3)
#   rhoo: Baryon number density
#   x: Perturbatice QCD parameter related to renormalization scale
def eQCD_density_pocket(rhoo, x):
    try:
        if rhoo <= 0.0:
            raise NonpositiveNumberDensity

        mu = fsolve(nQCD_pocket, 2.6, args = (x, rhoo))[0]

        return eQCD_pocket(mu, x)

    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")


# "Energy" density (g/cm^3) as a function of the pressure (Ba)
#   p: Pressure
#   x: Perturbatice QCD parameter related to renormalization scale
def eQCD_pressure_pocket(p, x):
    try:
        if p <= 0.0:
            raise NonpositivePressure

        mu = fsolve(pQCD_pocket, 2.6, args = (x, p))[0]

        return eQCD_pocket(mu, x)

    except NonpositivePressure:
        print("Pressure is nonpositive!")


# Old quark matter EoS from perturbative QCD
class qcd_pocket:

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

            mu = fsolve(nQCD_pocket, 2.6, args = (self.X, rhoo / cgs.mB))[0]

            return pQCD_pocket(mu, self.X)

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

            mu = fsolve(pQCD_pocket, 2.6, args = (self.X, press))[0]

            return eQCD_pocket(mu, self.X)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    # Energy density (g/cm^3) as a function of mass density (g/cm^3)
    def edens(self, rhoo):
        try:
            if rhoo <= 0.0:
                raise NonpositiveNumberDensity

            mu = fsolve(nQCD_pocket, 2.6, args = (self.X, rhoo / cgs.mB))[0]

            return eQCD_pocket(mu, self.X)

        except NonpositiveNumberDensity:
            print("Baryon number density is nonpositive!")

    # Baryon mass density (g/cm^3) as a function of pressure (Ba)
    def rho(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD_pocket, 2.6, args = (self.X, press))[0]

            return nQCD_pocket(mu, self.X) * cgs.mB

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    # Square of the speed of sound (unitless)
    def speed2(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD_pocket, 2.6, args = (self.X, press))[0]

            numerator = ( self.a * (4.0 * self.b - 3.0 * mu) + 4.0 * self.c1 * (self.b - mu)**2 ) * (self.b - mu)
            denominator = 2.0 * ( 6.0 * self.c1 * (self.b - mu)**3 + self.a * (6.0 * self.b**2 - 8.0 * self.b * mu + 3.0 * mu**2) )

            return numerator / denominator

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    def speed2_rho(self, rho):
        press = self.pressure(rho)
        return self.speed2(press)

    def gammaFunction(self, rho, flag = 1):
        press = self.pressure(rho)
        edens = self.edens_inv(press) * cgs.c**2.0
        speed2 = self.speed2(press)

        if flag == 1: # d(ln p)/d(ln n)
            return ( edens + press ) * speed2 / press
        else: # d(ln p)/d(ln eps)
            return edens * speed2 / press

    def pressure_edens(self, edens):
        try:
            if edens <= 0.0:
                raise NonpositivePressure

            mu = fsolve(eQCD_pocket, 2.6, args = (self.X, edens))[0]

            return pQCD_pocket(mu, self.X)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#NNNLO results

def alpha_s(mu, x):
    tmp = 2.0 * log( 0.8818342151675485 * mu * x )
    tmp9_inv = 1.0 / ( 9.0 * tmp )

    return 4.0 * pi * ( 1.0 - 7.111111111111111 * tmp9_inv * log(tmp) ) * tmp9_inv

def p_norm(a_s, x):
    p_n = 1.0 - 0.637 * a_s
    p_n -= ( 0.875 + 0.912 * log(x) + 0.304 * log(a_s) ) * a_s**2
    p_n += 0.4848163545236859 * a_s**3

    return p_n

def p_norm_derva(a_s, x):
    p_n_derva = - 0.637 - ( 2.054 + 1.824 * log(x) + 0.608 * log(a_s) ) * a_s
    p_n_derva += 1.4544490635710576 * a_s**2

    return p_n_derva

def alpha_s_derva(mu, x):
    tmp = log(x * mu * inv3)
    tmp2 = 1.9457221667250986 + 2.0 * tmp

    asd = 4.412881861832576 * log(tmp2) - 5.585053606381853 * tmp - 7.639922233058851
    asd /= mu * tmp2**3

    return asd

# Pressure (difference) (Ba) as a function of the baryon chemical potential (GeV)
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def pQCD(mu, x, p=0):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        a_s = alpha_s(mu, x)
        p_n = p_norm(a_s, x)

        if p_n <= 0.0 and p == 0:
            raise NonpositivePressure

        return pSB(mu) * p_n * cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm - p

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositivePressure:
        print("Pressure is nonpositive!")


# Number density (GeV^3) as a function of the baryon chemical potential (GeV)
# NB Number density = dP/d{mu}
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def nQCD_nu(mu, x):
    try:
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        a_s = alpha_s(mu, x)
        a_s_derva = alpha_s_derva(mu, x)

        p_n = p_norm(a_s, x)
        p_n_derva = p_norm_derva(a_s, x)

        pSB_derva = (mu * inv3)**3 * pi2inv
        p_SB = 0.75 * mu * inv3 * pSB_derva#0.75 * pi2inv * (mu * inv3)**4

        density = pSB_derva * p_n
        density += p_SB * p_n_derva * a_s_derva

        if density <= 0.0:
            raise NonpositiveNumberDensity

        return density

    except NonpositiveChemicalPotential:
        print("Chemical potential is nonpositive!")
    except NonpositiveNumberDensity:
        print("Baryon number density is nonpositive!")

def nQCD_nu_wo_errors(mu, x):
    a_s = alpha_s(mu, x)
    a_s_derva = alpha_s_derva(mu, x)

    p_n = p_norm(a_s, x)
    p_n_derva = p_norm_derva(a_s, x)

    pSB_derva = (mu * inv3)**3 * pi2inv
    p_SB = 0.75 * mu * inv3 * pSB_derva#0.75 * pi2inv * (mu * inv3)**4

    density = pSB_derva * p_n
    density += p_SB * p_n_derva * a_s_derva

    return density


# Number density (difference) (1/cm^3) as a function of the baryon chemical potential (GeV) 
# NB Number density = dP/d{mu}
#   params: High-density parameters [X, muFin]
#     mu: Baryon chemical potential (GeV)
#     x: Perturbatice QCD parameter related to renormalization scale
def nQCD(mu, x, rhoo=0):
    try:   
        if mu <= 0.0:
            raise NonpositiveChemicalPotential

        density = nQCD_nu(mu, x)

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

        if energy < 0.0 and e==0:
            raise NonpositiveEnergyDensity

        return energy * c2inv - e

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

        mu = fsolve(eQCD, muNNNLOstop, args = (x, e))[0]

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

        mu = fsolve(nQCD, muNNNLOstop, args = (x, rhoo))[0]

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

        mu = fsolve(pQCD, muNNNLOstop, args = (x, p))[0]

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

        mu = fsolve(nQCD, muNNNLOstop, args = (x, rhoo))[0]

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

        mu = fsolve(pQCD, muNNNLOstop, args = (x, p))[0]

        return eQCD(mu, x)

    except NonpositivePressure:
        print("Pressure is nonpositive!")


# NNNLO-quark-matter EoS from perturbative QCD
class qcd:

    def __init__(self, x):
        # x: pQCD scale parameter, see arXiv:1311.5154 for details
        self.X = x

    def press_edens_c2(self, rhoo):
        try:
            if rhoo <= 0.0:
                raise NonpositiveNumberDensity

            mu = fsolve(nQCD, muNNNLOstop, args = (self.X, rhoo / cgs.mB))[0]

            return pQCD(mu, self.X), eQCD(mu, self.X), self.speed2_mu(mu)

        except NonpositiveNumberDensity:
            print("Baryon number density is nonpositive!")

    # Pressure (Ba) as a function of mass density (g/cm^3)
    def pressure(self, rhoo):
        try:
            if rhoo <= 0.0:
                raise NonpositiveNumberDensity

            mu = fsolve(nQCD, muNNNLOstop, args = (self.X, rhoo / cgs.mB))[0]

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

            mu = fsolve(pQCD, muNNNLOstop, args = (self.X, press))[0]

            return eQCD(mu, self.X)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    # Energy density (g/cm^3) as a function of mass density (g/cm^3)
    def edens(self, rhoo):
        try:
            if rhoo <= 0.0:
                raise NonpositiveNumberDensity

            mu = fsolve(nQCD, muNNNLOstop, args = (self.X, rhoo / cgs.mB))[0]

            return eQCD(mu, self.X)

        except NonpositiveNumberDensity:
            print("Baryon number density is nonpositive!")

    # Baryon mass density (g/cm^3) as a function of pressure (Ba)
    def rho(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD, muNNNLOstop, args = (self.X, press))[0]

            return nQCD(mu, self.X) * cgs.mB

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    # Square of the speed of sound (unitless)
    def speed2(self, press):
        try:
            if press <= 0.0:
                raise NonpositivePressure

            mu = fsolve(pQCD, muNNNLOstop, args = (self.X, press))[0]

            return self.speed2_mu(mu)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

    def speed2_mu(self, mu):
        a_s = alpha_s(mu, self.X)
        p_n = p_norm(a_s, self.X)

        xlog = log(self.X)
        alpha_s_log = log(a_s)

        muinv3 = mu * inv3
        pSB_derva2 = muinv3**2 * pi2inv
        pSB_derva = pSB_derva2 * muinv3
        p_SB = 0.75 * pSB_derva2 * muinv3**2

        p_n_derva = p_norm_derva(a_s, self.X)

        p_norm_derva2 = - 2.6620000000000004 + 2.9088981271421153 * a_s
        p_norm_derva2 -= 0.608 * log(a_s) + 1.824 * log(self.X)

        tmp3log = log(self.X * muinv3)
        tmp2 = 1.9457221667250986 + 2.0 * tmp3log
        tmp2log = log(tmp2)
        tmp2_mu_inv = 1.0 / ( mu * tmp2**3 )
        alpha_s_derva1 = ( 4.412881861832576 * tmp2log - 5.585053606381853 * tmp3log - 7.639922233058851 ) * tmp2_mu_inv

        alpha_s_derva2 = ( 58.66350055865166 + ( 48.48702149593025 + 11.170107212763707 * tmp3log ) * tmp3log - ( 35.06353322870223 + 8.825763723665151 * tmp3log ) * tmp2log ) * (tmp2_mu_inv * tmp2)**2

        density = pSB_derva * p_n
        density += p_SB * p_n_derva * alpha_s_derva1

        density_derva = pSB_derva2 * p_n
        density_derva += 2.0 * pSB_derva * p_n_derva * alpha_s_derva1
        density_derva += p_SB * p_norm_derva2 * alpha_s_derva1**2
        density_derva += p_SB * p_n_derva * alpha_s_derva2

        return density / ( mu * density_derva )

    def speed2_rho(self, rho):
        press = self.pressure(rho)
        return self.speed2(press)

    def gammaFunction(self, rho, flag = 1):
        press = self.pressure(rho)
        edens = self.edens_inv(press) * cgs.c**2.0
        speed2 = self.speed2(press)

        if flag == 1: # d(ln p)/d(ln n)
            return ( edens + press ) * speed2 / press
        else: # d(ln p)/d(ln eps)
            return edens * speed2 / press

    def pressure_edens(self, edens):
        try:
            if edens <= 0.0:
                raise NonpositivePressure

            mu = fsolve(eQCD, muNNNLOstop, args = (self.X, edens))[0]

            return pQCD(mu, self.X)

        except NonpositivePressure:
            print("Pressure is nonpositive!")

if __name__ == "__main__":
    # Ba -> Mev/fm^3
    confacinv = 1000.0 / cgs.GeVfm_per_dynecm

    # Reference chemical potential (GeV)
    muQCD = 2.739512831322249

    # List of pQCD scale parameters
    x_list = [.6744593, .67524, .75, 1, 2, 4, 10, 100]

    for item in x_list:
        print("X:", round(item, 3), "\tn:", round(nQCD(muQCD, item) * cgs.mB / cgs.rhoS, 5), "\tp:", round(pQCD(muQCD, item) * confacinv, 2), "\tc2:", round(qcd(item).speed2_mu(muQCD), 5))

