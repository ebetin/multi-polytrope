import units as cgs
import numpy as np
#from mpmath import hyp2f1, re
from scipy.special import hyp2f1

from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# This file contains speed of sound square (c^2) interpolation related formulas etc.
# The used interpolation method is from arXiv:1903.09121.


# This class determines the values of physical quantities, such as pressure and densities.
class c2AGKNV:
    #speed of sound squared (cm^2/s^2)
    cgsunits = cgs.c**2
    cgsunits_inv = 1.0 / cgsunits

    #inverse baryon mass
    mB_inv = 1.0 / cgs.mB

    # giga electron volts in ergs
    GeV = 1.0e9 * cgs.eV

    # Inputs:
    #     muList: list of matching chemical potentials (GeV)
    #     c2List: list of matching speed of sound squares (unitless)
    #     lowDensity: physical quantities at the starting point of the interpolation
    def __init__(self, muList, c2List, lowDensity, approx = False):
        self.muList = muList
        self.c2List = c2List

        # Starting point (lowest density)
        self.p0 = lowDensity[0]   # Pressure (Ba)
        self.e0 = lowDensity[1]   # Energy density (g/cm^3)
        self.rho0 = lowDensity[2] # Baryon mass density (g/cm^3)
        self.c20 = lowDensity[3]  # speed of sound square (unitless)

        # Chemical potential at the starting point
        self.mu0 = cgs.mB * (self.e0 * self.cgsunits + self.p0) / ( self.rho0 * self.GeV )

        # Inserting the starting point into parameter lists
        self.muList.insert(0, self.mu0)
        self.c2List.insert(0, self.c20)

        self.approx = approx

        if approx:
            N = 500 # If one wants to be very save, use N = 10,000
            listRho = np.linspace(self.rho0, 15.0 * cgs.rhoS, N)
            listP = np.zeros(N)
            listP[0] = self.p0

            for i in range(1, N):
                listP[i] = self.pressure(listRho[i])

            self.listRhoLong = listRho
            self.listPLong = listP


    # Power used in the next function
    # Input:
    #     index: position of the interpolation segment
    def ai(self, index):
        c2iMu1 = self.muList[index - 1] * self.c2List[index]
        c21Mui = self.muList[index] * self.c2List[index - 1]

        return (self.muList[index] - self.muList[index - 1]) / (c2iMu1 - c21Mui)


    # Term in the mass density product
    # Inputs:
    #     mu: Chemical potential (GeV)
    #     index: position of the interpolation segment
    #     aiStatus: Is the power used? (default: True)
    def rhoBar(self, mu, index, aiStatus = True):
        c2iMu1 = self.muList[index - 1] * self.c2List[index]
        c21Mu1 = self.muList[index - 1] * self.c2List[index - 1]
        c21Mui = self.muList[index] * self.c2List[index - 1]

        # Determinating the value of the base
        # Last term that is relevant
        if mu < self.muList[index] and mu > self.muList[index - 1]:
            bar = c21Mu1 * (self.muList[index] - mu) + c2iMu1 * (mu - self.muList[index - 1])
            bar = 1.0 * bar / ( mu * (c21Mui - c21Mu1) )
        # Other relevant terms
        elif mu >= self.muList[index]:
            bar = 1.0 * c2iMu1 / c21Mui
        # Non-relevant terms
        else:
            return 1.0

        # If bar is negative, we can just return its value
        # This is not the right kinf of situation but it have to be used during determining
        # the unknown interpolation parameters!
        if bar < 0:
            return bar

        # Is the power used?
        if aiStatus:
            ai = self.ai(index)

            # If the power is NaN, we just return the bar
            if np.isnan(ai):
                return 1.0

            bar = bar**ai

        return bar


    # Listing of mass densities in the matching points (cf. c2List or muList)
    def rhoListing(self):
        N = len(self.c2List)
        listRho = N * [None]
        listRho[0] = self.rho0

        for i in range(1, N):
            listRho[i] = listRho[i-1] * self.rhoBar(self.muList[i], i)

        return listRho


    # Determinates the position of the segment from given mass density (rho; g/cm^3)
    def indexRho(self, rho):
        listInUse = self.rhoListing()
        
        try:
            index = [listInUse.index(x) for x in listInUse if x > rho][0]  

            return index
        except:
            return len(listInUse)-1


    # Calculates the chemical potential (GeV) from given mass density (rho; g/cm^3)
    def chemicalPotential(self, rho):
        index = self.indexRho(rho)
        ai = self.ai(index)
        barRho = 1.0 * rho / self.rhoListing()[index - 1] # cf. rhoBar

        numerator = self.c2List[index - 1] * self.muList[index] 
        numerator = numerator - self.c2List[index] * self.muList[index - 1]
        denominator = (1.0 + (1.0 * self.muList[index] / self.muList[index - 1] - 1.0) * barRho**(1.0 / ai)) 
        denominator = denominator * self.c2List[index - 1] - self.c2List[index]

        return numerator / denominator


    # Subterm in the pressure sum
    # Inputs:
    #     mu: Chemical potential (GeV)
    #     index: position of the segment
    def pressurePartial(self, mu, index):
        aiNegative = -1.0 * self.ai(index)
        termZ = mu * ( self.c2List[index - 1] - self.c2List[index] ) 
        termZ = termZ / ( self.c2List[index - 1] * self.muList[index] - self.c2List[index] * self.muList[index - 1] )

        if type(termZ) is np.ndarray:
            f = hyp2f1( aiNegative, 1.0, 2.0 + aiNegative, termZ[0] / (termZ[0] - 1.0) )

            return np.array( [ float(re((1.0 * mu * f / (1.0 + aiNegative))[0])) ])
        else:
            f = hyp2f1( aiNegative, 1.0, 2.0 + aiNegative, termZ / (termZ - 1.0) )

            return (1.0 * mu * f / (1.0 + aiNegative)).real


    # Term in the pressure sum
    # Inputs:
    #     rho: Mass density (g/cm^3)
    #     index: position of the segment
    def pressureTerm(self, rho, index):
        mu = self.chemicalPotential(rho)
        rhoList = self.rhoListing()

        if mu < self.muList[index] and mu > self.muList[index - 1]:
            high = rho * self.pressurePartial(mu, index)
        elif mu >= self.muList[index]:
            high = rhoList[index] * self.pressurePartial(self.muList[index], index)
        else:
            return 0.0

        low = rhoList[index - 1] * self.pressurePartial(self.muList[index - 1], index)

        return self.GeV * float(high - low) * self.mB_inv


    # Term in the pressure sum
    # Inputs:
    #     mu: Chemical potential (GeV)
    #     index: position of the segment
    def pressureMuTerm(self, mu, index):
        if mu < self.muList[index] and mu > self.muList[index - 1]:
            high = self.rhoMu(mu) * self.pressurePartial(mu, index)
        elif mu >= self.muList[index]:
            high = self.rhoMu(self.muList[index]) * self.pressurePartial(self.muList[index], index)
        else:
            return 0.0

        low = self.rhoMu(self.muList[index - 1]) * self.pressurePartial(self.muList[index - 1], index)

        return self.GeV * float(high - low) * self.mB_inv


    # Pressure (Ba) as a function of chemical potential (GeV)
    def pressureMu(self, mu):
        N = len(self.c2List)
        pressureMuVector = N * [None]
        pressureMuVector[0] = self.p0

        for i in range(1, N):
            pressureMuVector[i] = self.pressureMuTerm(mu, i)

        return sum(pressureMuVector)


    # Baryon mass density (g/cm^3) as a function of chemical potential (GeV)
    def rhoMu(self, mu):
        N = len(self.c2List)
        rhoMuVector = N * [None]
        rhoMuVector[0] = self.rho0

        for i in range(1, N):
            rhoMuVector[i] = self.rhoBar(mu, i)

        return np.prod(rhoMuVector)

    # Energy density (g/cm^3) as a function of the mass density (g/cm^3)
    def edens_inv_rho(self, rho):
        pressure = self.pressure(rho) # pressure (Ba)
        mu = self.chemicalPotential(rho) * self.GeV # chem.pot. (ergs)

        return (rho * mu * self.mB_inv - pressure) * self.cgsunits_inv

    # Speed of sound squared (unitless) as a function of the mass density (g/cm^3)
    def speed2_rho(self, rho):
        mu = self.chemicalPotential(rho) # chem.pot. (GeV)
        index = self.indexRho(rho)

        numerator = self.c2List[index - 1] * ( self.muList[index] - mu ) 
        numerator = numerator + self.c2List[index] * ( mu -self.muList[index - 1] )
        denominator = self.muList[index] - self.muList[index - 1]

        return 1.0 * numerator / denominator

    ################################################
    ################################################

    # Pressure (Ba) as a function of the mass density (g/cm)
    def pressure(self, rho, p = 0.0):
        N = len(self.c2List)
        pressureVector = N * [None]
        pressureVector[0] = self.p0

        for i in range(1, N):
            pressureVector[i] = self.pressureTerm(rho, i)

        return sum(pressureVector) - p


    # Vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press


    # Energy density (g/cm^3) as a function of the pressure (Ba)
    def edens_inv(self, pressure, e = 0, approx = 2):
        if approx == 0:
            rho = self.rho(pressure, False) # mass density (g/cm^3)
        elif approx == 1:
            rho = self.rho(pressure, True) # mass density (g/cm^3)
        else:
            rho = self.rho(pressure, self.approx) # mass density (g/cm^3)
        mu = self.chemicalPotential(rho) * self.GeV # chem.pot. (ergs)

        return (rho * mu * self.mB_inv - pressure) * self.cgsunits_inv - e


    # Mass density (g/cm^3) as a function of the pressure (Ba)
    def rho(self, pressure, approx):
        if approx:
            if pressure > self.p0:
                try:
                    fun = interp1d(self.listPLong, self.listRhoLong, kind = 'linear')

                    return fun(pressure)
                except:
                    rho = fsolve(self.pressure, 35.0*cgs.rhoS, args = pressure)[0]

                    return rho
            else:
                return self.rho0
        else:
            rho = fsolve(self.pressure, 2.0*cgs.rhoS, args = pressure)[0]

            return rho


    # Speed of sound squared (unitless) as a function of the pressure (Ba)
    def speed2(self, pressure):
        rho = self.rho(pressure, self.approx) # mass density (g/cm^3)
        mu = self.chemicalPotential(rho) # chem.pot. (GeV)
        index = self.indexRho(rho)

        numerator = self.c2List[index - 1] * ( self.muList[index] - mu ) 
        numerator = numerator + self.c2List[index] * ( mu -self.muList[index - 1] )
        denominator = self.muList[index] - self.muList[index - 1]

        return 1.0 * numerator / denominator


    def gammaFunction(self, rho, flag = 1):
        press = self.pressure(rho)
        edens = self.edens_inv_rho(rho) * self.cgsunits
        speed2 = self.speed2_rho(rho)

        if flag == 1: # d(ln p)/d(ln n)
            return ( edens / press + 1.0 ) * speed2
        else: # d(ln p)/d(ln eps)
            return edens / press * speed2

    # Energy density (g/cm^3) as a function of the mass density (g/cm^3)
    def edens_rho(self, rho, e = 0):
        mu = self.chemicalPotential(rho) * self.GeV # chem.pot. (ergs)
        pressure = self.pressure(rho)

        return (rho * mu * self.mB_inv - pressure) * self.cgsunits_inv - e 


    def pressure_edens(self, edens):
        edensGeV =  edens * self.cgsunits / cgs.GeVfm_per_dynecm

        if edensGeV < 0.7:
            rhoEstimate = edensGeV + 0.05
        else:
            rhoEstimate = 0.4 * edensGeV + 0.5

        rho = fsolve(self.edens_rho, rhoEstimate * cgs.mB * 1.0e39, args = edens)[0]

        return self.pressure(rho)



# This class tries to find the unknown interpolation parameters
class matchC2AGKNV:

    # Inputs:
    #     muKnown: list of known matching chemical potentials (GeV)
    #     c2Known: list of known matching speed of sound squares (unitless)
    #     lowDensity: physical quantities at the starting point of the interpolation
    #     highDensity: physical quantities at the ending point of the interpolation
    def __init__(self, muKnown, c2Known, lowDensity, highDensity):
        self.muKnown = muKnown
        self.c2Known = c2Known
        self.lowDensity = lowDensity

        self.pressureHigh = highDensity[0]
        self.densityHigh = highDensity[1]
        self.muHigh = highDensity[2]
        self.c2High = highDensity[3]
        
        self.resultsTol = False


    # Checking does the Unknown interpoaltion parameter, coeffUnknown,
    # agree with the known information about the EoS.
    def solveCoeff(self, coeffUnknown):
        # Original chemical potential and speed of sound square lists
        muList = self.muKnown[:]
        c2List = self.c2Known[:]

        # Including the last, still unknown, data point
        muList.append(coeffUnknown[0])
        c2List.append(coeffUnknown[1])

        # High density data point
        muList.append(self.muHigh)
        c2List.append(self.c2High)

        # c^2 EOS based on the given coefficients
        eos = c2AGKNV(muList, c2List, self.lowDensity)

        # Pressure at the upper bound
        pHigh = eos.pressureMu(self.muHigh)

        # Mass density at the upper bound
        rhoHigh = eos.rhoMu(self.muHigh)

        # Output list
        out = [pHigh / self.pressureHigh - 1.0]
        out.append(rhoHigh / self.densityHigh - 1.0)

        return out


    # Determinating the unknown coefficient pair
    # Input:
    #     tole: numerical tolerance (decault: 1.0e-10)
    def coeffValues(self, tole = 1.0e-10): 
        # initialization
        stopLooping = False
        coeffGuess = [0.0, 0.0]

        for j in range(len(self.muKnown)+2):
            for i in range(11):
                # Determinating the starting value of the search
                if j == len(self.muKnown):
                    coeffGuess[0] = self.muHigh + 0.01
                elif j > len(self.muKnown):
                    coeffGuess[0] = 5.0
                else:
                    coeffGuess[0] = self.muKnown[-j-1] + 0.01

                coeffGuess[1] = 0.01 + 0.1 * i

                [coeffs, infoC, flagC, mesgC] = fsolve(self.solveCoeff, coeffGuess, full_output=1, xtol = tole)

                testTol = self.solveCoeff(coeffs)

                # Checking is the correct results found
                if flagC == 1 and testTol[0] < tole and testTol[1] < tole:
                    stopLooping = True
                    break
            if stopLooping:
                break

        if not stopLooping:
            for i in range(100):
                for j in range(len(self.muKnown)+2):
                    if i%10 == 0:
                        continue

                    # Determinating the starting value of the search
                    if j == len(self.muKnown):
                        coeffGuess[0] = self.muHigh + 0.01
                    elif j > len(self.muKnown):
                        coeffGuess[0] = 5.0
                    else:
                        coeffGuess[0] = self.muKnown[-j-1] + 0.01

                    coeffGuess[1] = 0.01 + 0.01 * i

                    [coeffs, infoC, flagC, mesgC] = fsolve(self.solveCoeff, coeffGuess, full_output=1, xtol = tole)

                    testTol = self.solveCoeff(coeffs)

                    # Checking is the correct results found
                    if flagC == 1 and testTol[0] < tole and testTol[1] < tole:
                        stopLooping = True
                        break
                if stopLooping:
                    break        

        # Original chemical potential and speed of sound square lists
        muList = self.muKnown[:]
        c2List = self.c2Known[:]

        # Including the last, just solved, 
        muList.append(coeffs[0])
        c2List.append(coeffs[1])

        # High density data point
        muList.append(self.muHigh)
        c2List.append(self.c2High)

        return muList, c2List


    # Checking if the obtained coefficients are useable
    # Inputs:
    #     coeff: determined matching parameters
    #         [0]: chemical potential (GeV)
    #         [1]: speed of sound square (unitless) 
    #     tol: numerical tolerance 
    def coeffValuesOkTest(self, muList, c2List, tol = 1.0e-5):
        coeffMu = muList[-2]
        coeffC2 = c2List[-2]

        # Checking the causality etc.
        if coeffC2 > 1.0 or coeffC2 < 0.0:
            return False

        # Is the mathcing chemical potential in the right place?
        if coeffMu > self.muHigh or coeffMu < self.muKnown[-1]:
            return False

        coeffs = [coeffMu, coeffC2]
        testTol = self.solveCoeff(coeffs)

        # Are the gotten coefficients accurate enough?
        if testTol[0] > tol or testTol[1] > tol:
            return False
        
        return True