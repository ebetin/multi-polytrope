import units as cgs
import numpy as np

from scipy.special import hyp2f1

from scipy.optimize import fsolve

from math import pi, sin

import warnings
warnings.filterwarnings("error")

# This file contains speed of sound square (c^2) interpolation related formulas etc.
# The used interpolation method is from arXiv:1903.09121.

##########################################################
#constants

#speed of sound squared (cm^2/s^2)
cgsunits = cgs.c**2
cgsunits_inv = 1.0 / cgsunits

GeVfm_pre_dynecm_inv = 1.0 / cgs.GeVfm_per_dynecm
##########################################################

# This class determines the values of physical quantities, such as pressure and densities.
class c2AGKNV:
    #inverse baryon mass
    mB_inv = 1.0 / cgs.mB

    # giga electron volts in ergs
    GeV = 1.0e9 * cgs.eV

    # Inputs:
    #     muList: list of matching chemical potentials (GeV)
    #     c2List: list of matching speed of sound squares (unitless)
    #     lowDensity: physical quantities at the starting point of the interpolation
    def __init__(self, muList, c2List, lowDensity, approx = False, rhoHigh1 = 15.0, rhoHigh2 = 50.0 ):
        self.muList = muList
        self.c2List = c2List

        # Starting point (lowest density)
        self.p0 = lowDensity[0]   # Pressure (Ba)
        self.e0 = lowDensity[1]   # Energy density (g/cm^3)
        self.rho0 = lowDensity[2] # Baryon mass density (g/cm^3)
        self.c20 = lowDensity[3]  # speed of sound square (unitless)

        # Chemical potential at the starting point
        self.mu0 = cgs.mB * (self.e0 * cgsunits + self.p0) / ( self.rho0 * self.GeV )

        # Inserting the starting point into parameter lists
        self.muList.insert(0, self.mu0)
        self.c2List.insert(0, self.c20)

        self.c2List_len = len(self.c2List)

        self.approx = approx

        self.rhoHigh1 = rhoHigh1
        self.rhoHigh2 = rhoHigh2

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
            try:
                bar = bar**ai
            except:
                return np.inf

        return bar


    # Listing of mass densities in the matching points (cf. c2List or muList)
    def rhoListing(self):
        N = self.c2List_len
        listRho = N * [None]
        listRho[0] = self.rho0

        for i in range(1, N):
            listRho[i] = listRho[i-1] * self.rhoBar(self.muList[i], i)

        return listRho


    # Determinates the position of the segment from given mass density (rho; g/cm^3)
    def indexRho(self, rho):
        listInUse = self.rhoListing()
        
        try:
            return [listInUse.index(x) for x in listInUse if x > rho][0]

        except:
            return len(listInUse)-1


    # Calculates the chemical potential (GeV) from given mass density (rho; g/cm^3)
    def chemicalPotential(self, rho):
        index = self.indexRho(rho)

        ai_inv = 1.0 / self.ai(index)
        barRho = 1.0 * rho / self.rhoListing()[index - 1] # cf. rhoBar

        numerator = self.c2List[index - 1] * self.muList[index] 
        numerator -= self.c2List[index] * self.muList[index - 1]
        try:
            denominator = 1.0 + (1.0 * self.muList[index] / self.muList[index - 1] - 1.0) * barRho**ai_inv
        except:
            return 0.0
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

        aiNegInv = 1.0 / (1.0 + aiNegative)

        if type(termZ) is np.ndarray:
            termZinv = 1.0 / (1.0 - termZ[0])

            if -1.0 <= termZ[0] <= 1.0: # the hypergeometric function, abs(z)<=1 
                f = hyp2f1( 1.0, 1.0, 2.0 + aiNegative, termZ[0] ) * aiNegInv
            elif termZ[0] < -1.0: # analytical continuation, z < -1
                f = hyp2f1( 1.0, 1.0 + aiNegative, 2.0 + aiNegative, -termZinv * termZ[0] ) * aiNegInv * termZinv
            else: # analytical continuation, z > 1
                f = hyp2f1( 1.0, 1.0, 1.0 - aiNegative, 1.0 - termZ[0] ) / aiNegative
                f1 = (1.0 - termZ[0] + 0j)**aiNegative
                f2 = (termZ[0] + 0j)**(-1.0 - aiNegative)
                f3 = pi / sin(-pi * aiNegative)
                f = f + f1 * f2 * f3

            return np.array( [ float(((1.0 * mu *  ( 1.0 - f * aiNegative ) )[0]).real) ])

        else:
            termZinv = 1.0 / (1.0 - termZ)

            if -1.0 <= termZ <= 1.0: # the hypergeometric function, abs(z)<=1 
                f = hyp2f1( 1.0, 1.0, 2.0 + aiNegative, termZ ) * aiNegInv
            elif termZ < -1.0: # analytical continuation, z < -1
                f = hyp2f1( 1.0, 1.0 + aiNegative, 2.0 + aiNegative, -termZinv * termZ ) * termZinv * aiNegInv
            else: # analytical continuation, z > 1
                f = hyp2f1( 1.0, 1.0, 1.0 - aiNegative, 1.0 - termZ ) / aiNegative
                #if aiNegative > -1:
                try:
                    f1 = (1.0 - termZ + 0j)**aiNegative
                    f2 = (termZ + 0j)**(-1.0 - aiNegative)
                    f3 = pi / sin(-pi * aiNegative)
                    f = f + f1 * f2 * f3
                except:#else:
                    f = np.inf

            try:
                return ( 1.0 * mu * ( 1.0 - f * aiNegative ) ).real
            except:
                if (f>0 and aiNegative<0) or (f<0 and aiNegative>0):
                    return np.inf
                elif f==0 or aiNegative==0:
                    return mu
                else:
                    return -np.inf


    # Term in the pressure sum
    # Inputs:
    #     rho: Mass density (g/cm^3)
    #     index: position of the segment
    def pressureTerm(self, rho, index, muB = 0.0):
        try:
            if muB == 0.0:
                mu = self.chemicalPotential(rho)
            else:
                mu = muB
            rhoList = self.rhoListing()

            if mu < self.muList[index] and mu > self.muList[index - 1]:
                high = rho * self.pressurePartial(mu, index)
            elif mu >= self.muList[index]:
                high = rhoList[index] * self.pressurePartial(self.muList[index], index)
            else:
                return 0.0

            low = rhoList[index - 1] * self.pressurePartial(self.muList[index - 1], index)

            return self.GeV * np.float64(high - low) * self.mB_inv
        except:
            return np.inf


    # Term in the pressure sum
    # Inputs:
    #     mu: Chemical potential (GeV)
    #     index: position of the segment
    def pressureMuTerm(self, mu, index):
        if mu < self.muList[index] and mu > self.muList[index - 1]:
            high = self.rhoMu(mu) * self.pressurePartial(mu, index)
        elif mu >= self.muList[index]:
            try:
                high = self.rhoMu(self.muList[index]) * self.pressurePartial(self.muList[index], index)
            except:
                high = np.inf
        else:
            return 0.0

        low = self.rhoMu(self.muList[index - 1]) * self.pressurePartial(self.muList[index - 1], index)

        if np.isinf(high) and np.isinf(low):
            return np.inf # indefinite
        else:
            return self.GeV * np.float64(high - low) * self.mB_inv


    # Pressure (Ba) as a function of chemical potential (GeV)
    def pressureMu(self, mu):
        try:
            N = self.c2List_len
            sm = self.p0

            for i in range(1, N):
                sm += self.pressureMuTerm(mu, i)

            return sm
        except:
            return np.inf


    # Baryon mass density (g/cm^3) as a function of chemical potential (GeV)
    def rhoMu(self, mu):
        try:
            N = self.c2List_len
            prd = self.rho0

            for i in range(1, N):
                prd *= self.rhoBar(mu, i)

            return prd
        except:
            return np.inf

    ################################################
    def pec2inv(self, rho):
        #baryon chemical potential
        mu = self.chemicalPotential(rho)

        #calculating pressure
        def pressureTermFlatten(i):
            return self.pressureTerm(rho, i, mu)

        N = self.c2List_len

        pressureSum = self.p0
        pressureSum += sum( map( pressureTermFlatten, range(1, N) ) )

        #energy density
        eps = (rho * mu * self.GeV * self.mB_inv - pressureSum) * cgsunits_inv

        #inverse speed of sound square
        index = self.indexRho(rho)

        numerator = self.c2List[index - 1] * ( self.muList[index] - mu )
        numerator += self.c2List[index] * ( mu -self.muList[index - 1] )
        denominator = self.muList[index] - self.muList[index - 1]

        c2inv = denominator / numerator

        return pressureSum, eps, c2inv


    def approximation(self):
            N1 = 2000 # If one wants to be (very) save, use N = 2,000 (10,000)
            listRho1 = np.linspace(self.rho0, self.rhoHigh1 * cgs.rhoS, N1)
            listP1 = np.empty(N1)
            listE1 = np.empty(N1)
            listC2inv1 = np.empty(N1)

            for i, r in enumerate(listRho1):
                listP1[i], listE1[i], listC2inv1[i] = self.pec2inv(r)

            self.listRhoLong = listRho1
            self.listPLong = listP1
            self.listELong = listE1
            self.listC2invLong = listC2inv1

            N2 = 100
            listRho2 = np.linspace(self.rhoHigh1 * cgs.rhoS, self.rhoHigh2 * cgs.rhoS, N2) #TODO
            listP2 = np.empty(N2)
            listE2 = np.empty(N2)
            listC2inv2 = np.empty(N2)

            for i, r in enumerate(listRho2):
                listP2[i], listE2[i], listC2inv2[i] = self.pec2inv(r)

            self.listRhoLongHigh = listRho2
            self.listPLongHigh = listP2
            self.listELongHigh = listE2
            self.listC2invLongHigh = listC2inv2


    def press_edens_c2(self, rho):
        if rho <= self.listRhoLong[-1]: #TODO
            press = np.interp(rho, self.listRhoLong, self.listPLong)
            edens = np.interp(rho, self.listRhoLong, self.listELong)
            speed2 = 1.0 / np.interp(rho, self.listRhoLong, self.listC2invLong)

            return press, edens, speed2
        elif rho <= self.listRhoLongHigh[-1]:
            press = np.interp(rho, self.listRhoLongHigh, self.listPLongHigh)
            edens = np.interp(rho, self.listRhoLongHigh, self.listELongHigh)
            speed2 = 1.0 / np.interp(rho, self.listRhoLongHigh, self.listC2invLongHigh)

            return press, edens, speed2
        else:
            #baryon chemical potential
            mu = self.chemicalPotential(rho)

            #calculating pressure
            def pressureTermFlatten(i):
                return self.pressureTerm(rho, i, mu)

            N = self.c2List_len

            pressureSum = self.p0
            pressureSum += sum( map( pressureTermFlatten, range(1, N) ) )

            #energy density
            eps = (rho * mu * self.GeV * self.mB_inv - pressureSum) * cgsunits_inv

            #inverse speed of sound square
            index = self.indexRho(rho)

            numerator = self.c2List[index - 1] * ( self.muList[index] - mu )
            numerator += self.c2List[index] * ( mu -self.muList[index - 1] )
            denominator = self.muList[index] - self.muList[index - 1]

            c2 = numerator / denominator

            return pressureSum, eps, c2


    ################################################
    ################################################

    # Pressure (Ba) as a function of the mass density (g/cm)
    def pressure(self, rho, p = 0.0, muB = 0.0):
        N = self.c2List_len
        pressureSum = self.p0

        def pressureTermFlatten(i):
            return self.pressureTerm(rho, i, muB)

        pressureSum += sum( map( pressureTermFlatten, range(1, N) ) )

        return pressureSum - p


    # Vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press


    # Energy density (g/cm^3) as a function of the pressure (Ba)
    def edens_inv(self, pressure, e = 0, approx = 2, muB = 0.0):
        if approx == 0:
            rho = self.rho(pressure, False) # mass density (g/cm^3)
        elif approx == 1:
            if pressure <= self.listPLong[-1]:
                rho = self.rho(pressure, True) # mass density (g/cm^3)
            else:
                rho = self.rho(pressure, False) # mass density (g/cm^3)
        else:
            if self.approx:
                if pressure <= self.listPLong[-1]:
                    rho = self.rho(pressure, self.approx) # mass density (g/cm^3)
                else:
                    rho = self.rho(pressure, False) # mass density (g/cm^3)
            else:
                rho = self.rho(pressure, False) # mass density (g/cm^3)

        if muB == 0.0:
            mu = self.chemicalPotential(rho) * self.GeV # chem.pot. (ergs)
        else:
            mu = muB * self.GeV

        return (rho * mu * self.mB_inv - pressure) * cgsunits_inv - e


    # Mass density (g/cm^3) as a function of the pressure (Ba)
    def rho(self, pressure, approx):
        if approx:
            if pressure > self.p0:
                try:
                    return np.interp(pressure, self.listPLong, self.listRhoLong)
                except:
                    rho = fsolve(self.pressure, 2.0*cgs.rhoS, args = pressure)[0]

                    return rho
            else:
                return self.rho0
        else:
            rho = fsolve(self.pressure, 2.0*cgs.rhoS, args = pressure)[0]

            return rho


    # Speed of sound squared (unitless) as a function of the pressure (Ba)
    def speed2(self, pressure, muB = 0.0):
        try:
            rho = self.rho(pressure, self.approx) # mass density (g/cm^3)
        except:
            rho = self.rho(pressure, False)

        if muB == 0.0:
            mu = self.chemicalPotential(rho) # chem.pot. (GeV)
        else:
            mu = muB

        index = self.indexRho(rho)

        numerator = self.c2List[index - 1] * ( self.muList[index] - mu ) 
        numerator = numerator + self.c2List[index] * ( mu -self.muList[index - 1] )
        denominator = self.muList[index] - self.muList[index - 1]

        return 1.0 * numerator / denominator

    def speed2_rho(self, rho, muB = 0.0):
        if muB == 0.0:
            mu = self.chemicalPotential(rho) # chem.pot. (GeV)
        else:
            mu = muB
        index = self.indexRho(rho)

        numerator = self.c2List[index - 1] * ( self.muList[index] - mu ) 
        numerator = numerator + self.c2List[index] * ( mu -self.muList[index - 1] )
        denominator = self.muList[index] - self.muList[index - 1]

        return 1.0 * numerator / denominator


    def gammaFunction(self, rho, flag = 1):
        press = self.pressure(rho)
        edens = self.edens(rho) * cgsunits
        speed2 = self.speed2_rho(rho)

        if flag == 1: # d(ln p)/d(ln n)
            return ( edens / press + 1.0 ) * speed2
        else: # d(ln p)/d(ln eps)
            return edens / press * speed2

    # Energy density (g/cm^3) as a function of the mass density (g/cm^3)
    def edens(self, rho, e = 0, muB = 0.0):
        if muB == 0.0:
            mu = self.chemicalPotential(rho) * self.GeV # chem.pot. (ergs)
        else:
            mu = muB * self.GeV
        pressure = self.pressure(rho)

        return (rho * mu * self.mB_inv - pressure) * cgsunits_inv - e


    def pressure_edens(self, edens):
        if self.approx:
            try:
                if edens <= self.listELong[-1]:
                    return np.interp(edens, self.listELong, self.listPLong)
                elif edens <= self.listELongHigh[-1]:
                    return np.interp(edens, self.listELongHigh, self.listPLongHigh)
                else:
                    edensGeV =  edens * cgsunits * GeVfm_pre_dynecm_inv
                    rho = fsolve(self.edens, self.listELong[-1], args = edens)[0]
                    #print( (np.interp(edens, self.listELongHigh, self.listPLongHigh)- self.pressure(rho) ) / self.pressure(rho))
                    return self.pressure(rho)
                    return np.interp(edens, self.listELongHigh, self.listPLongHigh)
            except:
                edensGeV =  edens * cgsunits * GeVfm_pre_dynecm_inv
                rho = fsolve(self.edens, cgs.rhoS, args = edens)[0]
                return self.pressure(rho)
        else:
            edensGeV =  edens * cgsunits * GeVfm_pre_dynecm_inv

            #if edensGeV < 0.7:
            #    rhoEstimate = edensGeV + 0.05
            #else:
            #    rhoEstimate = 0.4 * edensGeV + 0.5

            #rho = fsolve(self.edens, rhoEstimate * cgs.mB * 1.0e39, args = edens)[0]
            rho = fsolve(self.edens, 1.1*cgs.rhoS, args = edens)[0]

            return self.pressure(rho)

    def tov(self, press):
        if self.approx:
            try:
                #eden = self.ep_interp(press)
                #speed2inv = self.c2invp_interp(press)
                eden      = np.interp(press, self.listPLong, self.listELong)
                speed2inv = np.interp(press, self.listPLong, self.listC2invLong)
            except:
                eden      = self.edens_inv(press)
                speed2inv = 1.0 / self.speed2(press)
        else:
            eden      = self.edens_inv(press)
            speed2inv = 1.0 / self.speed2(press)

        return eden, speed2inv



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

        self.pressureHighInv = 1.0 / highDensity[0]
        self.densityHighInv = 1.0 / highDensity[1]
        self.muHigh = highDensity[2]
        self.c2High = highDensity[3]
        
        self.resultsTol = False


    # Checking does the Unknown interpoaltion parameter, coeffUnknown,
    # agree with the known information about the EoS.
    def solveCoeff(self, coeffUnknown):
        # Original chemical potential and speed of sound square lists
        #muList = self.muKnown[:]
        #c2List = self.c2Known[:]

        # Including the last, still unknown, data point
        #muList.append(coeffUnknown[0])
        #c2List.append(coeffUnknown[1])

        muList = self.muListPre[:]
        c2List = self.c2ListPre[:]

        muList[-2] = coeffUnknown[0]
        c2List[-2] = coeffUnknown[1]

        # High density data point
        #muList.append(self.muHigh)
        #c2List.append(self.c2High)

        # c^2 EOS based on the given coefficients
        eos = c2AGKNV(muList, c2List, self.lowDensity)

        # Pressure at the upper bound
        pHigh = eos.pressureMu(self.muHigh)

        # Mass density at the upper bound
        rhoHigh = eos.rhoMu(self.muHigh)

        # Output list
        out = [pHigh * self.pressureHighInv - 1.0]
        out.append(rhoHigh * self.densityHighInv - 1.0)

        return out


    # Determinating the unknown coefficient pair
    # Input:
    #     tole: numerical tolerance (decault: 1.0e-10)
    def coeffValues(self, tole = 1.0e-10): 
        # initialization
        stopLooping = False
        coeffGuess = [0.0, 0.0]

        # Original chemical potential and speed of sound square lists
        muList = self.muKnown[:]
        c2List = self.c2Known[:]

        muList = muList + [0.0, self.muHigh]
        c2List = c2List + [0.0, self.c2High]

        self.muListPre = muList[:]
        self.c2ListPre = c2List[:]

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

        self.muSolved = coeffs[0]
        self.c2Solved = coeffs[1]

        # Original chemical potential and speed of sound square lists
        #muList = self.muKnown[:]
        #c2List = self.c2Known[:]

        # Including the last, just solved, 
        #muList.append(coeffs[0])
        #c2List.append(coeffs[1])

        muList[-2] = coeffs[0]
        c2List[-2] = coeffs[1]

        # High density data point
        #muList.append(self.muHigh)
        #c2List.append(self.c2High)

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
