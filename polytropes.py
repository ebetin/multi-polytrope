import units as cgs
from scipy.optimize import fsolve
from numpy import sqrt
from math import pi
from math import isnan
from math import log


class Error(Exception):
    """Base class for other exceptions"""
    pass

class IncorrectSymmetryEnergyModelError(Error):
    """Raised when the symmetry energy model is unvalid"""
    pass


#Monotropic eos
class monotrope:
    
    #transition continuity constant (unitless)
    a = 0.0

    #speed of sound squared (cm^2/s^2)
    cgsunits = cgs.c**2

    def __init__(self, K, G):
        #polytropic constant (a.u.)
        #  NB self.K in cgs units and K is gotten when c=1
        self.K = K / self.cgsunits

        #polytropic exponent (unitless)
        self.G = G

        #polytropic index (unitless)
        self.n = 1.0/(G - 1)

    #pressure P(rho)
    #  pressure (Ba)
    #  rho: mass density (g/cm^3)
    def pressure(self, rho):
        return self.cgsunits * self.K * rho**self.G

    #energy density eps(rho) (NB g/cm^3, not erg/(cm s^2)!)
    #  edens: energy density (g/cm^3)
    #  rho: mass density (g/cm^3)
    def edens(self, rho):
        if 1.0 - cgs.epsilonGamma < self.G < 1.0 + cgs.epsilonGamma:
            return (self.K * log(rho / cgs.mB) + 1.0 + self.a) * rho
        else:
            return (1.0 + self.a) * rho + (self.K / (self.G - 1.0)) * rho**self.G


    #energy density eps(P) (NB g/cm^3, not erg/(cm s^2)!)
    #  edens: energy density (g/cm^3)
    #  pressure (Ba)
    def edens_inv(self, pressure):
        rho = self.rho(pressure)

        if 1.0 - cgs.epsilonGamma < self.G < 1.0 + cgs.epsilonGamma:
            return (self.K * log(rho / cgs.mB) + 1.0 + self.a) * rho
        else:
            return (1.0 + self.a) * rho + pressure / ( (self.G - 1.0) * self.cgsunits )


    #for inverse functions lets define rho(P)
    #  press: pressure (Ba)
    #  rho: mass density (g/cm^3)
    def rho(self, press):
        if press < 0.0:
            return 0.0
        return ( press / (self.cgsunits * self.K) )**(1.0 / self.G)

    # First derivative of the pressure with respect to the (baryon) number density (1/cm^3)
    def Dpressure(self, rho):
        return self.cgsunits * self.K * self.G * cgs.mB * rho**(self.G-1.0)

    # First derivative of the energy density with respect to the (baryon) number density (1/cm^3) multiplied by c^2
    def Dedens(self, rho):
        if 1.0 - cgs.epsilonGamma < self.G < 1.0 + cgs.epsilonGamma:
            return (self.K * (log(rho / cgs.mB) + 1.0) + 1.0 + self.a) * cgs.mB * self.cgsunits
        else:
            return ( 1.0 + self.a + self.G / (self.G - 1.0) * self.K * rho**(self.G-1.0) ) * cgs.mB * self.cgsunits



# Piecewise polytropes
class polytrope:
    
    def __init__(self, tropes, trans, prev_trope = None ):
        #  prev_trope: previous monotrope which is not icnluded in the tropes (optional)

        #ordered list of monotropes (from low-density to high-density)
        self.tropes      = tropes

        #ordered list of matching/transition points, mass density (g/cm^3), between monotropes
        self.transitions = trans

        #transition/matching pressures (Ba) and energy densities (g/cm^3)
        self.prs  = []
        self.eds  = []

        for (trope, transition) in zip(self.tropes, self.transitions):
            # Does a previous polytrope (prev_trope) exist?
            if not( prev_trope == None ):
                #transition continuity constant
                trope.a = self._ai( prev_trope, trope, transition )

            #no previous trope, ie. this is the crust monotrope
            else:
                transition = 0.0


            ed = trope.edens(transition) 
            pr = trope.pressure(transition)

            self.prs.append( pr )
            self.eds.append( ed )

            prev_trope = trope


    #transition continuity constant (unitless)
    #  pm: previous monotrope, ie. the monotrope before matching point
    #  m: monotrope after the matching point
    #  tr: transtion mass density (g/cm^3)
    def _ai(self, pm, m, tr):
        pmG1 = 1.0 - cgs.epsilonGamma < pm.G < 1.0 + cgs.epsilonGamma
        mG1 = 1.0 - cgs.epsilonGamma < m.G < 1.0 + cgs.epsilonGamma

        if pmG1 and mG1:
            return pm.a
        elif pmG1 and not mG1:
            return pm.a + pm.K * ( log(tr / cgs.mB) - 1.0 / (m.G - 1.0) )
        elif mG1 and not pmG1:
            return pm.a - m.K * ( log(tr / cgs.mB) - 1.0 / (pm.G - 1.0) )
        else:
            return pm.a + (pm.K / (pm.G - 1.0)) * tr**(pm.G - 1.0) - (m.K / (m.G - 1.0)) * tr**(m.G - 1.0)


    #finds the correct monotrope for given (mass) density (g/cm^3)
    def _find_interval_given_density(self, rho):
        if rho <= self.transitions[0]:
            return self.tropes[0]

        for q in range( len(self.transitions) - 1 ):
            if self.transitions[q] <= rho < self.transitions[q+1]:
                return self.tropes[q]

        return self.tropes[-1]


    #inverted equations as a function of pressure (Ba)
    def _find_interval_given_pressure(self, press):
        if press <= self.prs[0]:
            return self.tropes[0]

        for q in range( len(self.prs) - 1):
            if self.prs[q] <= press < self.prs[q+1]:
                return self.tropes[q]

        return self.tropes[-1]


    ################################################## 
    def pressure(self, rho):
        trope = self._find_interval_given_density(rho)
        return trope.pressure(rho)

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    def edens_inv(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.edens_inv(press)

    def rho(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.rho(press)

    # Square of the speed of sound (unitless)
    def speed2(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.G * press / (press + trope.edens_inv(press) * cgs.c**2) 



class doubleMonotrope:

    G = 1.0
    
    # This class clues together double monotropic EoSs (Gandolfy type)
    # Inputs:
    #   trope = [trope1, trope2]: A list of monotropes (class monotrope objects)
    #   trans: Transition density between crust and the Gandolfy EoS (g/cm^3)
    #   S: Symmetry energy at the saturation density (erg)
    #   L: Derivative of the symmetry energy at the saturation density (erg)
    #   rhoS: Saturation density (g/cm^3)
    #   m: Mass of the baryon (g)
    #   flagBetaEQ: If True, the output is at beta-equilibrium (default True)
    #   flagMuon: If True, muons contribution is also included (default False)
    #   flagSymmetryEnergyModel: Symmetry energy models (default 1)
    #      1: SS(n) = S + L / 3 * (rho - rhoS) / rhoS
    #      2: SS(n) = S * (rho / rhoS)^G, where G = L / (3 * S)
    def __init__(self, trope, S, L, flagBetaEQ = True, flagMuon = False, flagSymmetryEnergyModel = 1):
        self.trope1      = trope[0]
        self.trope2      = trope[1]

        self.S = S
        self.L = L

        self.alphaCoeff = self.trope1.G - 1.0
        self.betaCoeff = self.trope2.G - 1.0
        self.aCoeff = self.trope1.K * cgs.mB * (cgs.rhoS / cgs.mB)**self.alphaCoeff / self.alphaCoeff
        self.bCoeff = self.trope2.K * cgs.mB * (cgs.rhoS / cgs.mB)**self.betaCoeff / self.betaCoeff

        self.flagMuon = flagMuon
        self.flagBetaEQ = flagBetaEQ
        self.flagSymmetryEnergyModel = flagSymmetryEnergyModel

    # Determines the value of the electron and muon factors (eg. x_e = rhoo_e / rhoo_B) equation 
    # Inputs:
    #   rho: mass density (g/cm^3)
    #   factors = [xe, xm]: electron (xe) and muon (xm) factors 
    # Output:
    #   out: list which contains two elements
    def protonFactorMuon(self, factors, rho):
        try:
            [electronFactor, muonFactor] = factors

            # Symmetry energy
            if self.flagSymmetryEnergyModel == 1:
                SS = self.S + self.L / 3.0 * (rho - cgs.rhoS) / cgs.rhoS
            elif self.flagSymmetryEnergyModel == 2:
                G = self.L / (3.0 * self.S)
                SS = self.S * pow(rho / cgs.rhoS, G)
            else:
                raise IncorrectSymmetryEnergyModelError

            # Temp functions
            coeff = 3.0 * pi**2 * rho / cgs.mB 
            A = cgs.hbar * cgs.c * pow(coeff, 1.0/3.0) / (8.0 * SS)
            B = (cgs.mmu * cgs.c / cgs.hbar)**2 * pow(coeff, -2.0/3.0)

            # Factor equations
            out = []   
            if electronFactor < 0.0 or muonFactor < 0.0: 
                out = [10.1, -10.1]
            else:
                out.append(muonFactor + electronFactor - 0.5 + A * pow(electronFactor, 1.0/3.0) )
                out.append(pow(electronFactor, 2.0/3.0) - pow(muonFactor, 2.0/3.0)  - B)

            return out

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print()


    # Determines the electron fraction in beta stabile matter when muons are not present, as a function of baryon mass density (g/cm^3)
    def electronFactor(self, rho):
        try:
            # Symmetry energy
            if self.flagSymmetryEnergyModel == 1:
                SS = self.S + self.L / 3.0 * (rho - cgs.rhoS) / cgs.rhoS
            elif self.flagSymmetryEnergyModel == 2:
                G = self.L / (3.0 * self.S)
                SS = self.S * pow(rho / cgs.rhoS, G)
            else:
                raise IncorrectSymmetryEnergyModelError

            # Temp variables
            chemicalPotential3 = (cgs.hbar * cgs.c)**3 * (3.0 * pi**2 * rho / cgs.mB)
            SQRT = 6.0 * (chemicalPotential3**3 + 864.0 * chemicalPotential3**2 * pow(SS, 3.0))
            coeff = pow(-72.0 * chemicalPotential3 * pow(SS, -3.0) + sqrt(SQRT) * pow(SS, -4.5), 1.0/3.0)

            xePart1 = chemicalPotential3 * pow(6.0, -1.0/3.0) * pow(SS, -3.0) / coeff
            xePart2 = coeff * pow(6.0, -2.0/3.0)

            # Electron factor
            xe = 0.0625 * (8.0 - xePart1 + xePart2) 

            return xe

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print() 


    # Calculates the electron (xe) and muon (xm) factors as a function of the mass density rho (g/cm^3)
    # The output is a list [xe, xm]
    def electronMuonFactors(self, rho):
        # Non-beta eq. (only neutrons)
        if not self.flagBetaEQ:
            xe = 0.0 # Electron factor rhoo_e / rhoo_B
            xm = 0.0 # Muon factor rhoo_{mu} / rhoo_B

        # Beta eq. with muons
        elif self.flagMuon and self.flagBetaEQ:
            xe = self.electronFactor(rho)

            # NB Muons do not exist if xe <= B^(3/2)
            B = (cgs.mmu * cgs.c / cgs.hbar)**2.0 * pow(3.0 * pi**2.0 * rho / cgs.mB, -2.0/3.0)

            # Muons are present
            #   NB The if statement works at least with the current symmetry energy models!
            if xe > pow(B, 1.5) or isnan(xe):
                [[xe, xm],info,flag,mesg] = fsolve(self.protonFactorMuon, [xe, 0.0], args = rho, full_output=1, xtol=1.0e-9)

                if flag != 1:
                    [[xe, xm],info,flag,mesg] = fsolve(self.protonFactorMuon, [1.0e-2, 1.0e-2], args = rho, full_output=1, xtol=1.0e-9)
                    if flag != 1:
                        xe = -1.0
                        xm = -1.0

            else: # Without muons
                xm = 0.0

        # Beta eq. with only electrons
        else:
            xm = 0.0;
            xe = self.electronFactor(rho)

        return [xe, xm]


    # Pressure (Ba) as a function of the mass density rho (g/cm^3)
    def pressure(self, rho, p=0.0):
        try:
            pressure1 = self.trope1.pressure(rho)
            pressure2 = self.trope2.pressure(rho)

            # Electron and muon factors (xe = rho_e / rho_B)
            [xe, xm] = self.electronMuonFactors(rho)

            # Proton factor rho_p / rho_B
            x = xe + xm

            if x > 0.0: # Beta eq.
                # Derivative of the symmetry energy multiplied by rhoo
                if self.flagSymmetryEnergyModel == 1:
                    DS = self.L / 3.0 * (rho / cgs.rhoS)
                elif self.flagSymmetryEnergyModel == 2:
                    G = self.L / (3.0 * self.S)
                    DS = G * self.S * pow(rho / cgs.rhoS, G)
                else:
                    raise IncorrectSymmetryEnergyModelError

                # Baryon number density
                nB = rho / cgs.mB

                # Temp constants
                pi2nB = pi**2 * nB
                coeff = 3.0 * pi2nB
                ch = cgs.c * cgs.hbar

                # Proton correction
                pressureProton = 4.0 * DS * x * (x - 1.0) * nB

                # Electron correction
                pressureElectron = 0.25 * xe * ch * pow(coeff * xe, 1.0/3.0) * nB

            else: # Non-beta eq.
                pressureProton = 0.0
                pressureElectron = 0.0


            if xm > 0.0: # w/ muons
                # Temp constant
                c2m2 = (cgs.c * cgs.mmu)**2
                hbar2 = cgs.hbar**2
                coeffPower = pow(coeff * xm, 1.0/3.0)

                # Muon chemical potential diveded by c
                chemicalPotentialMuon = sqrt( c2m2 + hbar2 * coeffPower**2 )

                pressureMuonLogless = 2.0 * ch * chemicalPotentialMuon * (-2.0 * hbar2 * pi2nB * xm + c2m2 * coeffPower)

                pressureMuonLog = c2m2**2 * cgs.c * (log(c2m2) - 2.0 * log(cgs.hbar * coeffPower + chemicalPotentialMuon) )

                # Muon correction
                pressureMuon = -(pressureMuonLog + pressureMuonLogless) / (16.0 * cgs.hbar**3 * pi**2)

            else: # w/out muons
                pressureMuon = 0.0
            

            pressure = pressure1 + pressure2 + pressureProton + pressureElectron + pressureMuon

            return pressure - p

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print()


    #vectorized version
    def pressures(self, rhos, p=0):
        press = []
        for rho in rhos:
            pr = self.pressure(rho, p)
            press.append( pr )
        return press


    # Energy density (g/cm^3) as a function of the mass density rho (g/cm^3)
    def edens(self, rho):
        try:
            # Electron and muon factors (xe = rho_e/rho_B)
            [xe, xm] = self.electronMuonFactors(rho)

            # Proton factor rho_p / rho_B
            x = xe + xm

            # Monotropic
            energyDensity1 = self.trope1.edens(rho) # a-alpha part ("low" density)
            energyDensity2 = self.trope2.edens(rho) # b-beta part ("high" density)

            if x > 0.0: # Beta eq.
                # Symmetry energy
                if self.flagSymmetryEnergyModel == 1:
                    SS = self.S + self.L / 3.0 * (rho - cgs.rhoS) / cgs.rhoS
                elif self.flagSymmetryEnergyModel == 2:
                    G = self.L / (3.0 * self.S)
                    SS = self.S * pow(rho / cgs.rhoS, G)
                else:
                    raise IncorrectSymmetryEnergyModelError

                # Baryon number density
                nB = rho / cgs.mB

                # Temp constant
                pi2nB = pi**2 * nB
                coeff = 3.0 * pi2nB
                nBc2Inv = nB / cgs.c**2

                # Proton correction
                energyDensityProton = 4.0 * SS * x * (x - 1.0) * nBc2Inv

                # Electron correction
                energyDensityElectron = 0.75 * xe * cgs.hbar * cgs.c * pow(coeff * xe, 1.0/3.0) * nBc2Inv

            else: # Non-beta eq.
                energyDensityProton = 0.0
                energyDensityElectron = 0.0
           

            if xm > 0.0: # w/ muons
                # Temp constant
                c2m2 = (cgs.c * cgs.mmu)**2
                cInv = 1.0 / cgs.c
                hbar2 = cgs.hbar**2
                coeffPower = pow(coeff * xm, 1.0/3.0)

                # Muon chemical potential diveded by c
                chemicalPotentialMuon = sqrt( c2m2 + hbar2 * coeffPower**2 )

                energyDensityMuonLogless = 2.0 * cgs.hbar * chemicalPotentialMuon * (6.0 * hbar2 * pi2nB * xm * cInv + cgs.mmu**2 * coeffPower)

                energyDensityMuonLog = c2m2**2 * cInv * (log(c2m2) - 2.0 * log(cgs.hbar * coeffPower + chemicalPotentialMuon) )

                # Muon correction
                energyDensityMuon = (energyDensityMuonLog + energyDensityMuonLogless) / (16.0 * cgs.hbar**3 * pi**2)

            else: # w/out muons
                energyDensityMuon = 0.0


            # Total energy density
            energyDensity = energyDensity1 + energyDensity2 + energyDensityProton + energyDensityElectron + energyDensityMuon

            return energyDensity

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print()



    def edens_inv(self, press):
        rho = self.rho(press)
        
        return self.edens(rho)

    def rho(self, press):
        rho = fsolve(self.pressures, cgs.rhoS, args = press)

        return rho[0]

    # Square of the speed of sound (unitless)
    def speed2(self, press):
        try:
            rho = self.rho(press)

            DenergyDensity1 = self.trope1.Dedens(rho) # a-alpha part ("low" density)
            DenergyDensity2 = self.trope2.Dedens(rho) # b-beta part ("high" density)
            Dpressure1 = self.trope1.Dpressure(rho)
            Dpressure2 = self.trope2.Dpressure(rho)

            # Electron and muon factors (xe = rho_e / rho_B)
            [xe, xm] = self.electronMuonFactors(rho)

            # Proton factor rho_p / rho_B
            x = xe + xm

            if x > 0.0: # Beta eq.
                # Symmetry energy related parameters
                if self.flagSymmetryEnergyModel == 1:
                    SS = self.S + self.L / 3.0 * (rho - cgs.rhoS) / cgs.rhoS
                    DS = self.L / 3.0 * (rho / cgs.rhoS)
                    DDS = 2.0 * DS
                elif self.flagSymmetryEnergyModel == 2:
                    G = self.L / (3.0 * self.S)
                    SS = self.S * pow(rho / cgs.rhoS, G)
                    DS = G * SS
                    DDS = (G + 1.0) * DS
                else:
                    raise IncorrectSymmetryEnergyModelError

                # Baryon number density
                nB = rho / cgs.mB

                # Temp constants
                pi2nB = pi**2 * nB
                coeff = pi2nB / 9.0
                ch = cgs.c * cgs.hbar
                coeffProton = 4.0 * x * (x - 1.0)

                # Proton correctin
                DpressureProton = coeffProton * DDS
                DenergyDensityProton = coeffProton * (SS + DS)

                # Electron correction
                DpressureElectron = xe * ch * pow(coeff * xe, 1.0/3.0)
                DenergyDensityElectron = 3.0 * DpressureElectron

            else: # Non-beta eq.
                DpressureProton = 0.0
                DpressureElectron = 0.0

                DenergyDensityProton = 0.0
                DenergyDensityElectron = 0.0


            if xm > 0.0: # w/ muons
                # Temp constant
                c2m2 = (cgs.c * cgs.mmu)**2
                hbar2 = cgs.hbar**2
                coeffPower = pow(coeff * xm, 1.0/3.0)
                coeffMuon = cgs.c * xm

                # Muon chemical potential diveded by c
                chemicalPotentialMuon = sqrt( c2m2 + hbar2 * coeffPower**2 )

                # Muon correction
                DpressureMuon = coeffMuon * hbar2 * pi * pow(pi * (nB * xm)**2 / 3.0, 1.0/3.0) / chemicalPotentialMuon
                DenergyDensityMuon = coeffMuon * chemicalPotentialMuon

            else: # w/out muons
                DpressureMuon = 0.0
                DenergyDensityMuon = 0.0
            
            Dpressure = Dpressure1 + Dpressure2 + DpressureProton + DpressureElectron + DpressureMuon
            DenergyDensity = DenergyDensity1 + DenergyDensity2 + DenergyDensityProton + DenergyDensityElectron + DenergyDensityMuon

            return Dpressure / DenergyDensity

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print()



#Combines EoS parts together, SPECIAL CASE
class combiningEos:
    
    def __init__(self, pieces, trans):
        #  prev_trope: previous monotrope which is not icnluded in the tropes (optional)

        #ordered list of monotropes (from low-density to high-density)
        self.pieces = pieces

        #ordered list of matching/transition points, mass density (g/cm^3), between monotropes
        self.transitions = trans

        #transition/matching pressures (Ba) and energy densities (g/cm^3)
        self.prs  = []
        self.eds  = []


        for (piece, transition) in zip(self.pieces, self.transitions):
            pr = piece.pressure(transition)
            ed = piece.edens_inv(pr)

            self.prs.append( pr )
            self.eds.append( ed )

            prev_piece = piece



    #finds the correct monotrope for given (mass) density (g/cm^3)
    def _find_interval_given_density(self, rho):
        if rho <= self.transitions[0]:
            return self.pieces[0]

        for q in range( len(self.transitions) - 1 ):
            if self.transitions[q] <= rho < self.transitions[q+1]:
                part = self.pieces[q]
                partLatter = self.pieces[q+1]
                K = partLatter.pressure(self.transitions[q+1])

                # Is the latent heat non-negative?
                if part.pressure(self.transitions[q+1]) >= (1.0 - 2.0e-8) * K:

                    if part.pressure(rho) > K:
                        # Create a EoS for the latent heat part
                        mono = monotrope(K, 0.0)
                        mono.a = (self.eds[q+1] + K) / self.transitions[q+1] - 1.0
                        poly = polytrope([mono], [0.0])
                        return poly
                    else:
                        return part

                # Negative latent heat!
                else:
                    poly = polytrope([monotrope(-1.0, 0.0)], [0.0])
                    return poly

        return self.pieces[-1]

    #inverted equations as a function of pressure (Ba)
    def _find_interval_given_pressure(self, press):
        if press <= self.prs[0]:
            return self.pieces[0]

        for q in range( len(self.prs) - 1):
            if self.prs[q] <= press < self.prs[q+1]:
                return self.pieces[q]

        return self.pieces[-1]


    ################################################## 
    def pressure(self, rho):
        trope = self._find_interval_given_density(rho)
        return trope.pressure(rho)

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    def edens_inv(self, press):
        trope = self._find_interval_given_pressure(press)

        return trope.edens_inv(press)

    def rho(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.rho(press)

    def speed2(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.speed2(press)
