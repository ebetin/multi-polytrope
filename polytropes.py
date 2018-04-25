import units as cgs
from scipy.optimize import fsolve
from numpy import sqrt
from numpy import log, exp
from math import pi
from math import isnan


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

    #energy density mu(rho) (NB g/cm^3, not erg/(cm s^2)!)
    #  edens: energy density (g/cm^3)
    #  rho: mass density (g/cm^3)
    def edens(self, rho):
        if self.G == 1.0:
            return (self.K * log(rho / cgs.mB) + 1.0 + self.a) * rho
        else:
            return (1.0 + self.a) * rho + (self.K / (self.G - 1.0)) * rho**self.G

    #for inverse functions lets define rho(P)
    #  press: pressure (Ba)
    #  rho: mass density (g/cm^3)
    def rho(self, press):
        if press < 0.0:
            return 0.0
        return ( press / (self.cgsunits * self.K) )**(1.0 / self.G)



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
        if pm.G == 1.0 and m.G == 1.0:
            return pm.a
        elif pm.G == 1.0 and m.G != 1.0:
            return pm.a + pm.K * ( log(tr / cgs.mB) - 1.0 / (p.G - 1.0) ) 
        elif pm.G != 1.0 and m.G == 1.0:
            return pm.a - p.K * (log(tr / cgs.mB) - 1.0 / (pm.G - 1.0))
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
        rho = trope.rho(press)
        return trope.edens(rho)

    def rho(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.rho(press)



class doubleMonotrope:
    
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
            out.append(muonFactor + electronFactor - 0.5 + A * pow(electronFactor, 1.0/3.0) )
            out.append(pow(electronFactor, 2.0/3.0) - pow(muonFactor, 2.0/3.0)  - B)

            return out

        except IncorrectSymmetryEnergyModelError:
            print("Incorrect value of the symmetry energy model!")
            print()


    # Determines the value of the electron factor (x_e = rhoo_e / rhoo_B) equation 
    # Inputs:
    #   rho: mass density (g/cm^3)
    #   factors: electron factor
    def protonFactorMuonless(self, factors, rho):
        try:
            [electronFactor] = factors

            # Symmetry energy
            if self.flagSymmetryEnergyModel == 1:
                SS = self.S + self.L / 3.0 * (rho - cgs.rhoS) / cgs.rhoS
            elif self.flagSymmetryEnergyModel == 2:
                G = self.L / (3.0 * self.S)
                SS = self.S * pow(rho / cgs.rhoS, G)
            else:
                raise IncorrectSymmetryEnergyModelError

            # Electron factor equation
            out = []  
            A = cgs.hbar * cgs.c * pow(3.0 * pi**2 * rho / cgs.mB, 1.0/3.0) / (8.0 * SS)  

            out.append(electronFactor - 0.5 + A * pow(electronFactor, 1.0/3.0) )

            return out

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
            [xe] = fsolve(self.protonFactorMuonless, 5.0e-2, args = rho)

            # NB Muons do not exist if xe <= B^(3/2)
            B = (cgs.mmu * cgs.c / cgs.hbar)**2.0 * pow(3.0 * pi**2.0 * rho / cgs.mB, -2.0/3.0)

            # Muons are present
            #   NB The if statement works at least with the current symmetry energy models!
            if xe > pow(B, 1.5) or isnan(xe):
                [[xe, xm],info,flag,mesg] = fsolve(self.protonFactorMuon, [5.0e-2, 0.0], args = rho, full_output=1) #XXX TESTAA, tarviiko full_output?, eli toimiiko aina => kaatuuko? (kato muos edellinen fsolve)

            else: # Without muons
                xm = 0.0

        # Beta eq. with only electrons
        else:
            xm = 0.0;

            [xe] = fsolve(self.protonFactorMuonless, 5.0e-2, args = rho)

        return [xe, xm]


    #XXX selitys
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
                coeff = 3.0 * pi**2 * nB
                ch = cgs.c * cgs.hbar

                # Proton correctin
                pressureProton = 4.0 * DS * x * (x - 1.0) * nB

                # Electron correction
                pressureElectron = 0.25 * xe * ch * pow(coeff * xe, 1.0/3.0) * nB

            else: # Non-beta eq.
                pressureProton = 0.0
                pressureElectron = 0.0


            if xm > 0.0: # w/ muons
                # Muon chemical potential
                chemicalPotentialMuon = sqrt(cgs.mmu**2 * cgs.c**4 + ch**2 * pow(coeff * xm, 2.0/3.0))

                pressureMuon = ch**2 * pi * nB * pow(pi * (xm * nB)**2 / 3.0, 1.0/3.0) / chemicalPotentialMuon
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
                coeff = 3.0 * pi**2 * nB
                c2Inv = 1.0 / cgs.c**2

                # Proton correction
                energyDensityProton = 4.0 * SS * x * (x - 1.0) * nB * c2Inv

                # Electron correction
                energyDensityElectron = 0.75 * xe * cgs.hbar * cgs.c * pow(coeff * xe, 1.0/3.0) * nB * c2Inv

            else: # Non-beta eq.
                energyDensityProton = 0.0
                energyDensityElectron = 0.0
           

            if xm > 0.0: # w/ muons
                # Muon chemical potential diveded by c^2
                chemicalPotentialMuon = sqrt(cgs.mmu**2 + cgs.hbar**2 * c2Inv * pow(coeff * xm, 2.0/3.0))

                energyDensityMuon = chemicalPotentialMuon * nB
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



#Combines EoS parts together, SPECIAL CASE
class combiningEos:
    
    def __init__(self, pieces, trans):
        #  prev_trope: previous monotrope which is not icnluded in the tropes (optional)
        print "AA"
        #ordered list of monotropes (from low-density to high-density)
        self.pieces = pieces
        print "BB"
        #ordered list of matching/transition points, mass density (g/cm^3), between monotropes
        self.transitions = trans
        print "CC"
        #transition/matching pressures (Ba) and energy densities (g/cm^3)
        self.prs  = []
        self.eds  = []
        print "DD"
        #prev_piece = None

        for (piece, transition) in zip(self.pieces, self.transitions):
            print "AAA", transition
            pr = piece.pressure(transition)
            print "BBB", pr
            ed = piece.edens_inv(pr)
            print "CCC", ed
            self.prs.append( pr )
            self.eds.append( ed )
            print "DDD"
            prev_piece = piece
            print "EEE"

        print "EE"

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
