import numpy as np
from math import pi


import units as cgs
from polytropes import monotrope, polytrope, doubleMonotrope, combiningEos
from crust import SLyCrust
from tov import tov
from pQCD import qcd, matchPolytopesWithLimits, pQCD, eQCD, nQCD
from tests import causalityPolytropes, hydrodynamicalStabilityPolytropes, causalityPerturbativeQCD, positiveLatentHeat


class structure:

    def __init__(self, gammasKnown, transitions, lowDensity, QCD):
        # Equation of state of the crust
        crustEoS = SLyCrust


        # QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
        a = lowDensity[0]
        alpha = lowDensity[1]
        b = lowDensity[2]
        beta = lowDensity[3]
        S = 16.0e6 * cgs.eV + a + b 
        L = 3.0 * (a * alpha + b * beta)

        # Transtion (matching) point between crust and QMC EoSs
        rhooCG = transitions[0]

        # Low-density dominated monotrope
        mAlpha = monotrope(a * alpha / (cgs.mB * cgs.rhoS**alpha), alpha+1.0)
        # High-density dominated monotrope
        mBeta = monotrope(b * beta / (cgs.mB * cgs.rhoS**beta), beta+1.0)

        # Transition continuity constants (unitless)
        mAlpha.a = -0.5
        mBeta.a = -0.5

        # Turn monotrope class objects into polytrope ones
        tropesAlpha = [mAlpha]
        tropesBeta = [mBeta]

        # Form the bimonotropic EoS
        gandolfiEoS = doubleMonotrope(tropesAlpha + tropesBeta, rhooCG, S, L, flagMuon=False, flagSymmetryEnergyModel=2, flagBetaEQ = False)

        # Pressure (Ba) and energy density (g/cm^3) at the end of the QMC EoS
        gandolfiPressureHigh = gandolfiEoS.pressure(transitions[1])
        gandolfiEnergyDensityHigh = gandolfiEoS.edens_inv(gandolfiPressureHigh)
        gandolfiMatchingHigh = [gandolfiPressureHigh, gandolfiEnergyDensityHigh]


        # Perturbative QCD EoS
        muQCD = QCD[0]
        X = QCD[1]
        qcdEoS = qcd(X)

        # Pressure and energy density at the beginning of the pQCD EoS
        qcdPressureLow = pQCD(muQCD, X)
        qcdEnergyDensityLow = eQCD(muQCD, X)
        qcdMathing = [qcdPressureLow, qcdEnergyDensityLow * cgs.c**2]


        # Determine polytropic exponents
        transitionsPoly = transitions[1:]
        transitionsSaturation = [x / cgs.rhoS for x in transitionsPoly]
        transitionsSaturation.append(nQCD(muQCD, X) / 0.16e39)

        polyConditions = matchPolytopesWithLimits(gandolfiMatchingHigh, qcdMathing, transitionsSaturation, gammasKnown)
        
        # Determined exponents
        gammasAll = polyConditions.GammaValues()

        
        # Check that the polytropic EoS is hydrodynamically stable, ie. all polytropic exponents are non-negative
        testHydro = hydrodynamicalStabilityPolytropes(gammasAll)

        if testHydro:
            # Check that the polytropi EoS is also causal
            testCausality = causalityPolytropes(gammasAll, transitionsSaturation, gandolfiMatchingHigh)

        else:
            testCausality = True
        #print "EI", gammasAll#XXX
        # Do not proceed if tested failed
        if gammasAll == None or not testHydro or not testCausality:
            self.tropes = None
            self.trans  = None
            self.eos = None
            self.realistic = False

        else:
            # Polytropic constants
            Ks = [gandolfiPressureHigh * transitionsPoly[0]**(-gammasAll[0])]

            for i in range(1, len(gammasAll)):
                Ks.append( Ks[i-1] * transitionsPoly[i]**(gammasAll[i-1]-gammasAll[i]) )

            # Create polytropic presentation 
            assert len(gammasAll) == len(Ks) == len(transitionsPoly)
            self.tropes = []
            self.trans  = []

            for i in range(len(gammasAll)):
                self.tropes.append( monotrope(Ks[i], gammasAll[i]) )
                self.trans.append( transitionsPoly[i] )

            # Fix the first transition continuity constant (unitless)
            self.tropes[0].a = ( gandolfiEnergyDensityHigh - gandolfiPressureHigh / (cgs.c**2 * (gammasAll[0] - 1.0)) ) / transitions[1] - 1.0

            # Create polytropic EoS
            polytropicEoS = polytrope( self.tropes, self.trans )

            # Combining EoSs
            combinedEoS = [crustEoS, gandolfiEoS, polytropicEoS, qcdEoS]
            transitionPieces = [0.0] + transitions[:2] + [nQCD(muQCD, X) * cgs.mB]

            self.eos = combiningEos(combinedEoS, transitionPieces)


            # Is the pQCD EoS causal?
            test3 = causalityPerturbativeQCD(muQCD)

            # Is the latent heat between EoS pieces positive?
            test4 = positiveLatentHeat(combinedEoS, transitionPieces)

            if not test3 or not test4:
                self.realistic = False
            else:
                self.realistic = True


    #solve TOV equations
    def tov(self):
        t = tov(self.eos)
        self.mass, self.rad, self.rho = t.mass_radius()
        self.maxmass = np.max( self.mass )








