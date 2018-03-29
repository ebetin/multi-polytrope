import numpy as np
from math import pi


import units as cgs
from polytropes import monotrope, polytrope, doubleMonotrope, combiningEos
from crust import SLyCrust
from tov import tov
from pQCD import qcd, matchPolytopesWithLimits, pQCD, eQCD, nQCD
from tests import causalityPolytropes, hydrodynamicalStabilityPolytropes, causalityPerturbativeQCD, positiveLatentHeat


class structure:


    #def __init__(self, gammas, Ks, transitions):

        # Create polytropic presentation 
        #assert len(gammas) == len(Ks) == len(transitions)
        #self.tropes = []
        #self.trans  = []
        #for i in range(len(gammas)):
            #self.tropes.append( monotrope(Ks[i], gammas[i]) )
            #self.trans.append( transitions[i] )

        #dense_eos = polytrope( self.tropes, self.trans )
        #self.eos = glue_crust_and_core( SLyCrust, dense_eos )

    def __init__(self, gammasKnown, transitions, lowDensity, QCD):
        #print gammasKnown, transitions, lowDensity, QCD, "ALKU" #XXX TEST
        # Equation of state of the crust
        crustEoS = SLyCrust


        # Outer core EoS
        a = lowDensity[0]
        alpha = lowDensity[1]
        b = lowDensity[2]
        beta = lowDensity[3]
        S = 16.0e6 * cgs.eV + a + b 
        L = 3.0 * (a * alpha + b * beta)
        rhooCG = transitions[0]


        mAlpha = monotrope(a * alpha / (cgs.mB * cgs.rhoS**alpha), alpha+1.0)
        mBeta = monotrope(b * beta / (cgs.mB * cgs.rhoS**beta), beta+1.0)

        mAlpha.a = -0.5
        mBeta.a = -0.5

        tropesAlpha = [mAlpha]
        tropesBeta = [mBeta]

        gandolfiEoS = doubleMonotrope(tropesAlpha + tropesBeta, rhooCG, S, L, cgs.rhoS, cgs.mB, flagMuon=False, flagSymmetryEnergyModel=2, flagBetaEQ = False)

        gandolfiPressureHigh = gandolfiEoS.pressure(transitions[1])
        gandolfiEnergyDensityHigh = gandolfiEoS.edens_inv(gandolfiPressureHigh)
        gandolfiMatchingHigh = [gandolfiPressureHigh, gandolfiEnergyDensityHigh]


        # Perturbative QCD EoS
        muQCD = QCD[0]
        X = QCD[1]
        qcdEoS = qcd(X)

        qcdPressureLow = pQCD(muQCD, X)
        qcdEnergyDensityLow = eQCD(muQCD, X)
        qcdMathing = [qcdPressureLow, qcdEnergyDensityLow * cgs.c**2]


        # Polytropic parameters
        transitionsPoly = transitions[1:]
        transitionsNumberDensity = [x / cgs.mB for x in transitionsPoly]
        transitionsNumberDensity.append(nQCD(muQCD, X))

        polyConditions = matchPolytopesWithLimits(gandolfiMatchingHigh, qcdMathing, transitionsNumberDensity, gammasKnown)


        gammasAll = polyConditions.GammaValues()
        test2 = hydrodynamicalStabilityPolytropes(gammasAll)

        if gammasAll == None or test2 == False or gammasAll[0] > 10.0: # XXX ONGELMA > 15.0
            self.tropes = None
            self.trans  = None
            self.eos = None
            self.realistic = False

        else:
            Ks = [gandolfiPressureHigh * transitionsPoly[0]**(-gammasAll[0])]

            for i in range(1, len(gammasAll)):
                Ks.append( Ks[i-1] * transitionsPoly[i]**(gammasAll[i-1]-gammasAll[i]) ) #XXX ONGELMA

            # Create polytropic presentation 
            assert len(gammasAll) == len(Ks) == len(transitionsPoly)
            self.tropes = []
            self.trans  = []

            for i in range(len(gammasAll)):
                self.tropes.append( monotrope(Ks[i], gammasAll[i]) )
                self.trans.append( transitionsPoly[i] )

            polytropicEoS = polytrope( self.tropes, self.trans )

            # Combining EoSs
            combinedEoS = [crustEoS, gandolfiEoS, polytropicEoS, qcdEoS]
            transitionPieces = [0.0] + transitions[:2] + [nQCD(muQCD, X) * cgs.mB]

            self.eos = combiningEos(combinedEoS, transitionPieces)

            # Is the obtained EoS realistic?
            test1 = causalityPolytropes(self.tropes, self.trans, qcdMathing)
            #test2 = hydrodynamicalStabilityPolytropes(self.tropes)
            test3 = causalityPerturbativeQCD(muQCD)
            test4 = positiveLatentHeat(combinedEoS, transitionPieces)

            if test1 == False or test3 == False or test4 == False or gammasAll == None:
                self.realistic = False
            else:
                self.realistic = True


    #solve TOV equations
    def tov(self):
        t = tov(self.eos)
        self.mass, self.rad, self.rho = t.mass_radius()
        self.maxmass = np.max( self.mass )








