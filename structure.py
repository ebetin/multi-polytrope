import numpy as np
from scipy.optimize import fsolve

# Units
import units as cgs

# EoSs models
from polytropes import monotrope
from polytropes import polytrope
from polytropes import doubleMonotrope
from polytropes import combiningEos
from polytropes import cEFT
from polytropes import cEFT_r4
from crust import BPS_crust
from pQCD import qcd
from c2Interpolation import matchC2AGKNV
from c2Interpolation import c2AGKNV

# pQCD related functions
from pQCD import matchPolytopesWithLimits 
from pQCD import pQCD
from pQCD import eQCD
from pQCD import nQCD

# TOV solver
from tov import tov

# Tests
from tests import causalityPolytropes
from tests import hydrodynamicalStabilityPolytropes
from tests import causalityPerturbativeQCD
from tests import positiveLatentHeat
from tests import causalityDoubleMonotrope

######################################################
# Constants
approx_rhoHigh = 12.0
rhoS_inv = 1.0 / cgs.rhoS
confacinv = 1000.0 / cgs.GeVfm_per_dynecm
confac = cgs.GeVfm_per_dynecm * 0.001
p_cc_xtol = 1.e-4
density_min = 0.#6.38371e-12 * cgs.mB * 1.e39 # TODO

# BPS crust
crustEoS_BPS = BPS_crust()

######################################################

class structurePolytrope:

    def __init__(self, gammasKnown, transitions, lowDensity, QCD):
        # Equation of state of the crust
        crustEoS = crustEoS_BPS

        # QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
        a = lowDensity[0]
        alpha = lowDensity[1]
        b = lowDensity[2]
        beta = lowDensity[3]
        S = cgs.Enuc + a + b 
        L = 3.0 * (a * alpha + b * beta)

        # Low-density dominated monotrope
        mAlpha = monotrope(a * alpha / (cgs.mB * cgs.rhoS**alpha), alpha + 1.0)
        # High-density dominated monotrope
        mBeta = monotrope(b * beta / (cgs.mB * cgs.rhoS**beta), beta + 1.0)

        # Transition continuity constants (unitless)
        mAlpha.a = -0.5
        mBeta.a = -0.5

        # Turn monotrope class objects into polytrope ones
        tropesAlpha = [mAlpha]
        tropesBeta = [mBeta]

        # Form the bimonotropic EoS
        gandolfiEoS = doubleMonotrope(tropesAlpha + tropesBeta, S, L, flagMuon=False, flagSymmetryEnergyModel=2, flagBetaEQ = True)

        # Causality test
        testCausalityGandolfi = causalityDoubleMonotrope(gandolfiEoS, transitions[0])

        testHydro = False
        self.gammasSolved = None

        if testCausalityGandolfi:
            # Pressure (Ba) and energy density (g/cm^3) at the end of the QMC EoS
            gandolfiPressureHigh = gandolfiEoS.pressure(transitions[0])
            gandolfiEnergyDensityHigh = gandolfiEoS.edens(transitions[0])
            gandolfiMatchingHigh = [gandolfiPressureHigh, gandolfiEnergyDensityHigh]

            # Perturbative QCD EoS
            muQCD = QCD[0]
            X = QCD[1]
            qcdEoS = qcd(X)

            # Pressure and energy density at the beginning of the pQCD EoS
            qcdPressureLow = pQCD(muQCD, X)
            qcdEnergyDensityLow = eQCD(muQCD, X)
            qcdMathing = [qcdPressureLow, qcdEnergyDensityLow * cgs.c**2]


            # Transition (matching) densities of the polytrypic EoS 
            transitionsPoly = transitions[:] # mass density
            transitionsSaturation = [x * rhoS_inv for x in transitionsPoly] # number density
            transitionsSaturation.append(nQCD(muQCD, X) * cgs.mB * rhoS_inv )

            # Determine polytropic exponents
            polyConditions = matchPolytopesWithLimits(gandolfiMatchingHigh, qcdMathing, transitionsSaturation, gammasKnown)

            gammasAll = polyConditions.GammaValues()
            self.gammasSolved = polyConditions.gammasSolved

        
            # Check that the polytropic EoS is hydrodynamically stable, ie. all polytropic exponents are non-negative
            testHydro = hydrodynamicalStabilityPolytropes(gammasAll)

        testCausality = True
        self.speed2max = 0.0

        if testHydro:
            # Check that the polytropi EoS is also causal
            testCausality, self.speed2max = causalityPolytropes(gammasAll, transitionsSaturation, gandolfiMatchingHigh)

        self.tropes = None
        self.trans  = None
        self.eos = None
        self.realistic = False

        if gammasAll is not None and testHydro and testCausality:
            # Polytropic constants
            Ks = [ceftPressureHigh * transitionsPoly[0]**(-gammasAll[0])]

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
            try:
                self.tropes[0].a = ( ceftEnergyDensityHigh - ceftPressureHigh / (cgs.c**2 * (gammasAll[0] - 1.0)) ) / transitions[0] - 1.0
            except:
                self.tropes[0].a = ( ceftEnergyDensityHigh - ceftPressureHigh / cgs.c**2 * log(transitions[0] / cgs.mB) ) / transitions[0] - 1.0

            # Create polytropic EoS
            polytropicEoS = polytrope( self.tropes, self.trans )

            # Transition between the crust and the core
            # This has been approximated so that rho_crust(p_cc) = rho_ceft(p_cc)
            def rho_diff(p):
                global rho_cc_tmp
                pp = p[0] * confac
                rho_cc_tmp = crustEoS.rho(pp)

                return rho_cc_tmp / ceftEoS.rho(pp) - 1.0

            try:
                p_cc = fsolve(rho_diff, 0.4, xtol=p_cc_xtol) * confac
                self.rho_cc = crustEoS.rho(p_cc)[0]

                # Combining EoSs
                combinedEoS = [crustEoS, ceftEoS, polytropicEoS, qcdEoS]
                transitionPieces = [density_min, self.rho_cc, transitions[0], rho_qcd]

                self.eos = combiningEos(combinedEoS, transitionPieces)


                # Is the pQCD EoS causal?
                test3, speed2pQCD = causalityPerturbativeQCD(qcdEoS, muQCD)

                if self.speed2max < speed2pQCD and not testHydro:
                    self.speed2max = speed2pQCD

                # Is the latent heat between EoS pieces positive?
                test4 = positiveLatentHeat(combinedEoS, transitionPieces)

                if test3 and test4:
                    self.realistic = True
            except:
                pass


    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        t = tov(self.eos, rhocs)

        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
            self.TDlist = np.zeros(len(self.mass))
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD, self.TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, self.TD2)
        else:
            self.mass, self.rad, self.rho, self.TDlist, self.TD2 = t.massRadiusTD(l, mRef2 = m2)
            self.TD = 1.0e10
            self.TDtilde = 1.0e10
        
        if len(self.mass) == 0:
            self.maxmass = 0.0
            self.maxmassrho = 0.0
            self.maxmassrad = 0.0
        else:
            self.indexM = np.argmax( self.mass )
            self.maxmass = self.mass[self.indexM]
            self.maxmassrho = self.rho[self.indexM]
            self.maxmassrad = self.rad[self.indexM]



    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.rad)

    # interpolate TD given a mass
    # note: structure must be solved beforehand
    def TD_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.TDlist)


class structurePolytropeWithCEFT:

    def __init__(self, gammasKnown, transitions, lowDensity, QCD, CEFT_model = 'HLPS'):
        # Equation of state of the crust
        crustEoS = crustEoS_BPS

        if CEFT_model == 'HLPS' or CEFT_model == 'HLPS3':
            # cEFT EoS parameters
            gamma = lowDensity[0]
            alphaL = lowDensity[1]
            etaL = lowDensity[2]

            # Form the bimonotropic EoS
            ceftEoS = cEFT(lowDensity)

            # Causality test
            testCausalityCEFT = True

        elif CEFT_model == 'HLPS+':
            # cEFT EoS parameters
            gamma = lowDensity[0]
            alphaL = lowDensity[1]
            etaL = lowDensity[2]
            zetaL = lowDensity[3]
            rho0 = lowDensity[4]

            # Form the bimonotropic EoS
            ceftEoS = cEFT_r4(lowDensity)

            # Causality test
            testCausalityCEFT = ceftEoS.realistic

        testHydro = False
        gammasAll = None

        if testCausalityCEFT:
            # Pressure (Ba) and energy density (g/cm^3) at the end of the QMC EoS
            ceftPressureHigh = ceftEoS.pressure(transitions[0])
            ceftEnergyDensityHigh = ceftEoS.edens(transitions[0])
            ceftMatchingHigh = [ceftPressureHigh, ceftEnergyDensityHigh]

            # Perturbative QCD EoS
            muQCD = QCD[0]
            X = QCD[1]
            qcdEoS = qcd(X)

            # Pressure and energy density at the beginning of the pQCD EoS
            qcdPressureLow = pQCD(muQCD, X)
            qcdEnergyDensityLow = eQCD(muQCD, X)
            qcdMathing = [qcdPressureLow, qcdEnergyDensityLow * cgs.c**2]


            # Transition (matching) densities of the polytrypic EoS
            rho_qcd = nQCD(muQCD, X) * cgs.mB
            transitionsPoly = transitions[:] # mass density
            transitionsSaturation = [x * rhoS_inv for x in transitionsPoly] # number density
            transitionsSaturation.append(rho_qcd * rhoS_inv )

            # Determine polytropic exponents
            polyConditions = matchPolytopesWithLimits(ceftMatchingHigh, qcdMathing, transitionsSaturation, gammasKnown)
            gammasAll = polyConditions.GammaValues()
            self.gammasSolved = polyConditions.gammasSolved

            # Check that the polytropic EoS is hydrodynamically stable, ie. all polytropic exponents are non-negative
            testHydro = hydrodynamicalStabilityPolytropes(gammasAll)

        testCausality = True
        self.speed2max = 0.0

        if testHydro:
            # Check that the polytropi EoS is also causal
            testCausality, self.speed2max = causalityPolytropes(gammasAll, transitionsSaturation, ceftMatchingHigh)

        self.tropes = None
        self.trans  = None
        self.eos = None
        self.realistic = False

        if gammasAll is not None and testHydro and testCausality:
            # Polytropic constants
            Ks = [ceftPressureHigh * transitionsPoly[0]**(-gammasAll[0])]

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
            try:
                self.tropes[0].a = ( ceftEnergyDensityHigh - ceftPressureHigh / (cgs.c**2 * (gammasAll[0] - 1.0)) ) / transitions[0] - 1.0
            except:
                self.tropes[0].a = ( ceftEnergyDensityHigh - ceftPressureHigh / cgs.c**2 * log(transitions[0] / cgs.mB) ) / transitions[0] - 1.0

            # Create polytropic EoS
            polytropicEoS = polytrope( self.tropes, self.trans )

            # Transition between the crust and the core
            # This has been approximated so that rho_crust(p_cc) = rho_ceft(p_cc)
            def rho_diff(p):
                global rho_cc_tmp
                pp = p[0] * confac
                rho_cc_tmp = crustEoS.rho(pp)

                return rho_cc_tmp / ceftEoS.rho(pp) - 1.0

            try:
                p_cc = fsolve(rho_diff, 0.4, xtol=p_cc_xtol) * confac
                self.rho_cc = crustEoS.rho(p_cc)[0]

                # Combining EoSs
                combinedEoS = [crustEoS, ceftEoS, polytropicEoS, qcdEoS]
                transitionPieces = [density_min, self.rho_cc, transitions[0], rho_qcd]

                self.eos = combiningEos(combinedEoS, transitionPieces)

                # Is the pQCD EoS causal?
                test3, speed2pQCD = causalityPerturbativeQCD(qcdEoS, muQCD)

                if self.speed2max < speed2pQCD and not testHydro:
                    self.speed2max = speed2pQCD

                # Is the latent heat between EoS pieces positive?
                test4 = positiveLatentHeat(combinedEoS, transitionPieces)

                if test3 and test4:
                    self.realistic = True
            except:
                pass


    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        t = tov(self.eos, rhocs)

        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
            self.TDlist = np.zeros(len(self.mass))
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD, self.TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, self.TD2)
        else:
            self.mass, self.rad, self.rho, self.TDlist, self.TD2 = t.massRadiusTD(l, mRef2 = m2)
            self.TD = 1.0e10
            self.TDtilde = 1.0e10
        
        if len(self.mass) == 0:
            self.maxmass = 0.0
            self.maxmassrho = 0.0
            self.maxmassrad = 0.0
        else:
            self.indexM = np.argmax( self.mass )
            self.maxmass = self.mass[self.indexM]
            self.maxmassrho = self.rho[self.indexM]
            self.maxmassrad = self.rad[self.indexM]


    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.rad)

    # interpolate TD given a mass
    # note: structure must be solved beforehand
    def TD_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.TDlist)


class structureC2AGKNV:

    def __init__(self, muDeltaKnown, c2Known, transitions, lowDensity, QCD, flag_muDelta = True):
        # Equation of state of the crust
        crustEoS = crustEoS_BPS

        # QMC EoS, see Gandolfi et al. (2012, arXiv:1101.1921) for details
        a = lowDensity[0]
        alpha = lowDensity[1]
        b = lowDensity[2]
        beta = lowDensity[3]
        S = cgs.Enuc + a + b 
        L = 3.0 * (a * alpha + b * beta)

        # Low-density dominated monotrope
        mAlpha = monotrope(a * alpha / (cgs.mB * cgs.rhoS**alpha), alpha + 1.0)
        # High-density dominated monotrope
        mBeta = monotrope(b * beta / (cgs.mB * cgs.rhoS**beta), beta + 1.0)

        # Transition continuity constants (unitless)
        mAlpha.a = -0.5
        mBeta.a = -0.5

        # Turn monotrope class objects into polytrope ones
        tropesAlpha = [mAlpha]
        tropesBeta = [mBeta]

        # Form the bimonotropic EoS
        gandolfiEoS = doubleMonotrope(tropesAlpha + tropesBeta, S, L, flagMuon=False, flagSymmetryEnergyModel=2, flagBetaEQ = True)

        # Causality test
        testCausalityGandolfi = causalityDoubleMonotrope(gandolfiEoS, transitions[1])

        muAll = None
        c2All = None
        testC2Ok = False
        self.muSolved = None
        self.c2Solved = None

        if testCausalityGandolfi:
            # Pressure (Ba), energy density (g/cm^3), mass density (g/cm^3), and
            # speed of sound squared (unitless) at the end of the QMC EoS
            gandolfiPressureHigh = gandolfiEoS.pressure(transitions[1])
            gandolfiEnergyDensityHigh = gandolfiEoS.edens(transitions[1])
            gandolfiDensityHigh = transitions[1]
            gandolfiSpeed2High = gandolfiEoS.speed2(gandolfiPressureHigh)
            gandolfiMatchingHigh = [gandolfiPressureHigh, gandolfiEnergyDensityHigh, gandolfiDensityHigh, gandolfiSpeed2High]

            # Determining matching chemical potentials
            if flag_muDelta:
                mu0 = cgs.mB * ( gandolfiEnergyDensityHigh * cgs.c**2 + gandolfiPressureHigh ) * 1.0e-9
                mu0 /= gandolfiDensityHigh * cgs.eV
                muKnown = [None] * len(muDeltaKnown)
                for i, mu in enumerate(muDeltaKnown):
                    if i == 0:
                        muKnown[0] = mu0 + mu
                    else:
                        muKnown[i] = muKnown[i-1] + mu
            else:
                muKnown = muDeltaKnown
            # Perturbative QCD EoS
            muQCD = QCD[0]
            X = QCD[1]
            qcdEoS = qcd(X)

            # Pressure, energy density, and speed of sound squared at the beginning of the pQCD EoS
            qcdPressureLow = pQCD(muQCD, X)
            qcdDensityLow = nQCD(muQCD, X) * cgs.mB
            qcdSpeed2Low = qcdEoS.speed2(qcdPressureLow)
            qcdMathing = [qcdPressureLow, qcdDensityLow, muQCD, qcdSpeed2Low]

            # Transition (matching) densities of the polytrypic EoS 
            transitionsPoly = transitions[1:] # mass density
            transitionsSaturation = [x * rhoS_inv for x in transitionsPoly] # number density
            transitionsSaturation.append(nQCD(muQCD, X) * cgs.mB * rhoS_inv )

            # Check that the last known chemical potential is small enough
            if muQCD > muKnown[-1]:
                # Determine unknown coefficients (chem.pot. and c2)
                c2Conditions = matchC2AGKNV( muKnown, c2Known, gandolfiMatchingHigh, qcdMathing )
            
                muAll, c2All = c2Conditions.coeffValues()
        
                # Checking that the gotten coefficients are suitable
                testC2Ok = c2Conditions.coeffValuesOkTest( muAll, c2All )

                self.muSolved = c2Conditions.muSolved
                self.c2Solved = c2Conditions.c2Solved

        self.eos = None
        self.realistic = False
        self.speed2max = 0.0

        if muAll is not None and c2All is not None and testC2Ok:
            # Create c2 EoS
            rho_qcd = nQCD(muQCD, X) * cgs.mB
            c2EoS = c2AGKNV( muAll, c2All, ceftMatchingHigh, approx = approximation, rhoHigh1 = approx_rhoHigh, rhoHigh2 = rho_qcd * rhoS_inv )
            if approximation:
                c2EoS.approximation()

            # Transition between the crust and the core
            # This has been approximated so that rho_crust(p_cc) = rho_ceft(p_cc)
            def rho_diff(p):
                global rho_cc_tmp
                pp = p[0] * confac
                rho_cc_tmp = crustEoS.rho(pp)

                return rho_cc_tmp / ceftEoS.rho(pp) - 1.0

            try:
                p_cc = fsolve(rho_diff, 0.4, xtol=p_cc_xtol) * confac
                self.rho_cc = crustEoS.rho(p_cc)[0]

                # Combining EoSs
                combinedEoS = [crustEoS, ceftEoS, c2EoS, qcdEoS]
                transitionPieces = [density_min, self.rho_cc, transitions[0], rho_qcd]
                self.eos = combiningEos(combinedEoS, transitionPieces)

                # Is the pQCD EoS causal?
                test2, speed2pQCD = causalityPerturbativeQCD(qcdEoS, muQCD)

                self.speed2max = max(c2Known)
                if self.speed2max < self.c2Solved:
                    self.speed2max = self.c2Solved
                if self.speed2max < speed2pQCD:
                    self.speed2max = speed2pQCD

                # Are all matching point continues (within numerical erros)?
                test3 = positiveLatentHeat(combinedEoS, transitionPieces)

                if test2 and test3:
                    self.realistic = True
            except:
                pass


    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        t = tov(self.eos, rhocs)

        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
            self.TDlist = np.zeros(len(self.mass))
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD, self.TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, self.TD2)
        else:
            self.mass, self.rad, self.rho, self.TDlist, self.TD2 = t.massRadiusTD(l, mRef2 = m2)
            self.TD = 1.0e10
            self.TDtilde = 1.0e10
        
        if len(self.mass) == 0:
            self.maxmass = 0.0
            self.maxmassrho = 0.0
            self.maxmassrad = 0.0
        else:
            self.indexM = np.argmax( self.mass )
            self.maxmass = self.mass[self.indexM]
            self.maxmassrho = self.rho[self.indexM]
            self.maxmassrad = self.rad[self.indexM]


    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.rad)

    # interpolate TD given a mass
    # note: structure must be solved beforehand
    def TD_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.TDlist)



class structureC2AGKNVwithCEFT:

    def __init__(self, muDeltaKnown, c2Known, transitions, lowDensity, QCD, approximation = False, CEFT_model = 'HLPS', flag_muDelta = True):
        # Equation of state of the crust
        crustEoS = crustEoS_BPS

        if CEFT_model == 'HLPS' or CEFT_model == 'HLPS3':
            # cEFT EoS parameters
            gamma = lowDensity[0]
            alphaL = lowDensity[1]
            etaL = lowDensity[2]

            # Form the bimonotropic EoS
            ceftEoS = cEFT(lowDensity)

            # Causality test
            testCausalityCEFT = True

        elif CEFT_model == 'HLPS+':
            # cEFT EoS parameters
            gamma = lowDensity[0]
            alphaL = lowDensity[1]
            etaL = lowDensity[2]
            zetaL = lowDensity[3]
            rho0 = lowDensity[4]

            # Form the bimonotropic EoS
            ceftEoS = cEFT_r4(lowDensity)

            # Causality test
            testCausalityCEFT = ceftEoS.realistic

        muAll = None
        c2All = None
        testC2Ok = False
        self.muSolved = None
        self.c2Solved = None

        if testCausalityCEFT:
            # Pressure (Ba) and energy density (g/cm^3) at the end of the QMC EoS
            ceftPressureHigh = ceftEoS.pressure(transitions[0])
            ceftEnergyDensityHigh = ceftEoS.edens(transitions[0])
            ceftDensityHigh = transitions[0]
            ceftC2High = ceftEoS.speed2_rho(transitions[0])
            ceftMatchingHigh = [ceftPressureHigh, ceftEnergyDensityHigh, ceftDensityHigh, ceftC2High]

            # Determining matching chemical potentials
            if flag_muDelta:
                mu0 = cgs.mB * ( ceftEnergyDensityHigh * cgs.c**2 + ceftPressureHigh ) * 1.0e-9
                mu0 /= ceftDensityHigh * cgs.eV
                muKnown = [None] * len(muDeltaKnown)
                for i, mu in enumerate(muDeltaKnown):
                    if i == 0:
                        muKnown[0] = mu0 + mu
                    else:
                        muKnown[i] = muKnown[i-1] + mu
            else:
                muKnown = muDeltaKnown

            # Perturbative QCD EoS
            muQCD, X = QCD
            qcdEoS = qcd(X)

            # Pressure, energy density, and speed of sound squared at the beginning of the pQCD EoS
            qcdPressureLow = pQCD(muQCD, X)
            qcdDensityLow = nQCD(muQCD, X) * cgs.mB
            qcdSpeed2Low = qcdEoS.speed2(qcdPressureLow)
            qcdMathing = [qcdPressureLow, qcdDensityLow, muQCD, qcdSpeed2Low]

            # Check that the last known chemical potential is small enough
            if muQCD > muKnown[-1]:
                # Determine unknown coefficients (chem.pot. and c2)
                c2Conditions = matchC2AGKNV( muKnown, c2Known, ceftMatchingHigh, qcdMathing )

                muAll, c2All = c2Conditions.coeffValues()

                # Checking that the gotten coefficients are suitable
                testC2Ok = c2Conditions.coeffValuesOkTest( muAll, c2All )

                self.muSolved = c2Conditions.muSolved
                self.c2Solved = c2Conditions.c2Solved

        self.eos = None
        self.realistic = False
        self.speed2max = 0.0

        if muAll is not None and c2All is not None and testC2Ok:
            # Create c2 EoS
            rho_qcd = nQCD(muQCD, X) * cgs.mB
            c2EoS = c2AGKNV( muAll, c2All, ceftMatchingHigh, approx = approximation, rhoHigh1 = approx_rhoHigh, rhoHigh2 = rho_qcd * rhoS_inv )
            if approximation:
                c2EoS.approximation()

            # Transition between the crust and the core
            # This has been approximated so that rho_crust(p_cc) = rho_ceft(p_cc)
            def rho_diff(p):
                global rho_cc_tmp
                pp = p[0] * confac
                rho_cc_tmp = crustEoS.rho(pp)

                return rho_cc_tmp / ceftEoS.rho(pp) - 1.0

            try:
                p_cc = fsolve(rho_diff, 0.4, xtol=p_cc_xtol) * confac
                self.rho_cc = crustEoS.rho(p_cc)[0]

                # Combining EoSs
                combinedEoS = [crustEoS, ceftEoS, c2EoS, qcdEoS]
                transitionPieces = [density_min, self.rho_cc, transitions[0], rho_qcd]
                self.eos = combiningEos(combinedEoS, transitionPieces)

                # Is the pQCD EoS causal?
                test2, speed2pQCD = causalityPerturbativeQCD(qcdEoS, muQCD)

                self.speed2max = max(c2Known)
                if self.speed2max < self.c2Solved:
                    self.speed2max = self.c2Solved
                if self.speed2max < speed2pQCD:
                    self.speed2max = speed2pQCD

                # Are all matching point continues (within numerical erros)?
                test3 = positiveLatentHeat(combinedEoS, transitionPieces)

                if test2 and test3:
                    self.realistic = True
            except:
                pass



    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        t = tov(self.eos, rhocs)
        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
            self.TDlist = np.zeros(len(self.mass))
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TD2 = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TDlist, self.TD, self.TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, self.TD2)
        else:
            self.mass, self.rad, self.rho, self.TDlist, self.TD2 = t.massRadiusTD(l, mRef2 = m2)
            self.TD = 1.0e10
            self.TDtilde = 1.0e10

        if len(self.mass) == 0:
            self.maxmass = 0.0
            self.maxmassrho = 0.0
            self.maxmassrad = 0.0
        else:
            self.indexM = np.argmax( self.mass )
            self.maxmass = self.mass[self.indexM]
            self.maxmassrho = self.rho[self.indexM]
            self.maxmassrad = self.rad[self.indexM]


    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass > self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.rad)

    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def TD_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        return np.interp(mass, self.mass, self.TDlist)


# test TOV solver with various setups
def test_tov():

    import matplotlib
    import matplotlib.pyplot as plt



    eos_Ntrope = 4 #polytrope order
    #cube = [10.35546398,  0.46968896,  6.90197682,  1.8698988,   3.63685125,  9.24444854, 0.18240452, 20.5893892 ]
    #cube = [13.72713733,  0.41359602,  1.75415611,  2.54708464,  2.9140442,   3.05818737, 16.93841636, 18.90894771]
    cube = [12.5,  0.46307764,  3.06898968,  2.69243094,  2.98691193,  2.75428241,
  3.6493058,   3.87470224, 26.40907385,  1.31246422,  1.49127976]


    ##################################################
    # build EoS
    ci = 5
    gammas = []  
    for itrope in range(eos_Ntrope-2):
        gammas.append(cube[ci])
        ci += 1

    trans  = [0.1 * cgs.rhoS, 1.1 * cgs.rhoS] #starting point
    for itrope in range(eos_Ntrope-1):
        trans.append(trans[-1] + cgs.rhoS * cube[ci]) 
        ci += 1

    a     = cube[0] * 1.0e6 * cgs.eV     # (erg)
    alpha = cube[1]                      # untiless
    b     = cube[2] * 1.0e6 * cgs.eV     # (erg)
    beta  = cube[3]                      # unitless
    S     = cgs.Enuc + a + b      # (erg)
    L     = 3.0 * (a * alpha + b * beta) # (erg)
    lowDensity = [a, alpha, b, beta]

    X = cube[4]
    muQCD = 2.6 # Transition (matching) chemical potential where pQCD starts (GeV)
    highDensity = [muQCD, X]

    ################################################## 
    # solve
    struc = structurePolytrope(gammas, trans, lowDensity, highDensity)
    print(struc.realistic)
    struc.tov()
    #struc.tov(m1 = 1.4 * cgs.Msun) #with love numbers



    #visualize

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('axes', labelsize=7)
    

    fig = plt.figure(figsize=(3.54, 2.19)) #single column fig
    #fig = plt.figure(figsize=(7.48, 4.0))  #two column figure
    gs = plt.GridSpec(1, 1)

    ax = plt.subplot(gs[0, 0])
    ax.minorticks_on()
    ax.set_xlim(9.0, 16.0)
    ax.set_ylim(0.0, 3.0)

    ax.set_xlabel(r'Radius $R$ (km)')
    ax.set_ylabel(r'Mass $M$ (M$_{\odot}$)')


    #mass, rad, rho = struc.t.mass_radius()
    print(struc.mass)
    print(struc.rad)
    ax.plot(struc.rad, struc.mass)

    mass = np.linspace(0.5, 1.0, 10)
    rad  = [struc.radius_at(m) for m in mass]
    ax.plot(rad, mass, "r-")

    print(mass)
    print(rad)


    plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    plt.savefig('mr_test.pdf')



if __name__ == "__main__":
    #main(sys.argv)
    #plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    #plt.savefig('mr.pdf')

    test_tov()
