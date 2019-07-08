import numpy as np
from math import pi
from scipy import interpolate


import units as cgs
from polytropes import monotrope, polytrope, doubleMonotrope, combiningEos
from crust import SLyCrust
from tov import tov
from pQCD import qcd, matchPolytopesWithLimits, pQCD, eQCD, nQCD
from tests import causalityPolytropes, hydrodynamicalStabilityPolytropes, causalityPerturbativeQCD, positiveLatentHeat, causalityDoubleMonotrope
from c2Interpolation import matchC2AGKNV, c2AGKNV


class structurePolytrope:

    def __init__(self, gammasKnown, transitions, lowDensity, QCD):
        # Equation of state of the crust
        crustEoS = SLyCrust

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

        if testCausalityGandolfi:
            # Pressure (Ba) and energy density (g/cm^3) at the end of the QMC EoS
            gandolfiPressureHigh = gandolfiEoS.pressure(transitions[1])
            gandolfiEnergyDensityHigh = gandolfiEoS.edens(transitions[1])
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
            transitionsPoly = transitions[1:] # mass density
            transitionsSaturation = [x / cgs.rhoS for x in transitionsPoly] # number density
            transitionsSaturation.append(nQCD(muQCD, X) * (cgs.mB / cgs.rhoS) )

            # Determine polytropic exponents
            polyConditions = matchPolytopesWithLimits(gandolfiMatchingHigh, qcdMathing, transitionsSaturation, gammasKnown)

            gammasAll = polyConditions.GammaValues()

        
            # Check that the polytropic EoS is hydrodynamically stable, ie. all polytropic exponents are non-negative
            testHydro = hydrodynamicalStabilityPolytropes(gammasAll)

        else:
            testHydro = False


        if testHydro:
            # Check that the polytropi EoS is also causal
            testCausality = causalityPolytropes(gammasAll, transitionsSaturation, gandolfiMatchingHigh)

        else:
            testCausality = True

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
            if 1.0 - cgs.epsilonGamma < gammasAll[0] < 1.0 + cgs.epsilonGamma:
                self.tropes[0].a = ( gandolfiEnergyDensityHigh - gandolfiPressureHigh / cgs.c**2 * log(transitions[1] / cgs.mB) ) / transitions[1] - 1.0
            else:
                self.tropes[0].a = ( gandolfiEnergyDensityHigh - gandolfiPressureHigh / (cgs.c**2 * (gammasAll[0] - 1.0)) ) / transitions[1] - 1.0

            # Create polytropic EoS
            polytropicEoS = polytrope( self.tropes, self.trans )

            # Combining EoSs
            combinedEoS = [crustEoS, gandolfiEoS, polytropicEoS, qcdEoS]
            transitionPieces = [0.0] + transitions[:2] + [nQCD(muQCD, X) * cgs.mB]

            self.eos = combiningEos(combinedEoS, transitionPieces)


            # Is the pQCD EoS causal?
            test3 = causalityPerturbativeQCD(qcdEoS, muQCD)

            # Is the latent heat between EoS pieces positive?
            test4 = positiveLatentHeat(combinedEoS, transitionPieces)

            if not test3 or not test4:
                self.realistic = False
            else:
                self.realistic = True


    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0):
        t = tov(self.eos)

        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TD, TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, TD2)
        else:
            self.mass, self.rad, self.rho, self.TD = t.massRadiusTD(l, mRef2 = m2)
            self.TDtilde = 1.0e10
        
        self.maxmass = np.max( self.mass )


    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        try:
            intp = interpolate.interp1d(self.mass, self.rad, kind='cubic')
            return intp(mass)
        except:
            # linear interpolant (faster, backup)
            return np.interp(mass, self.mass, self.rad)

class structureC2AGKNV:

    def __init__(self, muDeltaKnown, c2Known, transitions, lowDensity, QCD):
        # Equation of state of the crust
        crustEoS = SLyCrust

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

        if testCausalityGandolfi:
            # Pressure (Ba), energy density (g/cm^3), mass density (g/cm^3), and
            # speed of sound squared (unitless) at the end of the QMC EoS
            gandolfiPressureHigh = gandolfiEoS.pressure(transitions[1])
            gandolfiEnergyDensityHigh = gandolfiEoS.edens(transitions[1])
            gandolfiDensityHigh = transitions[1]
            gandolfiSpeed2High = gandolfiEoS.speed2(gandolfiPressureHigh)
            gandolfiMatchingHigh = [gandolfiPressureHigh, gandolfiEnergyDensityHigh, gandolfiDensityHigh, gandolfiSpeed2High]

            # Determining matching chemical potentials
            mu0 = cgs.mB * ( gandolfiEnergyDensityHigh * cgs.c**2 + gandolfiPressureHigh )      
            mu0 = mu0 / ( gandolfiDensityHigh * cgs.eV ) * 1.0e-9
            muKnown = []
            for mu in muDeltaKnown:
                if len( muKnown ) == 0:
                    muKnown.append( mu0 + mu )
                else:
                    muKnown.append( muKnown[-1] + mu )

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
            transitionsSaturation = [x / cgs.rhoS for x in transitionsPoly] # number density
            transitionsSaturation.append(nQCD(muQCD, X) * (cgs.mB / cgs.rhoS) )

            # Check that the last known chemical potential is small enough
            if muQCD > muKnown[-1]:
                # Determine unknown coefficients (chem.pot. and c2)
                c2Conditions = matchC2AGKNV( muKnown, c2Known, gandolfiMatchingHigh, qcdMathing )
            
                muAll, c2All = c2Conditions.coeffValues()
        
                # Checking that the gotten coefficients are suitable
                testC2Ok = c2Conditions.coeffValuesOkTest( muAll, c2All )
            else:
                muAll = None
                c2All = None
                testC2Ok = False

        # Do not proceed if tested failed
        if muAll == None or c2All == None or not testC2Ok:
            self.eos = None
            self.realistic = False

        else:
            # Create c2 EoS
            c2EoS = c2AGKNV( muAll, c2All, gandolfiMatchingHigh, approx = True )

            # Combining EoSs
            combinedEoS = [crustEoS, gandolfiEoS, c2EoS, qcdEoS]
            transitionPieces = [0.0] + transitions[:2] + [nQCD(muQCD, X) * cgs.mB]

            self.eos = combiningEos(combinedEoS, transitionPieces)

            # Is the pQCD EoS causal?
            test2 = causalityPerturbativeQCD(qcdEoS, muQCD)

            # Is the latent heat between EoS pieces positive?
            test3 = positiveLatentHeat(combinedEoS, transitionPieces)

            if not test2 or not test3:
                self.realistic = False
            else:
                self.realistic = True


    #solve TOV equations
    def tov(self, l = 2, m1 = -1.0, m2 = -1.0):
        t = tov(self.eos)

        assert isinstance(l, int)

        if m1 < 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho = t.mass_radius()
            self.TD = 1.0e10
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 < 0.0:
            self.mass, self.rad, self.rho, self.TD = t.massRadiusTD(l, mRef1 = m1)
            self.TDtilde = 1.0e10
        elif m1 > 0.0 and m2 > 0.0:
            self.mass, self.rad, self.rho, self.TD, TD2 = t.massRadiusTD(l, mRef1 = m1, mRef2 = m2)
            self.TDtilde = t.tidalDeformability(m1, m2, self.TD, TD2)
        else:
            self.mass, self.rad, self.rho, self.TD = t.massRadiusTD(l, mRef2 = m2)
            self.TDtilde = 1.0e10
        
        self.maxmass = np.max( self.mass )


    # interpolate radius given a mass
    # note: structure must be solved beforehand
    def radius_at(self, mass):

        if mass >= self.maxmass:
            return 0.0

        try:
            intp = interpolate.interp1d(self.mass, self.rad, kind='cubic')
            return intp(mass)
        except:
            # linear interpolant (faster, backup)
            return np.interp(mass, self.mass, self.rad)
        


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
    print struc.mass
    print struc.rad
    ax.plot(struc.rad, struc.mass)

    mass = np.linspace(0.5, 1.0, 10)
    rad  = [struc.radius_at(m) for m in mass]
    ax.plot(rad, mass, "r-")

    print mass
    print rad


    plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    plt.savefig('mr_test.pdf')





if __name__ == "__main__":
    #main(sys.argv)
    #plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    #plt.savefig('mr.pdf')

    test_tov()


