import numpy as np
import units as cgs
from math import exp, pi
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy.polynomial.chebyshev import chebval, chebder


class polytropicSpectral:

    def __init__(self, spectralCoefficients, lowDensity, pressureHigh):
        self.pressureLow = lowDensity[0]
        self.energyDensityLow = lowDensity[1]
        self.densityLow = lowDensity[2]
        self.pressureHigh = pressureHigh
        self.coeff = spectralCoefficients

    # Spectral function as a function of the pressure (Ba) 
    # The base functions are chosen to be Chebyshev polynomials
    def spectralRepresentation(self, pressure):
        # Scaled pressure so that the possible values are at interval [-1, 1]
        pressureScaled = 2.0 * (pressure - self.pressureHigh) / (self.pressureHigh - self.pressureLow) + 1.0

        # Argument of the exponent (see eq. 17 and the output)
        # In this case a Chebyshev series
        argument = chebval(pressureScaled, self.coeff)

        # Due to computational reasons there is a cutoff
        if argument > 700.0:
            return -1.0

        return exp(argument)

    def integrandMu(self, pressure):
        gamma = self.spectralRepresentation(pressure)

        return 1.0 / (gamma * pressure)

    def integralEnergyDensity(self, pressure):
        gamma = self.spectralRepresentation(pressure)

        return self.mu(pressure) / gamma

    def mu(self, pressure):
        integral = quad(self.integralMu, self.pressureLow, pressure)[0]

        return exp(-integral)

    ################################################
    # Pressure (Ba) as a function of the mass density (g/cm)
    def pressure(self, rho):
        press = fsolve(self.rho, self.pressureLow, args = rho)

        return press[0]

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    # Energy density (g/cm^3) as a function of the pressure (Ba)
    def edens_inv(self, pressure):
        integral = quad(self.integralEnergyDensity, self.pressureLow, pressure)[0]

        return ( self.energyDensityLow + integral / cgs.c**2 ) / self.mu(pressure)

    # Mass density (g/cm^3) as a function of the pressure (Ba)
    def rho(self, pressure, rho0 = 0.0):
        return self.densityLow / self.mu(pressure) - rho0

    # Speed of sound square devided by the speed of light square as a function of the pressure (Ba)
    def speed2(self, pressure):
        gamma = self.spectralRepresentation(pressure)

        if gamma < 0.0:
            return -1.0

        return ( gamma * pressure ) / ( pressure + self.edens_inv(pressure) * cgs.c**2 )
        


# Causal spectral representation of the EOS
# See arXiv:1804.04072 for details
class causalSpectral:

    def __init__(self, spectralCoefficients, lowDensity, pressureHigh):
        self.pressureLow = lowDensity[0]
        self.energyDensityLow = lowDensity[1]
        self.densityLow = lowDensity[2]
        self.pressureHigh = pressureHigh
        self.coeff = spectralCoefficients

    # Spectral function as a function of the pressure (Ba) 
    # The base functions are chosen to be Chebyshev polynomials
    # (see also eq. 10 and 17)
    def spectralRepresentation(self, pressure):
        # Scaled pressure so that the possible values are at interval [-1, 1]
        pressureScaled = 2.0 * (pressure - self.pressureHigh) / (self.pressureHigh - self.pressureLow) + 1.0

        # Argument of the exponent (see eq. 17 and the output)
        # In this case a Chebyshev series
        argument = chebval(pressureScaled, self.coeff)

        # Due to computational reasons there is a cutoff
        if argument > 700.0:
            return -1.0

        return exp(argument)

    # Integrand related to the value of the mass density as a function of the pressure (Ba)
    # See rho function for details
    def integrandRho(self, pressure):
        return ( self.spectralRepresentation(pressure) + 1.0 ) / ( pressure + self.edens_inv(pressure) * cgs.c**2 )


    ################################################
    # Pressure (Ba) as a function of the mass density (g/cm)
    def pressure(self, rho):
        press = fsolve(self.rho, self.pressureLow, args = rho)

        return press[0]

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    # Energy density (g/cm^3) as a function of the pressure (Ba)
    # (see eq. 12 and 18)
    def edens_inv(self, pressure):
        integral = quad(self.spectralRepresentation, self.pressureLow, pressure)[0]

        return self.energyDensityLow + (pressure - self.pressureLow + integral) / cgs.c**2

    # Mass density (g/cm^3) as a function of the pressure (Ba)
    def rho(self, pressure, rho0 = 0.0):
        integral = quad(self.integrandRho, self.pressureLow, pressure)[0]

        return self.densityLow * exp(integral) - rho0

    # Speed of sound square devided by the speed of light square as a function of the pressure (Ba)
    # (see eq. 10)    
    def speed2(self, pressure):
        gamma = self.spectralRepresentation(pressure)

        if gamma < 0.0:
            return -1.0

        return 1.0 / ( gamma + 1.0)

    # Derivative of the above function
    def speed2Derivative(self, pressure):
        # Spectral function
        gamma = self.spectralRepresentation(pressure)

        # Derivative of the below formula
        pressureScaledDerivative = 2.0 / (self.pressureHigh - self.pressureLow)

        # Scaled pressure so that the possible values are at interval [-1, 1]
        pressureScaled = pressureScaledDerivative * (pressure - self.pressureHigh) + 1.0

        # Coefficients of the derivated Chebyshev sum
        coeffDerivative = chebder(self.coeff, scl=pressureScaledDerivative)

        # Derivative of the Chebyshev sum
        sumDerivative = chebval( pressureScaled, coeffDerivative )

        return -sumDerivative * ( gamma / (gamma + 1.0)**2 )


# Interpolated spectral representation of the EOS
# NB Linear interpolation is used
class spectralInterpolated:

    def __init__(self, spectralEOS, pressureLimits, densityHigh, N=1000, flagEnergyDensity = True, flagSpeed2 = True, flagRho = True, flagRhoInv = True):
        eos = spectralEOS

        pressureLow = pressureLimits[0]
        pressureHigh = pressureLimits[1]
        press = np.linspace(pressureLow, pressureHigh, N)

        # Energy density (g/cm^3)
        if flagEnergyDensity:
            edens = [eos.edens_inv(p) for p in press]
            self.edensInterpolated = interp1d( press, edens )

        # Speed of sound square (unitless)
        if flagSpeed2:
            speed = [eos.speed2(p) for p in press]
            self.speedInterpolated = interp1d( press, speed )

        # Mass density (g/cm^3)
        if flagRho or flagRhoInv:
            rho = [eos.rho(p) for p in press]
            self.rhoInterpolated = interp1d( press, rho )

        # Pressure (Ba) as a function of mass density (g/cm^3)
        if flagRhoInc:
            if densityHigh - rho[-1] > 0.0:
                rho.append(densityHigh)
                press = np.append(press, pressureHigh)

            self.rhoInvInterpolated = interp1d( rho, press )

    def pressure(self, rho):
        return self.rhoInvInterpolated(rho)

    #vectorized version
    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    def edens_inv(self, pressure):
        return self.edensInterpolated(pressure)

    def rho(self, pressure, rho0 = 0.0):
        return self.rhoInterpolated(pressure)

    def speed2(self, pressure):
        return self.speedInterpolated(pressure)


# Calculates the unknown coefficients of the Chebyshev series
class matchPolytropicSpectralWithLimits:

    def __init__(self, coeffKnown, lowDensity, highDensity):
        self.coeffKnown = coeffKnown

        self.pressureLow = lowDensity[0]
        self.lowDensity = lowDensity[0:3]

        self.pressureHigh = highDensity[0]
        self.energyDensityHigh = highDensity[1]
        self.densityHigh = highDensity[2]

        if len(lowDensity) == 4:
            self.speed2Low = lowDensity[3]
        else:
            self.speed2Low = -1.0

        if len(highDensity) == 4:
            self.speed2High = highDensity[3]
        else:
            self.speed2High = -1.0

    # Determinates the difference between the boundary conditions and the given spectral EOS
    # coeffUnkown: nonpredetermined coefficients
    def solveCoeff(self, coeffUnknown):
        # All coefficients in the right order
        coeff1 = self.coeffKnown[:]
        coeff1.insert(0, coeffUnknown[0])
        coeff1.insert(1, coeffUnknown[1])

        if self.speed2Low > 0.0 or self.speed2High > 0.0:
            coeff1.insert(2, coeffUnknown[2])

        if self.speed2Low > 0.0 and self.speed2High > 0.0:
            coeff1.insert(3, coeffUnknown[3])

        # Spectral EOS based on the given coefficients
        spectralEOS = polytropicSpectral(coeff1, self.lowDensity, self.pressureHigh)

        # Energy density at the upper bound
        eHigh = spectralEOS.edens_inv(self.pressureHigh)

        # Mass density at the upper bound
        rhoHigh = spectralEOS.rho(self.pressureHigh)

        # Speed of sound at the lower bound
        if self.speed2Low > 0.0:
            c2Low = spectralEOS.speed2(self.pressureLow)

        # Speed of sound at the upper bound
        if self.speed2High > 0.0:
            c2High = spectralEOS.speed2(self.pressureHigh)

        # Output list
        out = [eHigh / self.energyDensityHigh - 1.0]
        out.append(rhoHigh / self.densityHigh - 1.0)
        
        if self.speed2Low > 0.0:
            out.append(c2Low - self.speed2Low)

        if self.speed2High > 0.0:
            out.append(c2High - self.speed2High)

        return out

    # Determines all coefficients of the Chebyshev series
    def coeffValues(self, coeffGuess=[0.1,1.0]):
        [coeffs, infoC, flagC, mesgC] = fsolve(self.solveCoeff, coeffGuess, full_output=1, xtol = 1.0e-9)

        # Did not work out
        if flagC != 1:
            coeffs = np.array([-1.0, -1.0])

        coeffAll = coeffs.tolist() + self.coeffKnown
            
        return coeffAll


# Calculates the unknown coefficients of the Chebyshev series
class matchCausalSpectralWithLimits:

    def __init__(self, coeffKnown, lowDensity, highDensity, xQCD = -1.0):
        self.coeffKnown = coeffKnown

        self.pressureLow = lowDensity[0]
        self.speed2Low = lowDensity[3]
        self.lowDensity = lowDensity[0:3]

        self.pressureHigh = highDensity[0]
        self.energyDensityHigh = highDensity[1]
        self.densityHigh = highDensity[2]
        self.speed2High = highDensity[3]

        self.x = xQCD

    # Determinates the difference between the boundary conditions and the given spectral EOS
    # coeffUnkown: nonpredetermined coefficients
    def solveCoeff(self, coeffUnknown):
        # All coefficients in the right order
        coeff1 = self.coeffKnown[:]
        coeff1.insert(0, coeffUnknown[0])
        coeff1.insert(1, coeffUnknown[1])
        coeff1.insert(2, coeffUnknown[2])
        coeff1.insert(3, coeffUnknown[3])
        
        if self.x > 0.9:
            coeff1.insert(4, coeffUnknown[4])

        # Spectral EOS based on the given coefficients
        spectralEOS = causalSpectral(coeff1, self.lowDensity, self.pressureHigh)

        # Speed of sound at the lower bound
        c2Low = spectralEOS.speed2(self.pressureLow)

        # Speed of sound at the upper bound
        c2High = spectralEOS.speed2(self.pressureHigh)

        # Energy density at the upper bound
        eHigh = spectralEOS.edens_inv(self.pressureHigh)

        # Mass density at the upper bound
        rhoHigh = spectralEOS.rho(self.pressureHigh)

        # Output list
        out = [c2Low - self.speed2Low]
        out.append(c2High - self.speed2High)
        out.append(eHigh / self.energyDensityHigh - 1.0)
        out.append(rhoHigh / self.densityHigh - 1.0)

        if self.x > 0.9:
            # Derivative of the speed of sound at the upper bound
            c2DervaHigh = spectralEOS.speed2Derivative(self.pressureHigh)
            out.append(c2DervaHigh / self.speed2DerivativeQCD(self.parametersQCD) - 1.0)

        return out

    # Determines all coefficients of the Chebyshev series
    def coeffValues(self, coeffGuess=[0.1,1.0,2.0,-2.0]):
        [coeffs, infoC, flagC, mesgC] = fsolve(self.solveCoeff, coeffGuess, full_output=1, xtol = 1.0e-9)

        # Did not work out
        if flagC != 1:
            coeffs = np.array([-1.0, -1.0, -1.0, -1.0])

        coeffAll = coeffs.tolist() + self.coeffKnown
            
        return coeffAll

    # Derivative of the speed of sound respect to the pressure (Ba)
    def speed2DerivativeQCD(self, pressure):
        # Chemical potential
        m = fsolve(pQCD, 2.6, args = (self.x, pressure))[0]

        a = 0.5034 * pow(self.x, 0.3553)
        b = 1.452 * pow(self.x, 0.9101)
        c = 0.9008

        speed2Derivative = -((54.0*a*(b - m)**2*(2.0*c*(b - m)**2*(5.0*b**2 + 4.0*b*m - 3.0*m**2) + a*b*(10.0*b**2 - 12.0*b*m + 3.0*m**2))*pi**2)/((a*(4.0*b - 3.0*m) + 4.0*c*(b - m)**2)*m**3*(6.0*c*(b - m)**3 + a*(6.0*b**2 - 8.0*b*m + 3.0*m**2))**2))
        speed2Derivative /= (cgs.GeV3_to_fm3 * cgs.GeVfm_per_dynecm)

        return speed2Derivative
