from scipy.optimize import fsolve
from math import log
from math import pi

import units as cgs

class qcd:

    def __init__(self, muEOS, xEOS, p0EOS, e0EOS, rhooEOS, gammaEOS):
        self.c = 0.9008
        self.d1 = 0.5034
        self.d2 = 1.452
        self.v1 = 0.3553
        self.v2 = 0.9101

        self.conv = 1.0e-6/1.9732705**3 #MeV^3->fm^-3 (approx.)

        self.x = xEOS 
        self.mu = muEOS #MeV!
        self.p0 = p0EOS
        self.e0 = e0EOS
        self.rhoo0 = rhooEOS
        self.gamma = gammaEOS

    def a(self):
        return self.d1 * pow(self.x,-self.v1)

    def b(self):
        return self.d2 * pow(self.x,-self.v2)

    def pSB(self, mu):
        return 0.75 / pi**2 * (mu / 3.0)**4

    def pQCD(self, mu, p=0):
        pressure = self.pSB(mu) * (self.c - self.a() / (0.001 * mu - self.b()))

        return pressure * self.conv - p #TODO units? (now MeV/fm^3)

    def nQCD(self, mu, rhoo=0):
        density = (mu / 3.0)**3 / pi**2 * (self.c - self.a() / (0.001 * mu - self.b())) + self.pSB(mu) * (0.001 * self.a()) / (0.001 * mu - self.b())**2

        return density * self.conv- rhoo #TODO units? (now 1/fm^3)

    def eQCD(self, mu, e=0):
        energy = mu * self.nQCD(mu) - self.pQCD(mu)

        return energy - e #TODO units? (now MeV/fm^3)

    def pQCD_energy(self, e):
        mu = fsolve(self.eQCD, self.mu, args = e)[0]

        return self.pQCD(mu)

    def pQCD_density(self, rhoo):
        mu = fsolve(self.nQCD, self.mu, args = rhoo)[0]

        return self.pQCD(mu)

    def nQCD_pressure(self, p):
        if p<0.0:
            return 0.0

        mu = fsolve(self.pQCD, self.mu, args = p)[0]

        return self.nQCD(mu)

    def eQCD_density(self, rhoo):
        mu = fsolve(self.nQCD, self.mu, args = rhoo)[0]

        return self.eQCD(mu)

    def eQCD_pressure(self, p):
        mu = fsolve(self.pQCD, self.mu, args = p)[0]

        return self.eQCD(mu)

    def solveGamma(self, gammaUnknown):
        p = [self.p0]#TODO units???
        e = [self.e0]#TODO units???

        rhoo = self.rhoo0[:]#TODO units???
        rhoo.append(self.nQCD(self.mu))

        n=len(rhoo)

        g = self.gamma[:]
        g.insert(0, gammaUnknown[0])
        g.insert(1, gammaUnknown[1])

        for k in range(1, n):
            p.append(p[-1] * (rhoo[k] / rhoo[k-1])**g[k-1])

            if g[k-1] == 1:
                e.append(e[k-1] * p[k] / p[k-1] * log(rhoo[k] / rhoo[k-1]))
            else:
                e.append(p[k] / (g[k-1] - 1.0) + (e[k-1] - p[k-1] / (g[k-1] - 1.0)) * (rhoo[k] / rhoo[k-1]))

        out = [p[-1] - self.pQCD(self.mu)]
        out.append(e[-1] - self.eQCD(self.mu))

        return out

class qmc:

    def __init__(self, rhooS, a, alpha, b, beta, m):
        self.a = a
        self.b = b
        self.m = m
        self.beta = beta
        self.alpha = alpha
        self.rhooS = rhooS
        self.S = 32.1 #MeV
        self.g = 0.8

    def beta_equilibrium(self, pf, rhoo):
         return (8.0 * pf - 4.0) * self.S * cgs.mev_per_erg * (rhoo / self.rhooS)**self.g + cgs.hbar * cgs.c * (3.0 * pi**2 * pf * rhoo / self.m)**(1.0/3.0)

    def proton_fraction(self, rhoo):
        return fsolve(self.beta_equilibrium, 0.00, args = rhoo)[0]
    
    def qmc_energy(self, rhoo, e=0):
        pf = self.proton_fraction(rhoo)

        qmc_binding_low = self.a * pow(rhoo / self.rhooS, self.alpha) 
        qmc_binding_high = self.b * pow(rhoo / self.rhooS, self.beta)
        qmc_beta = -4.0 * pf * (1.0-pf) * self.S * (rhoo / self.rhooS)**self.g
        qmc_electron = 0.75 * pf * cgs.hbar * cgs.c * (3.0 * pi**2 * pf * rhoo / self.m)**(1.0/3.0)
        qmc_binding = (qmc_binding_low + qmc_binding_high + qmc_beta) * cgs.mev_per_erg + qmc_electron

        return (qmc_binding / self.m + cgs.c**2) * rhoo - e

    def qmc_pressure(self, rhoo, p=0):
        pf = self.proton_fraction(rhoo)

        qmc_low_density = self.a * self.alpha * pow(rhoo / self.rhooS, self.alpha)
        qmc_high_density = self.b * self.beta * pow(rhoo / self.rhooS, self.beta)
        qmc_beta = -4.0 * pf * (1.0-pf) * self.S * self.g * (rhoo / self.rhooS)**self.g
        qmc_electron = 0.25 * pf * cgs.hbar * cgs.c * (3.0 * pi**2 * pf * rhoo / self.m)**(1.0/3.0)
        qmc_total = (qmc_low_density + qmc_high_density + qmc_beta) * cgs.mev_per_erg + qmc_electron

        return qmc_total * rhoo / self.m - p

    def qmc_chempot(self, rhoo, mu=0):
        #qmc_low = self.a * (self.alpha + 1.0) * pow(rhoo / self.rhooS, self.alpha)
        #qmc_high = self.b * (self.beta + 1.0) * pow(rhoo / self.rhooS, self.beta)
        p = qmc_pressure(rhoo)
        e = qmc_energy(rhoo)

        return (p + e) * self.m / rhoo - mu
        #return (qmc_low + qmc_high) * cgs.mev_per_erg + self.m * cgs.c**2 - mu

    def qmc_density(self, p):
        if p < 0.0:
            return 0.0

        return fsolve(self.qmc_pressure, 0.5 * self.rhooS, args = p)[0]

    def qmc_pressure_energy(self, e):
        rhoo = fsolve(self.qmc_energy, 0.5 * self.rhooS, args = e)[0]

        return self.qmc_pressure(rhoo)

    def qmc_pressure_chempot(self, mu):
        rhoo = fsolve(self.qmc_chempot, 0.5 * self.rhooS, args = mu)[0]

        return self.qmc_pressure(rhoo)

    def qmc_energy_pressure(self, p):
        rhoo = fsolve(self.qmc_pressure, 0.5 * self.rhooS, args = p)[0]

        return self.qmc_energy(rhoo)


    
#dense_eos = get_eos('PAL6')
x = qcd(2600, 1.2, 2.5, 150, [0.16, 0.64], [])
y = qmc(2.7e14, 12.7, 0.48, 4.35, 2.12, cgs.mn)
G=fsolve(x.solveGamma,[4.0,1.4])
print G, x.pQCD(2600), x.nQCD(2600), x.eQCD(2600)
