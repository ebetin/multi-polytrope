import sys
import os
import numpy as np
import units as cgs
from math import pi
from polytropes import monotrope, polytrope
from crust import SLyCrust
from eoslib import get_eos, glue_crust_and_core, eosLib
from scipy.integrate import odeint, solve_ivp
#from scipy.integrate import solve_ivp

from scipy.special import hyp2f1
try:
    from scipy.misc import factorial2
except:
    from scipy.special import factorial2

import math


#--------------------------------------------------

class tov:

    def __init__(self, peos, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        self.physical_eos = peos
        self.rhocs = rhocs

    def tov(self, y, r):
        P, m = y
        eden = self.physical_eos.edens_inv( P )

        dPdr = -cgs.G*(eden + P/cgs.c**2)*(m + 4.0*pi*r**3*P/cgs.c**2)
        dPdr = dPdr/(r*(r - 2.0*cgs.G*m/cgs.c**2))
        dmdr = 4.0*pi*r**2*eden

        return [dPdr, dmdr]

    def tovsolve(self, rhoc, N = 800):
        r = np.linspace(1e0, 18e5, N)
        P = self.physical_eos.pressure( rhoc )
        eden = self.physical_eos.edens_inv( P )
        m = 4.0*pi*r[0]**3*eden

        psol = odeint(self.tov, [P, m], r, rtol=1.0e-4, atol=1.0e-4)

        #sol = solve_ivp(
        #        self.tov,


        return r, psol[:,0], psol[:,1]


    def mass_radius(self, N = 200):
        mcurve = np.zeros(N)
        rcurve = np.zeros(N)
        #rhocs = np.logspace(14.3, 15.8, N)
        rhocs = self.rhocs
        mass_max = 0.0
        j = 0

        for rhoc in rhocs:
            rad, press, mass = self.tovsolve(rhoc)

            rad  /= 1.0e5 #cm to km
            mass /= cgs.Msun

            mstar = mass[-1]
            rstar = rad[-1]
            for i, p in enumerate(press):
                if p > 0.0:
                    tmp = p / (press[i-1]- p)
                    mstar = mass[i] - (mass[i-1] - mass[i]) * tmp
                    rstar = rad[i] - (rad[i-1] - rad[i]) * tmp
            mcurve[j] = mstar
            rcurve[j] = rstar

            if mass_max < mstar:
                mass_max = mstar
                j += 1
            else:
                break

        return mcurve[:j], rcurve[:j], rhocs[:j]



    # TOV [see Oppenheimer & Volkoff (1939), Phys. Rev. 55, 374] and (electric) Love number [see arXiv:1404.6798] solver
    def tovLove(self, r, y, l):
        P, m, eta = y
        eden, cS2Inv = self.physical_eos.tov( P )

        ## Temp constant
        rInv = 1.0 / r
        cL2Inv = 1.0 / cgs.c**2
        compactness = cgs.G * m * cL2Inv * rInv
        f = 1.0 / (1.0 - 2.0 * compactness)
        A = 2.0 * f * ( 1.0 - 3.0 * compactness - 2.0 * pi * cgs.G * cL2Inv * r**2 * (eden + 3.0 * cL2Inv * P) )
        B = f * ( l * (l + 1.0) - 4.0 * pi * cgs.G * cL2Inv * r**2 * (eden + cL2Inv * P) * (3.0 + cS2Inv))

        dPdr = -cgs.G*(eden + P*cL2Inv)*(m + 4.0*pi*r**3*P*cL2Inv) * rInv**2 * f
        dmdr = 4.0*pi*r**2*eden
        detadr = ( B - A * eta - eta * (eta - 1.0) ) * rInv

        return [dPdr, dmdr, detadr]

    def tovLoveSolve(self, rhoc, l):
        r = 1.0
        P = self.physical_eos.pressure( rhoc )
        eden = self.physical_eos.edens_inv( P )
        m = 4.0*pi*r**3*eden
        eta = 1.0 * l

        psol = solve_ivp(
                self.tovLove,
                (1e0, 17e5),
                [P, m, eta], 
                args=(1.0 * l, ), 
                rtol=1.0e-4, 
                atol=1.0e-6,
                #max_step = 1000,
                method = 'LSODA' #TODO is this ok?
                )

        return psol.t[:], psol.y[0], psol.y[1], psol.y[2]

    
    def massRadiusTD(self, l, mRef1 = -1.0, mRef2 = -1.0, N = 100):#TODO optimal value of N
        mcurve = np.zeros(N)
        rcurve = np.zeros(N)
        etaCurve = np.zeros(N)
        TDcurve = np.zeros(N)
        # rhoS = 10**14.427817280025156, 1.2rhoS = 10**14.46920996518338, 10**15.6 ~ 14.9rhoS
        #rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS))
        rhocs = self.rhocs
        #np.logspace(14.427817280025156, 15.6, N) #TODO lower limit
        mass_max = 0.0
        j = 0
        jRef1 = 0
        jRef2 = 0

        for rhoc in rhocs:
            rad, press, mass, eta = self.tovLoveSolve(rhoc, l)
            mstar = mass[-1]
            rstar = rad[-1]
            etaStar = eta[-1]
            for i, p in reversed(list(enumerate(press))):
                if p > 0.0:
                    if i == ( len(press) - 1 ):
                        break
                    if i > 0:
                        if press[i-1] == p:
                            continue
                    else:
                        mstar = mass[-1]
                        rstar = rad[-1]
                        etaStar = eta[-1]
                        break
                    tmp = p / (press[i-1] - p)
                    mstar = mass[i] - (mass[i-1] - mass[i]) * tmp
                    rstar = rad[i] - (rad[i-1] - rad[i]) * tmp
                    etaStar = eta[i] - (eta[i-1] - eta[i]) * tmp
                    if rstar < rad[i]:
                        mstar = mass[i]
                        rstar = rad[i]
                        etaStar = eta[i]
                    else:
                        if rstar > rad[i+1]:
                            mstar = mass[i]
                            rstar = rad[i]
                            etaStar = eta[i]
                    break
            mcurve[j] = mstar
            rcurve[j] = rstar
            etaCurve[j] = etaStar
            TDcurve[j] = self.loveElectric(l, mstar, rstar, etaStar, tdFlag = True)
            if mstar < mRef1 and mRef1 > 0.0:
                jRef1 += 1

            if mstar < mRef2 and mRef2 > 0.0:
                jRef2 += 1

            if mass_max < mstar:
                mass_max = mstar
                j += 1
            else:
                break

        if mRef1 > 0.0 and mass_max > mRef1:
            massTerm = (mRef1 - mcurve[jRef1]) / (mcurve[jRef1] - mcurve[jRef1-1])
            radiusRef1 = (rcurve[jRef1] - rcurve[jRef1-1]) * massTerm + rcurve[jRef1]
            etaRef1 = (etaCurve[jRef1] - etaCurve[jRef1-1]) * massTerm + etaCurve[jRef1]
            tidalDeformabilityRef1 = self.loveElectric(l, mRef1, radiusRef1, etaRef1, tdFlag = True)
        else:
            tidalDeformabilityRef1 = 0.0

        if mRef2 > 0.0  and mass_max > mRef2:
            massTerm = (mRef2 - mcurve[jRef2]) / (mcurve[jRef2] - mcurve[jRef2-1])
            radiusRef2 = (rcurve[jRef2] - rcurve[jRef2-1]) * massTerm + rcurve[jRef2]
            etaRef2 = (etaCurve[jRef2] - etaCurve[jRef2-1]) * massTerm + etaCurve[jRef2]
            tidalDeformabilityRef2 = self.loveElectric(l, mRef2, radiusRef2, etaRef2, tdFlag = True)
        else:
            tidalDeformabilityRef2 = 0.0

        rcurve=[x / 1.0e5 for x in rcurve]
        mcurve=[x / cgs.Msun for x in mcurve]
        if mRef1 > 0.0 and mRef2 < 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef1
        elif mRef1 > 0.0 and mRef2 > 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef1, tidalDeformabilityRef2
        elif mRef1 < 0.0 and mRef2 > 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef2

        return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j]


    def loveElectric(self, l, mass, radius, eta, tdFlag = False):
        # Temp constants
        compactness = 2.0 * cgs.G * mass / (radius * cgs.c**2)
        coeff = 2.0 * compactness / (1.0 - compactness)

        # Hypergeometric functions
        A1 = hyp2f1(-1.0*l, 2.0-1.0*l, -2.0*l, compactness)
        B1 = hyp2f1(1.0*l+1.0, 1.0*l+3.0, 2.0*l+2.0, compactness)

        # Derivatives of the hypergeometric functions multiplicated by the radius
        DA1 = 0.5 * compactness * (1.0*l - 2.0) * hyp2f1(1.0-1.0*l, 3.0-1.0*l, 1.0-2.0*l, compactness)
        DB1 = -0.5 * compactness * (1.0*l + 3.0) * hyp2f1(1.0*l+2.0, 1.0*l+4.0, 2.0*l+3.0, compactness)

        love = 0.5 * ( DA1 - (eta - 1.0*l - coeff) * A1 ) / ( (eta + 1.0*l + 1.0 - coeff) * B1 - DB1 )
        
        if tdFlag:
            return 2.0 * love * pow(0.5 * compactness, -2.0*l-1.0) / factorial2(2*l-1)

        else:
            return love


    def tidalDeformability(self, m1, m2, lambda1, lambda2):
        #if lambda1 > 1e10 or lambda2 > 1e10:
        #    return np.inf
        #else:
        return 16.0 / 13.0 * ( (m1 + 12.0 * m2) * m1**4 * lambda1 + (m2 + 12.0 * m1) * m2**4 * lambda2 ) / (m1 + m2)**5


#--------------------------------------------------
def main(argv):

    import matplotlib
    import matplotlib.pyplot as plt
    from label_line import label_line

    #from matplotlib import cm
    import palettable as pal
    cmap = pal.colorbrewer.qualitative.Set1_6.mpl_colormap
    #cmap = pal.cmocean.sequential.Matter_8.mpl_colormap #best so far
    #cmap = pal.wesanderson.Zissou_5.mpl_colormap

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


    # test single eos separately
    if False: 
        key = 'SLy'
        #key = 'BGN1H1'
        #key = 'ALF2'
        #key = 'ENG'
        #key = 'PAL6'

        dense_eos = get_eos(key)
        eos = glue_crust_and_core( SLyCrust, dense_eos )
        t = tov(eos)

        mass, rad, rho = t.mass_radius()
        print(mass)
        print(rad)
        ax.plot(rad, mass)


    if True:
        i = 0
        for key, value in eosLib.iteritems():

            print("Solving TOVs for:", key, "(",i, ")")

            dense_eos = get_eos(key)
            eos = glue_crust_and_core( SLyCrust, dense_eos )
            t = tov(eos)
            mass, rad, rhoc = t.mass_radius()
            
            linestyle='solid'
            col = 'k'
            if value[4] == 'npem':
                col = 'k'
            if value[4] == 'meson':
                col = 'b'
            if value[4] == 'hyperon':
                col = 'g'
            if value[4] == 'quark':
                col = 'r'

            l, = ax.plot(rad, mass, color=col, linestyle=linestyle, alpha = 0.9)


            # labels for lines
            near_y = 1.45
            near_x = None
            rotation_offset=180.0
            if key == 'APR3':
                near_x = 11.0
                near_y = None
            if key == 'ENG':
                near_x = 11.2
                near_y = None
            if key == 'ALF2':
                near_y = 1.0
                rotation_offset = 0.0
            if key == 'MPA1':
                near_x = 12.0
                near_y = None
            if key == 'MS1b':
                near_y = 0.4
            if key == 'MS1':
                near_y = 2.3

            label_line(l, key, near_y=near_y, near_x=near_x, rotation_offset=rotation_offset)



            i += 1


    #plot approximate central pressure isocurves
    #twice nuclear saturation density
    if False:
        x = [11.0, 15.8]
        y = [0.4, 2.5]
        ax.plot(x, y, "r--")

        txt = ax.text(15.5, 2.35, r'$2 \rho_n$', rotation=32, ha='center', va='center', size=8)
        txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=3))




if __name__ == "__main__":
    main(sys.argv)
    plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    plt.savefig('mr.pdf')

