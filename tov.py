import sys
import os
import numpy as np
import units as cgs
from math import pi, log
from polytropes import monotrope, polytrope
from crust import SLyCrust
from eoslib import get_eos, glue_crust_and_core, eosLib
from scipy.integrate import solve_ivp
#from scipy.integrate import solve_ivp

from scipy.special import hyp2f1
try:
    from scipy.misc import factorial2
except:
    from scipy.special import factorial2

###################################################
#constans
Msun_inv = 1.0 / cgs.Msun
c2inv = 1.0 / cgs.c**2
inv3 = 0.333333333333333333333333333333
press_min = 6.08173e-15 * cgs.GeVfm_per_dynecm * 0.001  # TODO

#--------------------------------------------------

class tov:

    def __init__(self, peos, rhocs = np.logspace(np.log10(1.1*cgs.rhoS), np.log10(11.0*cgs.rhoS)) ):
        self.physical_eos = peos
        self.rhocs = rhocs

    def tov(self, r, y):
        P, m = y
        if P < 0.0:
            return [-10.0, 0.0]

        eden = self.physical_eos.edens_inv( P )

        dPdr = -cgs.G*(eden + P*c2inv)*(m + 4.0*pi*r**3*P*c2inv)
        dPdr /= r*(r - 2.0*cgs.G*m*c2inv)
        dmdr = 4.0*pi*r**2*eden

        return [dPdr, dmdr]

    def tovsolve(self, rhoc, tol=1.e-4):
        r = 1.0
        P = self.physical_eos.pressure( rhoc )
        eden = self.physical_eos.edens_inv( P )
        m = 4.0*pi*r**3*eden

        def neg_press1(r, y):
            return y[0] - press_min
        neg_press1.terminal = True
        neg_press1.direction = -1
        psol = solve_ivp(
                self.tov,
                (1e0, 16e5),
                [P, m],
                rtol=tol,
                atol=tol,
                method = 'LSODA',
                events = neg_press1
                )

        return psol.t[:], psol.y[0], psol.y[1]


    def mass_radius(self):
        rhocs = self.rhocs
        N = len(rhocs)

        mcurve = np.zeros(N)
        rcurve = np.zeros(N)

        mass_max = 0.0
        j = 0

        for rhoc in rhocs:
            rad, press, mass = self.tovsolve(rhoc)

            rad  *= 1.0e-5 #cm to km
            mass *= Msun_inv

            mstar = mass[-1]
            rstar = rad[-1]

            mcurve[j] = mstar
            rcurve[j] = rstar

            if mass_max < mstar:
                mass_max = mstar
                j += 1
            else:
                break

        mass_max1 = 0.
        for rhoc in np.linspace(rhocs[j-2], rhocs[-1], int(1000*(rhocs[-1]-rhocs[-2])/cgs.rhoS)*(len(rhocs)-j+1)):
            rad, press, mass = self.tovsolve(rhoc, tol=1.e-5)

            rad  *= 1.0e-5 #cm to km
            mass *= Msun_inv

            mstar = mass[-1]
            rstar = rad[-1]

            if mass_max1 < mstar:
                mass_max1 = mstar
                rho_max = rhoc
                mass_max_r = rstar
            else:
                break

        if mass_max1 > mass_max:
            j += 1

        rhocs[j-1] = rho_max
        mcurve[j-1] = mass_max1
        rcurve[j-1] = mass_max_r

        return mcurve[:j], rcurve[:j], rhocs[:j]


    # TOV [see Oppenheimer & Volkoff (1939), Phys. Rev. 55, 374] and (electric) Love number [see arXiv:1404.6798] solver
    def tovLove(self, r, y, l):
        P, m, eta = y
        if P < 0.0:
            return [-1.e8, 0.0, 0.0]
        #elif P < press_min:
        #    return [-P, 0., 0.]

        eden, cS2Inv = self.physical_eos.tov( P )

        ## Temp constant
        rInv = 1.0 / r
        tmp = 4.0 * pi * r**2
        coeff = tmp * cgs.G * c2inv
        compactness = cgs.G * m * c2inv * rInv
        pc2inv = c2inv * P

        f = 1.0 / (1.0 - 2.0 * compactness)
        A = 2.0 * f * ( 1.0 - 3.0 * compactness - 0.5 * coeff * (eden + 3.0 * pc2inv) )
        B = f * ( l * (l + 1.0) - coeff * (eden + pc2inv) * (3.0 + cS2Inv))

        dPdr = -cgs.G * (eden + pc2inv) * (m * rInv + tmp * pc2inv) * rInv * f
        dmdr = tmp * eden
        detadr = ( B - eta * (eta + A - 1.0) ) * rInv

        return [dPdr, dmdr, detadr]

    def tovLove_inv(self, P, y, l):
        r, m, eta = y

        eden, cS2Inv = self.physical_eos.tov( P )

        ## Temp constant
        rInv = 1.0 / r
        tmp = 4.0 * pi * r**2
        coeff = tmp * cgs.G * c2inv
        compactness = cgs.G * m * c2inv * rInv
        pc2inv = c2inv * P

        f = 1.0 / (1.0 - 2.0 * compactness)
        A = 2.0 * f * ( 1.0 - 3.0 * compactness - 0.5 * coeff * (eden + 3.0 * pc2inv) )
        B = f * ( l * (l + 1.0) - coeff * (eden + pc2inv) * (3.0 + cS2Inv))

        dPdr = -cgs.G * (eden + pc2inv) * (m * rInv + tmp * pc2inv) * rInv * f
        drdP = 1.0 / dPdr
        dmdr = tmp * eden
        detadr = ( B - eta * (eta + A - 1.0) ) * rInv

        return [drdP, dmdr * drdP, detadr * drdP]

    def tovLoveSolve(self, rhoc, l, tol=1.e-4):
        r = 1.0
        P = self.physical_eos.pressure( rhoc )
        eden = self.physical_eos.edens_inv( P )
        m = 4.0*pi*r**3*eden
        eta = 1.0 * l

        def neg_press(r, y, l):
            return y[0] - press_min
        neg_press.terminal = True
        neg_press.direction = -1
        psol = solve_ivp(
                self.tovLove,
                (1e0, 16e5),
                [P, m, eta], 
                args=(1.0 * l, ), 
                rtol=tol,
                atol=tol,
                method = 'LSODA',
                events = neg_press
                )

        # radius (cm), pressure (Ba), mass (g), eta (-)
        return psol.t[:], psol.y[0], psol.y[1], psol.y[2]
    
    def massRadiusTD(self, l, mRef1 = -1.0, mRef2 = -1.0, N = 100):
        rhocs = self.rhocs
        N = len(rhocs)

        mcurve = np.zeros(N)
        rcurve = np.zeros(N)
        etaCurve = np.zeros(N)
        TDcurve = np.zeros(N)

        mass_max = 0.0
        j = 0
        jRef1 = 0
        jRef2 = 0
        for rhoc in rhocs:
            rad, press, mass, eta = self.tovLoveSolve(rhoc, l)
            mstar = mass[-1]
            rstar = rad[-1]
            etaStar = eta[-1]

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

        mass_max1 = 0.
        for rhoc in np.linspace(rhocs[j-2], rhocs[-1], int(1000*(rhocs[-1]-rhocs[-2])/cgs.rhoS)*(len(rhocs)-j+1)):
            rad, press, mass, eta = self.tovLoveSolve(rhoc, l, tol=1.e-5)
            mstar = mass[-1]
            rstar = rad[-1]
            etaStar = eta[-1]
            td_star = self.loveElectric(l, mstar, rstar, etaStar, tdFlag = True)

            if mass_max1 < mstar:
                mass_max1 = mstar
                rho_max = rhoc
                mass_max_r = rstar
                mass_max_td = td_star
            else:
                break

        if mass_max1 > mass_max:
            j += 1
        mass_max = mass_max1
        rhocs[j-1] = rho_max
        mcurve[j-1] = mass_max
        rcurve[j-1] = mass_max_r
        TDcurve[j-1] = mass_max_td

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

        rcurve=[x * 1.0e-5 for x in rcurve]
        mcurve=[x * Msun_inv for x in mcurve]

        if mRef1 > 0.0 and mRef2 < 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef1
        elif mRef1 > 0.0 and mRef2 > 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef1, tidalDeformabilityRef2
        elif mRef1 < 0.0 and mRef2 > 0.0:
            return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j], tidalDeformabilityRef2

        return mcurve[:j], rcurve[:j], rhocs[:j], TDcurve[:j]


    def loveElectric(self, l, mass, radius, eta, tdFlag = False):
        # Temp constants
        compactness = 2.0 * cgs.G * mass * c2inv / radius
        comp_inv1 = 1.0 / (compactness - 1.0)
        coeff = 2.0 * compactness * comp_inv1

        if l==2:
            comp_log1 = log(1.0 - compactness)
            comp6_inv = 1.0 / compactness**6
            tmp = -2.5 * compactness * comp6_inv

            B1 = (compactness - 2.0) * compactness * (compactness * (6.0 + compactness) - 6.0 ) * comp_inv1**2 + 12.0 * comp_log1
            DB1 = compactness * (-60.0 + 150.0 * compactness - 110.0 * compactness**2 + 15.0 * compactness**3 + 3.0 * compactness**4) * comp_inv1**3 + 60.0 * comp_log1

            love = -0.5 * (eta - 2.0 + coeff)
            love /= ( (eta + 3.0 + coeff) * B1 - DB1 ) * tmp
        else:
            # Hypergeometric functions
            A1 = hyp2f1(-1.0*l, 2.0-1.0*l, -2.0*l, compactness)
            B1 = hyp2f1(1.0*l+1.0, 1.0*l+3.0, 2.0*l+2.0, compactness)

            # Derivatives of the hypergeometric functions multiplicated by the radius
            DA1 = 0.5 * compactness * (1.0*l - 2.0) * hyp2f1(1.0-1.0*l, 3.0-1.0*l, 1.0-2.0*l, compactness)
            DB1 = -0.5 * compactness * (1.0*l + 3.0) * hyp2f1(1.0*l+2.0, 1.0*l+4.0, 2.0*l+3.0, compactness)

            love = 0.5 * ( DA1 - (eta - 1.0*l + coeff) * A1 )
            love /= (eta + 1.0*l + 1.0 + coeff) * B1 - DB1

        if tdFlag:
            if l == 2:
                return (21.0 + inv3) * compactness * comp6_inv * love
            return 2.0 * love * pow(0.5 * compactness, -2.0*l-1.0) / factorial2(2*l-1)

        return love


    def tidalDeformability(self, m1, m2, lambda1, lambda2):
        # 16/13 ~ 1.23076923077
        return 1.23076923077 * ( (m1 + 12.0 * m2) * m1**4 * lambda1 + (m2 + 12.0 * m1) * m2**4 * lambda2 ) / (m1 + m2)**5


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

