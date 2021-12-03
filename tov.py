import sys
import os
import numpy as np
import units as cgs
from math import pi, log, sqrt, asin
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
        y_len = len(y)

        if y_len == 2:
            P, m = y
        elif y_len == 3:
            P, m, eta = y
        elif y_len == 4:
            P, m, eta, _ = y
        # TODO else error!

        if P < 0.0:
            if y_len == 3:
                return -0.1*P, 0., 0.
            elif y_len == 4:
                return -0.1*P, 0., 0., 0.
            return -0.1*P, 0.

        tov_point = self.physical_eos.tov(P, length=y_len-1)

        if y_len == 2:
            eden = tov_point[0]
        elif y_len == 3:
            eden, cS2Inv = tov_point
        elif y_len == 4:
            eden, cS2Inv, rho = tov_point

        ## Temp constant
        rInv = 1.0 / r
        tmp = 4.0 * pi * r**2
        compactness = cgs.G * m * c2inv * rInv
        pc2inv = c2inv * P

        f = 1.0 / (1.0 - 2.0 * compactness)
        dPdr = -cgs.G * (eden + pc2inv) * (m * rInv + tmp * pc2inv) * rInv * f
        dmdr = tmp * eden

        if y_len > 2:
            coeff = tmp * cgs.G * c2inv
            A = 2.0 * f * ( 1.0 - 3.0 * compactness - 0.5 * coeff * (eden + 3.0 * pc2inv) )
            B = f * ( l * (l + 1.0) - coeff * (eden + pc2inv) * (3.0 + cS2Inv))

            detadr = ( B - eta * (eta + A - 1.0) ) * rInv

        if y_len == 3:
            return dPdr, dmdr, detadr
        if y_len == 4:
            dmbdr = tmp * sqrt(f) * rho
            return dPdr, dmdr, detadr, dmbdr

        return dPdr, dmdr

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

    def tovLoveSolve(self, rhoc, l=2, tol=1.e-4, flag_td=False, flag_mb=False):
        r = 1.0
        P = self.physical_eos.pressure( rhoc )
        eden = self.physical_eos.edens_inv( P )
        m = 4.0*pi*r**3*eden
        point_initial = [P, m]

        if flag_td:
            eta = 1.0 * l
            point_initial.append(eta)

            if flag_mb:
                tmp = 8. * inv3 * pi * cgs.G * eden * c2inv
                tmp_sqrt = sqrt(tmp)
                try:
                    mb = 2. * pi * rhoc * ( asin(r * tmp_sqrt) / tmp_sqrt - r * sqrt(1. - r**2 * tmp) ) / tmp
                except:
                    mb = 4. * inv3 * pi * rhoc * r**3
                point_initial.append(mb)

        try:
            def neg_press(r, y, l):
                return y[0] - press_min
            neg_press.terminal = True
            neg_press.direction = -1

            psol = solve_ivp(
                    self.tovLove,
                    (1e0, 16e5),
                    point_initial,
                    args=(1.0 * l, ),
                    rtol=tol,
                    atol=tol,
                    method = 'LSODA',
                    events = neg_press
                    )
        except:
            psol = solve_ivp(
                    self.tovLove,
                    (1e0, 16e5),
                    point_initial,
                    args=(1.0 * l, ),
                    rtol=tol,
                    atol=tol,
                    method = 'LSODA'
                    )

            press_list_index = [i for i, item in enumerate(psol.y[0]) if item < press_min]
            press_list = psol.y[0][:press_list_index[0]]
            press_list_len = len(press_list)
            rad_list = psol.t[:press_list_len]
            mass_list = psol.y[1][:press_list_len]

            if flag_td:
                eta_list = psol.y[2][:press_list_len]
                if flag_mb:
                    mb_list = psol.y[3][:press_list_len]
                    return rad_list, press_list, mass_list, eta_list, mb_list
                return rad_list, press_list, mass_list, eta_list
            return rad_list, press_list, mass_list

        # radius (cm), pressure (Ba), mass (g), eta (-), baryonic mass (g)
        if flag_td and flag_mb:
            return psol.t[:], psol.y[0], psol.y[1], psol.y[2], psol.y[3]
        elif flag_td:
            return psol.t[:], psol.y[0], psol.y[1], psol.y[2]

        return psol.t[:], psol.y[0], psol.y[1]
    
    def massRadiusTD(self, l, mRef1=-1., mRef2=-1., N=100, flag_mb=False, flag_td=False, flag_td_list=False):
        rhocs = self.rhocs
        N = len(rhocs)

        mcurve = np.zeros(N)
        rcurve = np.zeros(N)

        td_curve = None
        if flag_td_list and flag_td:
            td_curve = np.zeros(N)

        mass_max = 0.0
        j = 0
        jRef1 = 0
        jRef2 = 0
        for rhoc in rhocs:
            if flag_td_list and flag_td:
                rad, _, mass, eta = self.tovLoveSolve(rhoc, l, flag_td=True, flag_mb=False, tol=1.e-5)
            else:
                rad, _, mass = self.tovLoveSolve(rhoc, l, flag_td=False, flag_mb=False, tol=1.e-5)

            mstar = mass[-1]
            rstar = rad[-1]

            mcurve[j] = mstar
            rcurve[j] = rstar

            if flag_td_list and flag_td:
                eta_star = eta[-1]
                td_curve[j] = self.loveElectric(l, mstar, rstar, eta_star, tdFlag = True)

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
        mass_max_td = np.inf
        mass_max_b = 0.
        rhocs_short = np.linspace(rhocs[j-2], rhocs[-1], int(1000.*(rhocs[-1]-rhocs[-2])/cgs.rhoS)*(len(rhocs)-j+1))
        for rhoc in rhocs_short:
            if mRef1 > 0 or mRef2 > 0:
                if flag_mb:
                    rad, _, mass, eta, massb = self.tovLoveSolve(rhoc, l, flag_td=True, flag_mb=True, tol=1.e-5)
                    etaStar = eta[-1]
                    td_star = self.loveElectric(l, mstar, rstar, etaStar, tdFlag = True)
                    mbstar = massb[-1]
                else:
                    rad, _, mass, eta = self.tovLoveSolve(rhoc, l, flag_td=True, flag_mb=False, tol=1.e-5)
                    etaStar = eta[-1]
                    td_star = self.loveElectric(l, mstar, rstar, etaStar, tdFlag = True)
            else:
                rad, _, mass, eta = self.tovLoveSolve(rhoc, l, tol=1.e-5)

            mstar = mass[-1]
            rstar = rad[-1]

            if mass_max1 < mstar:
                mass_max1 = mstar
                rho_max = rhoc
                mass_max_r = rstar
                if flag_td:
                    mass_max_td = td_star
                    if flag_mb:
                        mass_max_b = mbstar * Msun_inv
            else:
                break

        if mass_max1 > mass_max:
            j += 1

        if j-1 == len(rhocs):
            rcurve = [x * 1.0e-5 for x in rcurve]
            mcurve = [x * Msun_inv for x in mcurve]
            return mcurve, rcurve, rhocs, td_curve, [mass_max_td, np.inf, np.inf], [mass_max_b, 0., 0.], [0., 0.]

        mass_max = mass_max1
        rhocs[j-1] = rho_max
        mcurve[j-1] = mass_max
        rcurve[j-1] = mass_max_r
        if flag_td_list and flag_td:
            td_curve[j-1] = td_star

        def rtm(m_ref, j_ref):
            td = np.inf
            massb = 0
            rad = 0

            if mass_max < m_ref or m_ref <= 0:
                return rad, td, massb

            def mr(m1, m2, r1, r2, f_td, f_mb, tol=1.e-5, mref=None):
                mass_est = (m_ref - m1) / (m1 - m2)
                rho_est = (r1 - r2) * mass_est + r1
                if not min(r1, r2) < rho_est < max(r1, r2):
                    rho_est = 0.5 * (r1 + r2)
                    if mref is not None:
                        rho_est = r2
                        if abs(m_ref - m1) < abs(m_ref - m2):
                            rho_est = r1

                out = self.tovLoveSolve(rho_est, l, flag_td=f_td, flag_mb=f_mb, tol=tol)

                if f_td and f_mb:
                    return out

                r, _, m = out
                return m, r, rho_est

            mass_a, rad_a, rho_a = mr(mcurve[j_ref], mcurve[j_ref-1], rhocs[j_ref], rhocs[j_ref-1], False, False)
            mass_b, rad_b, rho_b = mr(mass_a[-1], mcurve[j_ref], rho_a, rhocs[j_ref], False, False)
            res = mr(mass_b[-1], mass_a[-1], rho_b, rho_a, flag_td, flag_mb, mref=m_ref)
            rad = res[0][-1] * 1.e-5

            if flag_td:
                td = self.loveElectric(l, res[2][-1], res[0][-1], res[3][-1], tdFlag = True)
            if flag_mb:
                massb = res[4][-1] * Msun_inv

            return rad, td, massb

        radRef1, tidalDeformabilityRef1, massbRef1 = rtm(mRef1, jRef1)
        radRef2, tidalDeformabilityRef2, massbRef2 = rtm(mRef2, jRef2)

        rcurve=[x * 1.0e-5 for x in rcurve][:j]
        mcurve=[x * Msun_inv for x in mcurve][:j]
        rhocs = rhocs[:j]
        if flag_td_list and flag_td:
            td_curve = td_curve[:j]

        return mcurve, rcurve, rhocs, td_curve, [mass_max_td, tidalDeformabilityRef1, tidalDeformabilityRef2], [mass_max_b, massbRef1, massbRef2], [radRef1, radRef2]


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

