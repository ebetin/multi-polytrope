from polytropes import monotrope
from polytropes import polytrope
import units as cgs
import numpy as np
from scipy.interpolate import interp1d

##################################################
# Constants
c_squared = cgs.c**2
c2_inv = 1.0 / c_squared

##################################################
#SLy (Skyrme) crust (Douchin & Haensel A&A 380, 151 (2001); see also Read et al. Phys.Rev. D79, 124032 (2009))
KSLy = [6.80110e-9, 1.06186e-6, 5.32697e1, 3.99874e-8] #Scaling constants
GSLy = [1.58425, 1.28733, 0.62223, 1.35692] #polytropic indices
RSLy = [1.e4, 2.44034e7, 3.78358e11, 2.62780e12 ] #transition depths

tropes = []
trans = []

pm = None
for (K, G, r) in zip(KSLy, GSLy, RSLy):
    m = monotrope(K * c_squared, G)
    tropes.append( m )

    #correct transition depths to avoid jumps
    if not(pm == None):
        rho_tr = (m.K / pm.K )**( 1.0/( pm.G - m.G ) )
    else:
        rho_tr = r
    pm = m

    trans.append(rho_tr)

#Create crust using polytrope class
SLyCrust = polytrope(tropes, trans)

##################################################
# BPS crust

class BPS_crust:
    def __init__(self):
        with open('bps_negele_mev_edited.dat', 'r') as f:
            eos_data = f.readlines()

            confac = cgs.GeVfm_per_dynecm * 0.001
            eos_data_len = len(eos_data)

            list_press = [0] * eos_data_len
            list_edens = [0] * eos_data_len
            list_rho = [0] * eos_data_len
            for i, line in enumerate(eos_data):
                line_mod = line.strip().split()

                list_press[eos_data_len-1-i] = float(line_mod[1]) * confac  # Ba
                list_edens[eos_data_len-1-i] = float(line_mod[2]) * confac * c2_inv  # g/cm^3
                list_rho[eos_data_len-1-i] = float(line_mod[3]) * cgs.mB * 1.e39  # g/cm^3

        list_speed2 = np.gradient(list_press, list_edens)
        list_speed2 = [item * c2_inv for item in list_speed2]

        self.pressure_interp = interp1d(list_rho, list_press)
        self.edens_interp = interp1d(list_rho, list_edens)
        self.edens_inv_interp = interp1d(list_press, list_edens)
        self.rho_interp = interp1d(list_press, list_rho)
        self.speed2_interp = interp1d(list_press, list_speed2)
        self.speed2_rho_interp = interp1d(list_rho, list_speed2)
        self.pressure_edens_interp = interp1d(list_edens, list_press)


        self.rho0 = list_rho[0]
        self.edens0 = list_edens[0]
        self.press0 = list_press[0]

        g = 1.0 + self.press0 * c2_inv / (self.edens0 - self.rho0)
        k = self.press0 * self.rho0**(-g)

        mono = monotrope(k, g)
        self.poly = polytrope([mono], [0.0])

    def pressure(self, rho):
        if rho < self.rho0:
            return self.poly.pressure(rho)
        return self.pressure_interp(rho)

    def pressures(self, rhos):
        press = []
        for rho in rhos:
            pr = self.pressure(rho)
            press.append( pr )
        return press

    def edens(self, rho):
        if rho < self.rho0:
            return self.poly.edens(rho)
        return self.edens_interp(rho)

    def edens_inv(self, press):
        if press < self.press0:
            return self.poly.edens_inv(press)
        return self.edens_inv_interp(press)

    def rho(self, press):
        if press < self.press0:
            return self.poly.rho(press)
        return self.rho_interp(press)

    def speed2(self, press):
        if press < self.press0:
            return self.poly.speed2(press)
        return self.speed2_interp(press)

    def speed2_rho(self, rho):
        if rho < self.rho0:
            return self.poly.speed2_rho(rho)
        return self.speed2_rho_interp(rho)

    def pressure_edens(self, edens):
        if edens < self.edens0:
            return self.pressure_edens(edens)
        return self.pressure_edens_interp(edens)

    def gammaFunction(self, rho, flag = 1):
        press = self.pressure(rho)
        edens = self.edens(rho) * c_squared
        speed2 = self.speed2_rho(rho)

        if flag == 1: # d(ln p)/d(ln n)
            return ( edens / press + 1.0 ) * speed2
        else: # d(ln p)/d(ln eps)
            return edens / press * speed2

    def tov(self, press, length=2):
        if length > 0:
            eden = self.edens_inv(press)
            res = [eden]
        if length > 1:
            speed2inv = 1.0 / self.speed2(press)
            res.append(speed2inv)
        if length > 2:
            rho = self.rho(press)
            res.append(rho)

        return res

    def press_edens_c2(self, rho):
        press = self.pressure(rho)
        eden = self.edens(rho)
        speed2 = self.speed2_rho(rho)

        return press, eden, speed2

