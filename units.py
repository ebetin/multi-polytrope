

##################################################
#Physical conversion factors 

#Particle Data Group (Review of particle Physics 2015) values
# in SI units
#c   = 2.99792458e8 #speed of light
#h   = 6.62607004081e-34 #Planck constant
#G   = 6.6730831e-11 #gravitational constant
eVSI = 1.602176620898e-19 # electron volt/Joule
pcSI = 3.08567758149e16   # parsec in m
RsSI = 2.9532500772e3     # Sch radius of Sun in m
Rs   = 2.9532500772       # Sch radius of Sun in m


c    = 2.99792458e10    # speed of light
G    = 6.6730831e-8     # gravitational constant
me   = 9.1093897e-28    # mass of electron
mp   = 1.6726231e-24    # mass of proton
mn   = 1.6749286e-24    # mass of neutron
mmu  = 1.8835327e-25    # mass of muon
kB   = 1.380658e-16     # Boltzmann constant
hP   = 6.6260755e-27    # Planck cosntant
hbar = 1.05457266e-27   # reduced Planck constant
eC   = 4.8032068e-10    # elementary charge
mH   = 1.6733e-24       # mass of hydrogen atom
eV   = 1.602177e-12     # electron volt
pc   = 3.08567758149e18 # parsec


#plasma electron relativity temperature
Tr = me*c**2/kB

#neutron plasma relativity temperature
Tn = mn*c**2/kB

#electron Compton wavelength
lambdaC = hbar/me/c

#plasma reference pressure
Pr = me * c**2 / lambdaC**3

#neutron Compton wavelength
lambdaCn = hbar/mn/c

#Neutron reference pressure
Prn = mn * c**2 / lambdaCn**3



#Other conversion factors
kelvin_per_keV = 1.16045e7
erg_per_kev = 1.0e-10 / eVSI


GeVfm_per_dynecm = 1.e9 * eV / (1.0e-13)**3

GeV3_to_fm3 = 1.0e3/1.9732705**3 #GeV^3->fm^-3 (approx.)



##################################################
# Operate in units where G = c = 1.
# Use units of solar masses

##################################################
# (G Msun /c^2 )^-1
solar_mass_per_km = 2.0e3/Rs

# (G Msun/c^3)^-1
solar_mass_per_s = 2.0*c/Rs



##################################################
# other conversion factors
km_per_kpc = pcSI #km / kpc = m/pc
cm_per_tenkpc = 1.0e-6 / pcSI


# Blackbody constant
# hzkeV^4 * 2*h/c^2
#constbb = 1.0e15 * (eV/hP)**4 * 2.0*hP/c**2




# other constants
##################################################
#mass of sun
Msun   = 1.988435e33
#kg_per_Msun    = 1.988435e30

#baryon mass
mB = mn

#mass saturation density
rhoS = 0.16e39 * mB

#nuclear binding energy at the saturation density
Enuc = 16.0e6 * eV

#computational constants
##################################################
epsilonGamma = 1.0e-4
