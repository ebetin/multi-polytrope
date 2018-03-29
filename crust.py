from polytropes import monotrope
from polytropes import polytrope
import units as cgs


##################################################
#SLy (Skyrme) crust
KSLy = [6.80110e-9, 1.06186e-6, 5.32697e1, 3.99874e-8] #Scaling constants
GSLy = [1.58425, 1.28733, 0.62223, 1.35692] #polytropic indices
RSLy = [1.e4, 2.44034e7, 3.78358e11, 2.62780e12 ] #transition depths

tropes = []
trans = []

pm = None
for (K, G, r) in zip(KSLy, GSLy, RSLy):
    m = monotrope(K*cgs.c**2, G)
    tropes.append( m )

    #correct transition depths to avoid jumps
    if not(pm == None):
        rho_tr = (m.K / pm.K )**( 1.0/( pm.G - m.G ) )
        #print rho_tr, np.log10(rho_tr), r, rho_tr/r
    else:
        rho_tr = r
    pm = m

    trans.append(rho_tr)

#Create crust using polytrope class
SLyCrust = polytrope(tropes, trans)


#Gandolf's crust #TODO beta stability
rhoo_tr_Gandolfi = 0.2 * 0.16 # (1/fm^3)
a = 13.4e6 * cgs.eV#12.7e6 * cgs.eV
alpha = 0.514#0.49
b = 5.62e6 * cgs.eV#1.78e6 * cgs.eV
beta = 2.436 #2.26
ns = 0.16 * 1.0e39 * cgs.mn

tropesAlpha = []
tropesBeta = []
transAlpha = []
transBeta = []

mAlpha = monotrope(a * alpha / (cgs.mn * ns**alpha), alpha+1.0)
mBeta = monotrope(b * beta / (cgs.mn * ns**beta), beta+1.0)

mAlpha.a = -0.5
mBeta.a = -0.5

tropesAlpha.append( mAlpha )
tropesBeta.append( mBeta )

transAlpha.append(rhoo_tr_Gandolfi * 1.0e39 * cgs.mn)
transBeta.append(rhoo_tr_Gandolfi * 1.0e39 * cgs.mn)

GandoliCrustAlpha = polytrope(tropesAlpha, transAlpha)
GandoliCrustBeta = polytrope(tropesBeta, transBeta)
