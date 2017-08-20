import numpy as np
from math import pi


import units as cgs
from polytropes import monotrope, polytrope
from crust import SLyCrust
from tov import tov


class structure:


    def __init__(self, gammas, Ks, transitions):

        # Create polytropic presentation 
        assert len(gammas) == len(Ks) == len(transitions)
        self.tropes = []
        self.trans  = []
        for i in range(len(gammas)):
            self.trope.append( monotrope(gammas[i], Ks[i]) )
            self.trans.append( transitions[i] )

        dense_eos = polytrope( tropes, trans )



    #solve TOV equations
    def tov(self):
        t = tov(self.eos)
        self.mass, self.rad, self.rho = t.mass_radius()
        self.maxmass = np.max( self.mass )







