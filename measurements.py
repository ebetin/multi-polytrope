# define different measurements here


# non correlated 2D Gaussian 
def gaussian_MR(mass, rad, conf):
    rad_gauss  = -0.5*( (rad  - conf["rad_mean"]) /conf["rad_std" ])**2.0
    mass_gauss = -0.5*( (mass - conf["mass_mean"])/conf["mass_std"])**2.0

    return rad_gauss + mass_gauss

# Values from Nattila et al 2017 for 4U 1702-429
NSK17 = { "rad_mean": 12.4,
           "rad_std": 0.4,
         "mass_mean": 1.8,
          "mass_std": 0.2,
        }








