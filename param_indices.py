def blob_indices(parameter_indices, eosmodel = 0, flag_TOV = True, flag_GW = True, flag_Mobs = True, flag_MRobs = True, flag_baryonic_mass=True, flag_TD=False):
    """
    blob_indices does generate two list that contain information about the blob parameters

    :param parameter_indices:   Initial version of the return parameter 'param'. Mainly grid data
    :param eosmodel:            Interpolation model, either 0=polytropic (default) or
                                1=speed-of-sound
    :param flag_TOV:            If False, then all parameters that need information from the TOV
                                equations are excluded (default: True)
    :param flag_GW:             If True, then gravitational-wave (GW) observation related
                                parameters are included (default: True)
    :param flag_Mobs:           If True, then mass parameters corresponding to given radio mass
                                (M) measurements are indluded (default: True)
    :param flag_MRobs:          If True, then mass parameters corresponding to given mass-radius
                                (MR) measurements are indluded (default: True)
    :param flag_baryonic_mass:  This should be True (default), if baryonic masses have been calculated
    :param flag_TD:             True only if one wants to save tidal deformability (TD) data
                                (default: False)
    :return param2:             a list containing names of the blob parameters
    :return param:              a dictionary mapping parameter names and their numerical values
    """
    param = parameter_indices
    param2 = []
    ci = 0

    #-------------------------------------------------------------
    # add M-R grid
    # saves radius values (km) for given mass grid ('mass_grid')
    if flag_TOV:
        for im, mass  in enumerate(param['mass_grid']):
            param2.append('rad_'+str(im))
            param['rad_'+str(im)] = ci
            ci += 1

    #-------------------------------------------------------------
    # add eps-P grid
    # saves pressure values (MeV/fm^3) for given energy-density grid ('eps_grid')
    for ir, eps  in enumerate(param['eps_grid']):
        param2.append('Peps_'+str(ir))
        param['Peps_'+str(ir)] = ci
        ci += 1

    #-------------------------------------------------------------
    # add nsat - p grid
    # saves pressure values (MeV/fm^3) for given number-density grid ('nsat_long_grid')
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_p_'+str(ir))
        param['nsat_p_'+str(ir)] = ci
        ci += 1

    #-------------------------------------------------------------
    # add nsat - eps grid
    # saves energy-density values (MeV/fm^3) for given number-density grid ('nsat_long_grid')
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_eps_'+str(ir))
        param['nsat_eps_'+str(ir)] = ci
        ci += 1

    #-------------------------------------------------------------
    # add nsat - gamma grid
    # saves gamma (polytropic exponent) values (unitless) for given number-density grid ('nsat_long_grid')
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_gamma_'+str(ir))
        param['nsat_gamma_'+str(ir)] = ci
        ci += 1

    #-------------------------------------------------------------
    # add nsat - c^2 grid
    # saves squared-speed-of-sound values (unitless) for given number-density grid ('nsat_long_grid')
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_c2_'+str(ir))
        param['nsat_c2_'+str(ir)] = ci
        ci += 1

    #-------------------------------------------------------------
    # add nsat - press grid
    # saves normalized pressures (unitless) for given number-density grid ('nsat_long_grid')
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_press_'+str(ir))
        param['nsat_press_'+str(ir)] = ci
        ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved
    if flag_TOV:
        #-------------------------------------------------------------
        # add nsat - mass grid
        # saves mass values (Msun) for given number-density grid ('nsat_short_grid')
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_mass_'+str(ir))
            param['nsat_mass_'+str(ir)] = ci
            ci += 1
        #-------------------------------------------------------------
        # add nsat - radius grid
        # saves radius values (km) for given number-density grid ('nsat_short_grid')
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_radius_'+str(ir))
            param['nsat_radius_'+str(ir)] = ci
            ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved together with
    # tidal deformability (TD) data, AND one wants to save TD data
    if flag_TOV and flag_GW and flag_TD:
        #-------------------------------------------------------------
        #   add nsat - TD grid
        # saves TD values (unitless) for given number-density grid ('nsat_short_grid')
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_TD_'+str(ir))
            param['nsat_TD_'+str(ir)] = ci
            ci += 1

        #-------------------------------------------------------------
        # add M-TD grid
        # saves TD values (unitless) for given mass grid ('mass_grid')
        for im, mass  in enumerate(param['mass_grid']):
            param2.append('TD_'+str(im))
            param['TD_'+str(im)] = ci
            ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved
    if flag_TOV:
        # add parameters for the TOV configuration
        # maximum mass (Msun)
        param2.append('mmax')
        param['mmax'] = ci
        ci += 1
        # corresponding radius (km)
        param2.append('mmax_rad')
        param['mmax_rad'] = ci
        ci += 1
        # central number density (ns)
        param2.append('mmax_rho')
        param['mmax_rho'] = ci
        ci += 1
        # central pressure (MeV/fm^3)
        param2.append('mmax_press')
        param['mmax_press'] = ci
        ci += 1
        # central energy density (MeV/fm^3)
        param2.append('mmax_edens')
        param['mmax_edens'] = ci
        ci += 1
        # normalized central pressure (unitless)
        param2.append('mmax_ppFD')
        param['mmax_ppFD'] = ci
        ci += 1
        # central speed of sound squared (unitless)
        param2.append('mmax_c2')
        param['mmax_c2'] = ci
        ci += 1
        # central polytropic exponent (unitless)
        param2.append('mmax_gamma')
        param['mmax_gamma'] = ci
        ci += 1

    ##############################################################
    # crust-core transition (mass) density (g/cm^3)
    param2.append('rho_cc')
    param['rho_cc'] = ci
    ci += 1

    #-------------------------------------------------------------
    # max squared speed of sound of the given EoS (unitless)
    param2.append('c2max')
    param['c2max'] = ci
    ci += 1

    ##############################################################
    # Solved interpolation parameters by the code
    if eosmodel == 0: # polytrope
        # first polytropic exponent, gamma (unitless)
        param2.append('gamma1')
        param['gamma1'] = ci
        ci += 1

        # second gamma (unitless)
        param2.append('gamma2')
        param['gamma2'] = ci
        ci += 1
    elif eosmodel == 1: # speed of sound
        # solved chemical potential (Gev)
        param2.append('mu_param')
        param['mu_param'] = ci
        ci += 1

        # solved squared speed of sound (unitless)
        param2.append('c2_param')
        param['c2_param'] = ci
        ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved
    # radii (km), masses from radio observations
    if flag_TOV:
        # NB one is using mass measurements
        if flag_Mobs:
            # PSR J0348+0432 (~2.01), arXiv:1304.6875
            param2.append('r0348')
            param['r0348'] = ci
            ci += 1
        # NB one is using mass and/or mass-radius measurements
        if flag_Mobs or flag_MRobs:
            # PSR J0740+6620 (~2.08), arXiv:2104.00880
            param2.append('r0740')
            param['r0740'] = ci
            ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved
    # AND MR measurement are used
    if flag_TOV and flag_MRobs:
        # Radii (km) of MR measurements
        # 4U 1702-429, arXiv:1709.09120
        param2.append('r1702')
        param['r1702'] = ci
        ci += 1
        # NGC 6304, helium atmosphere, arXiv:1709.05013
        param2.append('r6304')
        param['r6304'] = ci
        ci += 1
        # NGC 6397, helium atmosphere, arXiv:1709.05013
        param2.append('r6397')
        param['r6397'] = ci
        ci += 1
        # M28, helium atmosphere, arXiv:1709.05013
        param2.append('rM28')
        param['rM28'] = ci
        ci += 1
        # M30, hydrogen atmosphere, arXiv:1709.05013
        param2.append('rM30')
        param['rM30'] = ci
        ci += 1
        # 47 Tuc X7, hydrogen atmosphere, arXiv:1709.05013
        param2.append('rX7')
        param['rX7'] = ci
        ci += 1
        # wCen, hydrogen atmosphere, arXiv:1709.05013
        param2.append('rwCen')
        param['rwCen'] = ci
        ci += 1
        # M13, hydrogen atmosphere, arXiv:1803.00029
        param2.append('rM13')
        param['rM13'] = ci
        ci += 1
        # 4U 1724-307, arXiv:1509.06561
        param2.append('r1724')
        param['r1724'] = ci
        ci += 1
        # SAX J1810.8-260, arXiv:1509.06561
        param2.append('r1810')
        param['r1810'] = ci
        ci += 1
        # J0030+0451, arXiv:1912.05705
        param2.append('r0030')
        param['r0030'] = ci
        ci += 1

    ##############################################################
    # these items are saved only if the TOV equations are solved together with
    # tidal deformability (TD) data
    if flag_TOV and flag_GW:
        # AND one wants to save TD data
        if flag_TD:
            # Tidal deformabilities (TD; unitless)
            # TD of the TOV NS
            param2.append('mmax_TD')
            param['mmax_TD'] = ci
            ci += 1
            # TD of the heavier NS in GW170817
            param2.append('GW170817_TD1')
            param['GW170817_TD1'] = ci
            ci += 1
            # TD of the lighter NS in GW170817
            param2.append('GW170817_TD2')
            param['GW170817_TD2'] = ci
            ci += 1

        #-------------------------------------------------------------
        # Radii (km)
        # radii of the heavier NS in GW170817
        param2.append('GW170817_r1')
        param['GW170817_r1'] = ci
        ci += 1
        # radii of the lighter NS in GW170817
        param2.append('GW170817_r2')
        param['GW170817_r2'] = ci
        ci += 1

        #-------------------------------------------------------------
        # if one has also calculated baryonic masses
        if flag_baryonic_mass:
            # Baryonic masses (M_sun)
            # for the TOV star
            param2.append('mmax_B')
            param['mmax_B'] = ci
            ci += 1
            # for the heavier NS in GW170817 system
            param2.append('GW170817_mB1')
            param['GW170817_mB1'] = ci
            ci += 1
            # for the heavier NS in GW170817 system
            param2.append('GW170817_mB2')
            param['GW170817_mB2'] = ci
            ci += 1

    return param2, param

def param_names(eosmodel, ceftmodel, eos_Nseg, pt = 0, latex = False, flag_TOV = True, flag_GW = True, flag_Mobs = True, flag_MRobs = True, flag_const_limits = False):
    """
    param_names auto-generates parameter names to a list

    :param eosmodel:            Interpolation model, either 0=polytropic or 1=speed-of-sound
    :param ceftmodel:           Low-density, chiral-effective-field-theory model
    :param eos_Nseg:            Number of EoS segments
    :param pt:                  Point where a forced 1st order phase transition (pt) happens
                                (default: 0 = no pt)
    :param latex:               If True, then parameter names are writen using laTex notation.
                                This is useful with plot script (default: False)
    :param flag_TOV:            If False, then all parameters that need information from the TOV
                                equations are excluded (default: True)
    :param flag_GW:             If True, then gravitational-wave observation related parameters
                                are included (default: True)
    :param flag_Mobs:           If True, then mass parameters corresponding to given radio mass
                                measurements are indluded (default: True)
    :param flag_MRobs:          If True, then mass parameters corresponding to given mass-radius
                                measurements are indluded (default: True)
    :param flag_const_limits:   If the flag param is True, then predetermined low- and
                                high-density EoS models have been used (default: False)
    :return parameters:         a list containing parameter names that are used in the dataset
    """

    parameters = []

    if not flag_const_limits:  # constant limit = fixed low- and high-denisty EoS
        # cEFT parameters (all are unitless)
        if ceftmodel == 'HLPS':  # original, Hebeler et al. 2013 (arXiv:1303.4662) model
            parameters = [r"$\alpha_L$", r"$\eta_L$"] if latex else ["alphaL", "etaL"]
        elif ceftmodel == 'HLPS3':  # same but 'gamma' parameter is also included
            parameters = [r"$\alpha_L$", r"$\eta_L$", r"$\gamma_L$"] if latex else ["alphaL", "etaL", "gamma"]
        elif ceftmodel == 'HLPS+':  # our extension with a new term (meaning two new params)
            parameters = [r"$\alpha_L$", r"$\eta_L$", r"$\gamma_L$", r"$\zeta_L$", r"$\bar{\rho}_0$"] if latex else ["alphaL", "etaL", "gammaL", "zetaL", "rho0"]

        # pQCD parameter, see e.g. Fraga et al. (2014, arXiv:1311.5154) for details
        # the parameter is unitless
        if latex:
            parameters.append(r"$X$")
        else:
            parameters.append("X")

    # interpolation parameters
    # eosmodel: 0 (piecewise polytropes) and 1 (piecewise linear c_s^2)
    if eosmodel == 0:
        # append gammas, i.e. polytropic expotent (unitless)
        for itrope in range(eos_Nseg-2):
            if itrope + 1 != pt:
                if latex:
                    parameters.append(r"$\gamma_{{{0}}}$".format((3+itrope)))
                else:
                    parameters.append("gamma"+str(3+itrope))

        # append transition densities/verticies [unit: ns=saturation density~0.16/fm^3]
        for itrope in range(eos_Nseg-1):
            if latex:
                #parameters.append(r"$\Delta n_{{{0}}}$".format(1+itrope))
                parameters.append(r"$n_{{t,{0}}}$".format(1+itrope))
            else:
                #parameters.append("trans_delta"+str(1+itrope))
                parameters.append("trans"+str(1+itrope))

    elif eosmodel == 1:
        # append chemical potentials at transition points [GeV]
        # (NB last one will be calculated by the code)
        for itrope in range(eos_Nseg-2):
            if latex:
                parameters.append(r"$\mu_{{{0}}}$".format((1+itrope)))
            else:
                parameters.append("mu"+str(1+itrope))

        # append speed of sound squared (unitless)
        # (NB last one will be calculated by the code)
        for itrope in range(eos_Nseg-2):
            if latex:
                parameters.append(r"$c^2_{{{0}}}$".format(1+itrope))
            else:
                parameters.append("speed"+str(1+itrope))

    # GW170817 event
    if flag_GW and flag_TOV:
        if latex:
            # chirp mass [solar mass]
            parameters.append(r"$\mathcal{M}_{GW170817}$")
            # mass ratio, q \in [0,1] (unitless)
            parameters.append(r"$q_{GW170817}$")
        else:
            parameters.append("chrip_mass_GW170817")
            parameters.append("mass_ratio_GW170817")

    # mass measurements in solar masses
    if flag_TOV:
        if flag_Mobs:
            if latex:
                # PSR J0348+0432 (~2.01), arXiv:1304.6875
                parameters.append(r"$M_{0348}$")
            else:
                parameters.append("mass_0348")

        if flag_MRobs or flag_Mobs:
            # PSR J0740+6620 (~2.08), arXiv:2104.00880
            if latex:
                parameters.append(r"$M_{0740}$")
            else:
                parameters.append("mass_0740")

    # mass-radius measurements, mass parameters in solar masses
    if flag_TOV and flag_MRobs:
        if latex:
            # 4U 1702-429, arXiv:1709.09120
            parameters.append(r"$M_{1702}$")
            # NGC 6304, helium atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{6304}$")
            # NGC 6397, helium atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{6397}$")
            # M28, helium atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{M28}$")
            # M30, hydrogen atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{M30}$")
            # 47 Tuc X7, hydrogen atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{X7}$")
            # wCen, hydrogen atmosphere, arXiv:1709.05013
            parameters.append(r"$M_{\omega Cen}$")
            # M13, hydrogen atmosphere, arXiv:1803.00029
            parameters.append(r"$M_{M13}$")
            # 4U 1724-307, arXiv:1509.06561
            parameters.append(r"$M_{1724}$")
            # SAX J1810.8-260, arXiv:1509.06561
            parameters.append(r"$M_{1810}$")
            # J0030+0451, arXiv:1912.05705
            parameters.append(r"$M_{0030}$")
        else:
            parameters.append("mass_1702")
            parameters.append("mass_6304")
            parameters.append("mass_6397")
            parameters.append("mass_M28")
            parameters.append("mass_M30")
            parameters.append("mass_X7")
            parameters.append("mass_wCen")
            parameters.append("mass_M13")
            parameters.append("mass_1724")
            parameters.append("mass_1810")
            parameters.append("mass_0030")

    return parameters
