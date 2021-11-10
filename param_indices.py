def blob_indices(parameter_indices, eosmodel = 0, flag_TOV = True, flag_GW = True, flag_Mobs = True, flag_MRobs = True, flag_baryonic_mass=True, flag_TD=False):
    param = parameter_indices
    param2 = []
    ci = 0

    if flag_TOV:
        #add M-R grid
        for im, mass  in enumerate(param['mass_grid']):
            param2.append('rad_'+str(im))
            param['rad_'+str(im)] = ci
            ci += 1

    #add eps-P grid
    for ir, eps  in enumerate(param['eps_grid']):
        param2.append('Peps_'+str(ir))
        param['Peps_'+str(ir)] = ci
        ci += 1

    #add nsat - p grid
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_p_'+str(ir))
        param['nsat_p_'+str(ir)] = ci
        ci += 1

    #add nsat - eps grid
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_eps_'+str(ir))
        param['nsat_eps_'+str(ir)] = ci
        ci += 1

    #add nsat - gamma grid
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_gamma_'+str(ir))
        param['nsat_gamma_'+str(ir)] = ci
        ci += 1

    #add nsat - c^2 grid
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_c2_'+str(ir))
        param['nsat_c2_'+str(ir)] = ci
        ci += 1

    #add nsat - press grid
    for ir, nsat  in enumerate(param['nsat_long_grid']):
        param2.append('nsat_press_'+str(ir))
        param['nsat_press_'+str(ir)] = ci
        ci += 1

    if flag_TOV:
        #add nsat - mass grid
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_mass_'+str(ir))
            param['nsat_mass_'+str(ir)] = ci
            ci += 1

        #add nsat - radius grid
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_radius_'+str(ir))
            param['nsat_radius_'+str(ir)] = ci
            ci += 1

    if flag_TOV and flag_GW and flag_TD:
        #add nsat - TD grid
        for ir, nsat  in enumerate(param['nsat_short_grid']):
            param2.append('nsat_TD_'+str(ir))
            param['nsat_TD_'+str(ir)] = ci
            ci += 1

        #add M-TD grid
        for im, mass  in enumerate(param['mass_grid']):
            param2.append('TD_'+str(im))
            param['TD_'+str(im)] = ci
            ci += 1

    if flag_TOV:
        #add mmax parameters
        param2.append('mmax')
        param['mmax'] = ci
        ci += 1
        param2.append('mmax_rad')
        param['mmax_rad'] = ci
        ci += 1
        param2.append('mmax_rho')
        param['mmax_rho'] = ci
        ci += 1
        param2.append('mmax_press')
        param['mmax_press'] = ci
        ci += 1
        param2.append('mmax_edens')
        param['mmax_edens'] = ci
        ci += 1
        param2.append('mmax_ppFD')
        param['mmax_ppFD'] = ci
        ci += 1
        param2.append('mmax_c2')
        param['mmax_c2'] = ci
        ci += 1
        param2.append('mmax_gamma')
        param['mmax_gamma'] = ci
        ci += 1

    # crust-core transition (mass) density (g/cm^3)
    param2.append('rho_cc')
    param['rho_cc'] = ci
    ci += 1

    # max squared speed of sound
    param2.append('c2max')
    param['c2max'] = ci
    ci += 1

    if eosmodel == 0:
        # first gamma
        param2.append('gamma1')
        param['gamma1'] = ci
        ci += 1

        # second gamma
        param2.append('gamma2')
        param['gamma2'] = ci
        ci += 1
    elif eosmodel == 1:
        # solved chemical potential (Gev)
        param2.append('mu_param')
        param['mu_param'] = ci
        ci += 1

        # solved squared speed of sound
        param2.append('c2_param')
        param['c2_param'] = ci
        ci += 1

    if flag_TOV:
        if flag_Mobs:
            param2.append('r0348')
            param['r0348'] = ci
            ci += 1
        if flag_Mobs or flag_MRobs:
            param2.append('r0740')
            param['r0740'] = ci
            ci += 1

    if flag_TOV and flag_MRobs:
        # Radii
        param2.append('r1702')
        param['r1702'] = ci
        ci += 1
        param2.append('r6304')
        param['r6304'] = ci
        ci += 1
        param2.append('r6397')
        param['r6397'] = ci
        ci += 1
        param2.append('rM28')
        param['rM28'] = ci
        ci += 1
        param2.append('rM30')
        param['rM30'] = ci
        ci += 1
        param2.append('rX7')
        param['rX7'] = ci
        ci += 1
        param2.append('rwCen')
        param['rwCen'] = ci
        ci += 1
        param2.append('rM13')
        param['rM13'] = ci
        ci += 1
        param2.append('r1724')
        param['r1724'] = ci
        ci += 1
        param2.append('r1810')
        param['r1810'] = ci
        ci += 1
        param2.append('r0030')
        param['r0030'] = ci
        ci += 1

    if flag_TOV and flag_GW:
        if flag_TD:
            # Tidal deformabilities
            param2.append('mmax_TD')
            param['mmax_TD'] = ci
            ci += 1
            param2.append('GW170817_TD1')
            param['GW170817_TD1'] = ci
            ci += 1
            param2.append('GW170817_TD2')
            param['GW170817_TD2'] = ci
            ci += 1

        # Radii
        param2.append('GW170817_r1')
        param['GW170817_r1'] = ci
        ci += 1
        param2.append('GW170817_r2')
        param['GW170817_r2'] = ci
        ci += 1

        if flag_baryonic_mass:
            # Baryonic masses (M_sun)
            param2.append('mmax_B')
            param['mmax_B'] = ci
            ci += 1
            param2.append('GW170817_mB1')
            param['GW170817_mB1'] = ci
            ci += 1
            param2.append('GW170817_mB2')
            param['GW170817_mB2'] = ci
            ci += 1

    return param2, param

#auto-generated parameter names
def param_names(eosmodel, ceftmodel, eos_Nseg, pt = 0, latex = False, flag_TOV = True, flag_GW = True, flag_Mobs = True, flag_MRobs = True, flag_const_limits = False):
    parameters = []
    if not flag_const_limits:
        #cEFT parameters
        if ceftmodel == 'HLPS':
            parameters = [r"$\alpha_L$", r"$\eta_L$"] if latex else ["alphaL", "etaL"]
        elif ceftmodel == 'HLPS3':
            parameters = [r"$\alpha_L$", r"$\eta_L$", r"$\gamma_L$"] if latex else ["alphaL", "etaL", "gamma"]
        elif ceftmodel == 'HLPS+':
            parameters = [r"$\alpha_L$", r"$\eta_L$", r"$\gamma_L$", r"$\zeta_L$", r"$\bar{\rho}_0$"] if latex else ["alphaL", "etaL", "gammaL", "zetaL", "rho0"]

        #pQCD parameters
        if latex:
            parameters.append(r"$X$")
        else:
            parameters.append("X")

    #Interpolation parameters
    if eosmodel == 0:
        #append gammas
        for itrope in range(eos_Nseg-2):
            if itrope + 1 != pt:
                if latex:
                    parameters.append(r"$\gamma_{{{0}}}$".format((3+itrope)))
                else:
                    parameters.append("gamma"+str(3+itrope))

        #append transition depths
        for itrope in range(eos_Nseg-1):
            if latex:
                #parameters.append(r"$\Delta n_{{{0}}}$".format(1+itrope))
                parameters.append(r"$n_{{t,{0}}}$".format(1+itrope))
            else:
                #parameters.append("trans_delta"+str(1+itrope))
                parameters.append("trans"+str(1+itrope))

    elif eosmodel == 1:
        #append chemical potential depths (NB last one will be determined)
        for itrope in range(eos_Nseg-2):
            if latex:
                #parameters.append(r"$\Delta\mu_{{{0}}}$".format((1+itrope)))
                parameters.append(r"$\mu_{{{0}}}$".format((1+itrope)))
            else:
                #parameters.append("mu_delta"+str(1+itrope))
                parameters.append("mu"+str(1+itrope))

        #append speed of sound squared (NB last one will be determined)
        for itrope in range(eos_Nseg-2):
            if latex:
                parameters.append(r"$c^2_{{{0}}}$".format(1+itrope))
            else:
                parameters.append("speed"+str(1+itrope))

    # GW170817
    if flag_GW and flag_TOV:
        if latex:
            parameters.append(r"$\mathcal{M}_{GW170817}$")
            parameters.append(r"$q_{GW170817}$")
        else:
            parameters.append("chrip_mass_GW170817")
            parameters.append("mass_ratio_GW170817")

    # M measurements
    if flag_TOV:
        if flag_Mobs:
            if latex:
                parameters.append(r"$M_{0348}$")
            else:
                parameters.append("mass_0348")

        if flag_MRobs or flag_Mobs:
            if latex:
                parameters.append(r"$M_{0740}$")
            else:
                parameters.append("mass_0740")

    # MR measurements
    if flag_TOV and flag_MRobs:
        if latex:
            parameters.append(r"$M_{1702}$")
            parameters.append(r"$M_{6304}$")
            parameters.append(r"$M_{6397}$")
            parameters.append(r"$M_{M28}$")
            parameters.append(r"$M_{M30}$")
            parameters.append(r"$M_{X7}$")
            parameters.append(r"$M_{\omega Cen}$")
            parameters.append(r"$M_{M13}$")
            parameters.append(r"$M_{1724}$")
            parameters.append(r"$M_{1810}$")
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
