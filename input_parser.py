import os
import argparse


def parse_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', '--nsteps', 
            dest='nsteps', 
            default=500,
            type=int,
            help='Number of steps (default: 500)')

    parser.add_argument('-d', '--dir', 
            dest='outputdir', 
            default="chains",
            type=str,
            help='Name of output dir (default: chains)')

    parser.add_argument('-s', '--seed', 
            dest='seed', 
            default=0,
            type=int,
            help='Random number generator seed (default: 0 = random)')

    parser.add_argument('-n', '--nseg', 
            dest='eos_nseg', 
            default=4,
            type=int,
            help='Number of segments/polytropes (default: 4)')

    parser.add_argument('-p', '--ptrans', 
            dest='ptrans', 
            default=0,
            type=int,
            help='Phase transition location (default: 0)')

    parser.add_argument('--debug', 
            dest='debug', 
            default=False,
            type=bool,
            help='Debug mode (default: False)')

    parser.add_argument('--ngrid', 
            dest='ngrid', 
            default=200,
            type=int,
            help='Number of grid points for M/R/P grids (default: 200)')

    parser.add_argument('-m','--model', 
            dest='model', 
            default=0,
            type=int,
            help='Interpolation model [poly = 0 (default), c2 = 1]')

    parser.add_argument('-w','--walkers', 
            dest='walkers', 
            default=2,
            type=int,
            help='Walker multiplier (default: 2)')

    parser.add_argument('--subconf', 
            dest='subconf',
            default=False,
            type=bool,
            help='Discarding subconformal (c_s^2 > 1/3) EoSs (default: False)')

    parser.add_argument('-c','--ceft',
            dest='ceft',
            default='HLPS+',
            type=str,
            help='cEFT model [HLPS, HLPS3, HLPS+ (default)]')

    parser.add_argument('-x','--xmodel', 
            dest='xmodel', 
            default='uniform',
            type=str,
            help='Distribution for the pQCD parameter X [uniform (default), log-uniform, log-normal]')

    parser.add_argument('--new',
            dest='new_run',
            default=True,
            type=int,
            help='Entirely new run? (Yes: 1 (default), No: 0)')

    args = parser.parse_args()




    return args




