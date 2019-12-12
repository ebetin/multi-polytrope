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
            default=1,
            type=int,
            help='Random number generator seed (default: 1)')


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
            help='Debug mode(default: False)')

    parser.add_argument('--ngrid', 
            dest='ngrid', 
            default=200,
            type=int,
            help='Number of grid points for M/R/P grids (default: 200)')

    args = parser.parse_args()




    return args




