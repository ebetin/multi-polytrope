#!/bin/bash
#SBATCH -A 2018-2-41
#SBATCH -J poly
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH -t 0-23:00:00
#SBATCH -n 1

# activate threading
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=true
export HDF5_USE_FILE_LOCKING=FALSE

source $HOME/emcee.sh

# go to working directory
cd $MCMC

srun -n 1 python mcmc.py
