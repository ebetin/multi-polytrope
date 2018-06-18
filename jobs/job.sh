#!/bin/bash
#SBATCH -A SNIC2018-5-16
#SBATCH -J eos
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH -t 02-02:10:00
#SBATCH -n 1

# export module libraries
export PYTHONPATH=$PYTHONPATH:/pfs/nobackup/home/n/natj/multi-polytrope

# activate threading
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# go to working directory
cd /pfs/nobackup/home/n/natj/multi-polytrope

# print useful intro
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"


mpirun -n 1 python sample.py
