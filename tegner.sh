module load git/2.3.0
module load gcc/6.2.0
module load hdf5/1.10.0-gcc-6.2
module load anaconda/py36/5.0.1

export CXX=mpicxx
export CC=mpicc
export FC=mpif90

export PYTHON_EXECUTABLE=python3

export LUSTREDIR=/cfs/klemming/nobackup/j/jnattila
export MCMC=$LUSTREDIR/multi-polytrope

export PYTHONPATH=$PYTHONPATH:/cfs/klemming/nobackup/j/jnattila/lib/python3.6/site-packages

export PYTHONPATH=$PYTHONPATH:$MCMC

export PYTHONDONTWRITEBYTECODE=true
export HDF5_USE_FILE_LOCKING=FALSE

source activate custom
