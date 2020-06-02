# multi-polytrope

## installation
 - Install `multinest` from https://github.com/JohannesBuchner/MultiNest
    - `cd build` `cmake ..` `make`
    - add `our/multinest/path/Multinest/lib` to `LD_LIBRARY_PATH`variable
 - Install pymultinest
    - `pip install pymultinest`
 - Install `mpi4py` for running the sampler in parallel


## usage

Run sampler with

```bash
mpirun -n X python3 multin_poly_ceft.py --nsteps 100 --dir chains --nseg 4 --ngrid 50
```

See `input_parser.py` for more run time options.




---------
# Old notes

## UPDATES

- [x] Implement QMC crust to `crust.py`
    - This should be just a simple wrapper to transform `L`, `S`, `a`, and `b` to polytropic indices and then using `monotrope`+`polytrope` classes to make it into an eos?
    - UPDATES:
      - One should use `b` and `beta` instead of `L` and `S` 
      - New doubleMonotrope class (works like polytrope)
    
- [X] pQCD functions to their own file `pQCD.py`?
    - Should just give constraints for the last monotrope in `polytrope` collection so that it satisfies `X`?
    - UPDATES:
      - `pQCD.py` file
      - `matchPolytopesWithLimits` class calculates the unknown gammas
      - `qcd` class acts like polytrope class


## TODO

- [ ] Fix numerical problems
    - Associated w/ comment ONGELMA (too large values of gamma1?)
    - Do we get all realistic gammas?

- [ ] Check calculations/formulas
    - Especially doubleMonotrope class

- [ ] Include M-R observations

- [ ] Include tidal diformability constrains (?)
    - How one should do this?


## Tegner installation


load modules and python
```
source emcee.sh
python --version
```

install emcee
```
git clone https://github.com/dfm/emcee.git
cd emcee/
python setup.py install --prefix=$LUSTREDIR
```


install schwimmbad (mpi)
```
git clone https://github.com/adrn/schwimmbad.git
cd schwimmbad/
python setup.py install --prefix=$LUSTREDIR
```

install hdf5
```
git clone https://github.com/h5py/h5py.git
export CC=gcc
python setup.py build
python setup.py install --prefix=/cfs/klemming/nobackup/j/jnattila
```

