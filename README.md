# multi-polytrope

## installation
 - Install `multinest` from https://github.com/JohannesBuchner/MultiNest
    - `cd build` `cmake ..` `make`
    - add `our/multinest/path/Multinest/lib` to `LD_LIBRARY_PATH`variable
 - Install pymultinest
    - `pip install pymultinest`
 - Optionally one can install `mpi4py` for running the sampler in parallel

## TODO

- [ ] Implement QMC crust to `crust.py`
    - This should be just a simple wrapper to transform `L`, `S`, `a`, and `b` to polytropic indices and then using `monotrope`+`polytrope` classes to make it into an eos?
    
- [ ] pQCD functions to their own file `pQCD.py`?
    - Should just give constraints for the last monotrope in `polytrope` collection so that it satisfies `X`?



