# MultiNest usage notes


## serial mode

```bash
python3 multin.py
```

## parallel mode

```bash
mpirun -n 2 python3 multin_mpi.py
```

## analyzing results

Then call `multinest_marginals.py` as

```bash
python3 multinest_marginals.py chains/2-
```



