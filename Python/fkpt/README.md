# FKPT Python Implementation

This directory implements a python module called `fkpt` that reproduces the calculations in the C function `k_functions()` defined in `kfunctions.c`.

The `trace` branch of https://github.com/dkirkby/fkpt adds optional timing measurements and HDF5 output to the C code.

To create an HDF5 containing both the inputs required by `k_functions()` and its outputs for validation, use:
```bash
./fkpt Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat dumpKfunctions=kfunctions_snapshot_new.h5
```

To measure the timing of the C code, use:
```bash
./fkpt chatty=1 Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat
```
and look for the line:
```
  Total k-loop time:  128.480 ms, avg per k:   1.071 ms
```

To validate the python code against the HDF5 and measure its timing use:
```bash
cd Python
python -m fkpt.test
```

Summary of timing results:

| Platform        | C     | numpy |
|-----------------|-------|-------|
| Apple M1 Max    | 128ms | 102ms |
