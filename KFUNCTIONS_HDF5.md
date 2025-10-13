# k-functions HDF5 Snapshot Documentation

## Overview

This document describes the HDF5 snapshot format for capturing all inputs and outputs of the `compute_kfunctions()` function in **fkpt**. This snapshot provides a complete, portable reference for implementing or testing alternative implementations of the k-functions computation in other programming languages.

## Purpose

The HDF5 snapshot captures:
1. **All input parameters** - Cosmology, model, numerical settings
2. **Input linear power spectra** - Both wiggle and no-wiggle versions
3. **Intermediate computational products** - Growth factors, kernel constants, sigma values
4. **All output k-functions** - Complete set of 1-loop SPT components for both wiggle and no-wiggle

This enables:
- **Reference testing** - Verify alternate implementations match the C reference
- **Cross-language development** - Implement k-functions in Python, Julia, Rust, etc.
- **Debugging** - Compare intermediate values when tracking down discrepancies
- **Documentation** - Self-contained snapshot with all metadata

## Building with HDF5 Support

### Prerequisites

Install HDF5 development libraries:

**Conda/Mamba (Recommended):**
```bash
conda install -c conda-forge hdf5
# or
mamba install -c conda-forge hdf5
```

**macOS (Homebrew):**
```bash
brew install hdf5
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libhdf5-dev
```

**CentOS/RHEL:**
```bash
sudo yum install hdf5-devel
```

### Compiling

The `Makefile` is already configured for conda environments. If you installed HDF5 via conda, the Makefile will automatically use `$CONDA_PREFIX` to find the headers and libraries.

**For conda/mamba users** (default in Makefile):
```bash
# Makefile already configured to use:
# CFLAGS = -g -O3 -DUSE_HDF5 -I$(CONDA_PREFIX)/include
# LDFLAGS = -lm -L$(CONDA_PREFIX)/lib -lhdf5

make clean
make
```

**For Homebrew on macOS**, edit the `Makefile` and uncomment:
```makefile
CFLAGS = -g -O3 -DUSE_HDF5 -I/usr/local/opt/hdf5/include $(OPTIONS)
LDFLAGS = -lm -L/usr/local/opt/hdf5/lib -lhdf5
```

**For system HDF5 on Linux**, edit the `Makefile` and uncomment:
```makefile
CFLAGS = -g -O3 -DUSE_HDF5 -I/usr/include/hdf5/serial $(OPTIONS)
LDFLAGS = -lm -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
```

## Usage

### Generating a Snapshot

Use the `dumpKfunctions` command-line parameter:

```bash
./fkpt Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 \
       fnamePS=pkl_z05.dat dumpKfunctions=kfunctions_snapshot.h5
```

This will:
1. Run the normal k-functions computation
2. Write a complete HDF5 snapshot to the specified file
3. Continue with the rest of the fkpt computation

### Verifying the Snapshot

Use the `h5dump` utility (comes with HDF5):

```bash
# View file structure
h5dump -n kfunctions_snapshot.h5

# View metadata
h5dump -g /metadata kfunctions_snapshot.h5

# View specific dataset
h5dump -d /outputs/kfunctions_wiggle/P22dd kfunctions_snapshot.h5
```

## HDF5 File Structure

```
kfunctions_snapshot.h5
├── /metadata                          # Run metadata
│   ├── timestamp (attribute)          # When snapshot was created
│   └── command_line (attribute)       # Command used to generate
│
├── /parameters                        # All input parameters
│   ├── /cosmology
│   │   ├── Om          [scalar]       # Matter density (z=0)
│   │   ├── h           [scalar]       # Hubble parameter
│   │   ├── zout        [scalar]       # Output redshift
│   │   ├── f0          [scalar]       # Growth rate at k→0
│   │   └── Dplus       [scalar]       # Growth factor D+(z)/D+(0)
│   │
│   ├── /model
│   │   ├── mgmodel     [string]       # Model name: LCDM, HS, DGP, etc.
│   │   └── fR0         [scalar]       # f(R) parameter (for HS model)
│   │
│   ├── /k_grid
│   │   ├── kmin        [scalar]       # Minimum k [h/Mpc]
│   │   ├── kmax        [scalar]       # Maximum k [h/Mpc]
│   │   └── Nk          [integer]      # Number of k-values in output
│   │
│   ├── /numerical
│   │   └── nquadSteps  [integer]      # Number of momentum integration steps
│   │
│   └── /kernels                       # LCDM kernel constants
│       ├── KA_LCDM     [scalar]       # Second-order kernel constant
│       ├── KAp_LCDM    [scalar]       # Derivative kernel constant
│       ├── KB_LCDM     [scalar]       # Alternate second-order constant
│       ├── KR1_LCDM    [scalar]       # Third-order kernel constant
│       └── KR1p_LCDM   [scalar]       # Derivative third-order constant
│
├── /sigma_values                      # Variance and damping integrals
│   ├── sigma8          [scalar]       # RMS fluctuation in 8 Mpc/h spheres
│   ├── sigma2L         [scalar]       # Linear displacement variance
│   ├── sigma2v         [scalar]       # Velocity dispersion variance
│   ├── Sigma2          [scalar]       # BAO damping scale
│   └── deltaSigma2     [scalar]       # Differential BAO damping
│
├── /inputs                            # Input linear power spectra
│   ├── linear_ps_wiggle    [nPSLT × 4]   # Columns: k, P(k), P''(k), f(k)
│   └── linear_ps_nowiggle  [nPSLT × 4]   # No-wiggle version
│
└── /outputs                           # Computed k-functions
    ├── /kfunctions_wiggle             # With BAO wiggles
    │   ├── k             [Nk]         # k-values [h/Mpc]
    │   ├── P22dd         [Nk]         # P22 density-density
    │   ├── P22du         [Nk]         # P22 density-velocity
    │   ├── P22uu         [Nk]         # P22 velocity-velocity
    │   ├── I1udd1A       [Nk]         # RSD A-term integrals
    │   ├── I2uud1A       [Nk]         # (5 A-terms total)
    │   ├── I2uud2A       [Nk]
    │   ├── I3uuu2A       [Nk]
    │   ├── I3uuu3A       [Nk]
    │   ├── I2uudd1BpC    [Nk]         # RSD D-term integrals (B+C-G)
    │   ├── I2uudd2BpC    [Nk]         # (7 D-terms total)
    │   ├── I3uuud2BpC    [Nk]
    │   ├── I3uuud3BpC    [Nk]
    │   ├── I4uuuu2BpC    [Nk]
    │   ├── I4uuuu3BpC    [Nk]
    │   ├── I4uuuu4BpC    [Nk]
    │   ├── Pb1b2         [Nk]         # Bias terms
    │   ├── Pb1bs2        [Nk]         # (7 bias terms)
    │   ├── Pb22          [Nk]
    │   ├── Pb2s2         [Nk]
    │   ├── Ps22          [Nk]
    │   ├── Pb2theta      [Nk]
    │   ├── Pbs2theta     [Nk]
    │   ├── P13dd         [Nk]         # P13 components
    │   ├── P13du         [Nk]
    │   ├── P13uu         [Nk]
    │   ├── sigma32PSL    [Nk]         # Velocity broadening term
    │   ├── pkl           [Nk]         # Linear P(k) at output k-grid
    │   └── fk            [Nk]         # Growth rate f(k) at output k-grid
    │
    └── /kfunctions_nowiggle           # Same 27 arrays for no-wiggle
        └── (same structure as wiggle)
```

## Array Dimensions

- **nPSLT**: Number of points in input linear power spectrum (typically ~800)
- **Nk**: Number of output k-values (set by `Nk` parameter, default 120)

All output arrays in `/outputs/kfunctions_*` have shape `[Nk]`.

Input power spectrum arrays have shape `[nPSLT, 4]` where columns are:
1. k [h/Mpc]
2. P(k) [(Mpc/h)³]
3. P''(k) - Second derivative from spline (for interpolation)
4. f(k) - Growth rate as function of k

## Units

| Quantity | Units | Notes |
|----------|-------|-------|
| k | h/Mpc | Comoving wavenumber |
| P(k) | (Mpc/h)³ | Power spectrum |
| Om | dimensionless | Ωₘ(z=0) |
| h | dimensionless | H₀ = 100h km/s/Mpc |
| fR0 | dimensionless | f(R₀) for Hu-Sawicky model |
| sigma8, sigma2L, etc. | (Mpc/h)² | Variance integrals |

## Reading Snapshots in Other Languages

### Python (h5py)

```python
import h5py
import numpy as np

with h5py.File('kfunctions_snapshot.h5', 'r') as f:
    # Read metadata
    timestamp = f['/metadata'].attrs['timestamp']

    # Read parameters
    Om = f['/parameters/cosmology/Om'][0]
    h = f['/parameters/cosmology/h'][0]

    # Read input power spectrum
    ps_wiggle = f['/inputs/linear_ps_wiggle'][:]  # shape: (nPSLT, 4)
    k_in = ps_wiggle[:, 0]
    P_in = ps_wiggle[:, 1]

    # Read output k-functions
    k_out = f['/outputs/kfunctions_wiggle/k'][:]
    P22dd = f['/outputs/kfunctions_wiggle/P22dd'][:]

    # Read sigma values
    sigma8 = f['/sigma_values/sigma8'][0]
```

### Julia (HDF5.jl)

```julia
using HDF5

h5open("kfunctions_snapshot.h5", "r") do file
    # Read parameters
    Om = read(file, "/parameters/cosmology/Om")[1]
    h = read(file, "/parameters/cosmology/h")[1]

    # Read input
    ps_wiggle = read(file, "/inputs/linear_ps_wiggle")
    k_in = ps_wiggle[:, 1]

    # Read output
    k_out = read(file, "/outputs/kfunctions_wiggle/k")
    P22dd = read(file, "/outputs/kfunctions_wiggle/P22dd")

    # Read metadata
    timestamp = read_attribute(file["/metadata"], "timestamp")
end
```

### Rust (hdf5-rust)

```rust
use hdf5::File;

let file = File::open("kfunctions_snapshot.h5")?;

// Read parameters
let Om = file.dataset("/parameters/cosmology/Om")?.read_scalar::<f64>()?;

// Read arrays
let ps_wiggle = file.dataset("/inputs/linear_ps_wiggle")?
    .read_2d::<f64>()?;

let k_out = file.dataset("/outputs/kfunctions_wiggle/k")?
    .read_1d::<f64>()?;
```

## Verification and Testing

### Comparison Workflow

1. **Generate reference snapshot** from C implementation
2. **Load snapshot** in alternate language
3. **Implement k-functions** using same numerical algorithms
4. **Compare outputs** element-wise with reference

### Tolerance Guidelines

Recommended relative tolerances for numerical comparison:

| Quantity | Relative Tolerance | Notes |
|----------|-------------------|-------|
| P22, P13 | 1e-6 | Main SPT components |
| RSD integrals (A, D) | 1e-5 | More numerically sensitive |
| Bias terms | 1e-6 | Usually stable |
| sigma values | 1e-8 | Global integrals |
| Kernel constants | 1e-10 | Exact differential equation solutions |

### Example Verification Script (Python)

```python
import h5py
import numpy as np

def compare_snapshots(ref_file, test_file, rtol=1e-6):
    """Compare two k-functions snapshots."""
    with h5py.File(ref_file, 'r') as ref, h5py.File(test_file, 'r') as test:
        # Compare all wiggle outputs
        group = '/outputs/kfunctions_wiggle'
        arrays_to_check = [
            'P22dd', 'P22du', 'P22uu',
            'P13dd', 'P13du', 'P13uu',
            'I1udd1A', 'I2uud1A', 'I2uud2A', 'I3uuu2A', 'I3uuu3A',
            'I2uudd1BpC', 'I2uudd2BpC', 'I3uuud2BpC', 'I3uuud3BpC',
            'I4uuuu2BpC', 'I4uuuu3BpC', 'I4uuuu4BpC',
            'Pb1b2', 'Pb1bs2', 'Pb22', 'Pb2s2', 'Ps22',
            'Pb2theta', 'Pbs2theta', 'sigma32PSL'
        ]

        results = {}
        for name in arrays_to_check:
            ref_data = ref[f'{group}/{name}'][:]
            test_data = test[f'{group}/{name}'][:]

            max_diff = np.max(np.abs((test_data - ref_data) / ref_data))
            results[name] = max_diff

            if max_diff > rtol:
                print(f"FAIL: {name} max relative diff = {max_diff:.2e}")
            else:
                print(f"PASS: {name} max relative diff = {max_diff:.2e}")

        return results

# Usage
results = compare_snapshots('reference.h5', 'my_implementation.h5')
```

## Implementation Notes

### Key Dependencies

The k-functions computation depends on:

1. **Linear power spectrum P(k)** - Must be pre-computed (e.g., from CLASS, CAMB)
2. **Growth factors** - Computed from differential equations in `models.c`
3. **Kernel functions** - Second and third-order SPT kernels (f(R) extensions)
4. **Momentum integrals** - Nested loop quadrature over p = kr

### Computational Bottlenecks

- **Momentum loop** (Q and R functions): ~95% of runtime
  - Trapezoid quadrature with ~300 steps
  - Nested Gauss-Legendre integration (10 points)
  - Called once per output k-value (typically 120 times)

### Wiggle vs. No-Wiggle

Both versions are computed simultaneously:
- **Wiggle**: Full linear P(k) with BAO oscillations
- **No-wiggle**: Smooth P(k) obtained by removing BAO features

The no-wiggle version is used for IR-resummation in `rsd.c`.

## Troubleshooting

### Build Errors

**Error:** `hdf5.h: No such file or directory`
- **Fix:** Install HDF5 development package and add include path to `CFLAGS`

**Error:** `undefined reference to H5Fcreate`
- **Fix:** Add `-lhdf5` to `LDFLAGS` in Makefile

### Runtime Errors

**Error:** `HDF5 support not compiled`
- **Fix:** Rebuild with `-DUSE_HDF5` flag

**Error:** `Could not create HDF5 file`
- **Fix:** Check file path and write permissions

### Verification Failures

If your implementation disagrees with reference:

1. **Check input matching** - Verify linear P(k) is identical
2. **Compare sigma values** - Should match to 1e-8
3. **Check kernel constants** - Should match to 1e-10
4. **Compare P22, P13** - Check individual k-values
5. **Inspect RSD integrals** - Most sensitive to quadrature settings

## Example: Complete Workflow

```bash
# 1. Build with HDF5 support
make clean
make CFLAGS="-O3 -DUSE_HDF5" LDFLAGS="-lm -lhdf5"

# 2. Generate reference snapshot
./fkpt Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 \
       fnamePS=pkl_z05.dat dumpKfunctions=reference_snapshot.h5

# 3. Verify file structure
h5dump -n reference_snapshot.h5

# 4. Load in Python and implement your version
python my_kfunctions_implementation.py

# 5. Compare results
python verify_implementation.py reference_snapshot.h5 my_output.h5
```

## References

- **fkpt paper**: arXiv:2312.10510
- **HDF5 documentation**: https://www.hdfgroup.org/solutions/hdf5/
- **h5py (Python)**: https://docs.h5py.org/
- **HDF5.jl (Julia)**: https://github.com/JuliaIO/HDF5.jl

## Changelog

- **2025-01-12**: Initial HDF5 snapshot implementation
