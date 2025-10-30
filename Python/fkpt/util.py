import time

import numpy as np

from scipy.special import roots_legendre

from fkpt.types import Float64NDArray, KFunctionsInitData, KFunctionsOut, KFunctionsCalculator
from fkpt.snapshot import KFunctionsSnapshot


def setup_kfunctions(
        k_in: Float64NDArray,
        kmin: int, kmax: int, Nk: int,
        nquadSteps: int, NQ: int=10, NR: int=10
        ) -> KFunctionsInitData:

    # Initialize logarithmic output k grid
    logk_grid = np.geomspace(kmin, kmax, Nk)

    # Set up quadrature k grid
    pmin = max(k_in[0], 0.01 * kmin)
    pmax = min(k_in[-1], 16.0 * kmax)
    kk_grid = np.geomspace(pmin, pmax, nquadSteps)

    # Initialize Gauss-Legendre nodes and weights on [-1,1]
    xxQ, wwQ = roots_legendre(NQ)
    xxR, wwR = roots_legendre(NR)

    return KFunctionsInitData(
        k_in, logk_grid, kk_grid,
        xxQ, wwQ, xxR, wwR,
    )

def validate_kfunctions(
        X: KFunctionsOut,
        snapshot: KFunctionsSnapshot,
        rtol: float = 1e-5,
        atol: float = 1e-8
        ) -> bool:
    """Validate that the input k-grid matches the snapshot k-grid."""

    ok = True
    for i, name in enumerate(("kfuncs_wiggle", "kfuncs_nowiggle")):
        B = getattr(snapshot, name)
        for field in KFunctionsOut._fields:
            a = getattr(X, field)[i]
            b = getattr(B, field)
            if not np.allclose(a, b, rtol=rtol, atol=atol):
                diff = atol + rtol * np.abs(b)
                nfail = np.where(np.abs(a - b) > diff)[0].size
                max_abs_diff = np.max(np.abs(a - b))
                max_rel_diff = np.max(np.abs(a - b) / (np.abs(b) + atol))
                print(f"{name:<15s}.{field:<10s} validation fails at {nfail:3d}/{len(b)} elements: max diffs {max_abs_diff:<.3e} (abs) {max_rel_diff:<.3e} (rel)")
                ok = False

    return ok

def measure_kfunctions(
        calculator_cls: KFunctionsCalculator,
        snapshot: KFunctionsSnapshot,
        nruns: int = 10
        ) -> None:
    """Measure k-functions using the provided calculator and snapshot data."""

    # Prepare k-functions input
    k_in = snapshot.ps_wiggle.k
    kmin = snapshot.k_grid.kmin
    kmax = snapshot.k_grid.kmax
    Nk = snapshot.k_grid.Nk
    nquadSteps = snapshot.numerical.nquadSteps
    NQ = 10
    NR = 10

    start_time = time.time()
    kfuncs_in = setup_kfunctions(
        k_in, kmin, kmax, Nk,
        nquadSteps, NQ, NR
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"setup_kfunctions in {1e3 * elapsed_time:.2f} ms")

    start_time = time.time()
    calculator = calculator_cls()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{calculator_cls.__name__}.ctor in {1e3 * elapsed_time:.2f} ms")

    start_time = time.time()
    calculator.initialize(kfuncs_in)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{calculator_cls.__name__}.initialize in {1e3 * elapsed_time:.2f} ms")

    # kernel constants
    if False: # _KERNELS_LCDMfk_ on line 287
        A = snapshot.kernels.KA_LCDM
        ApOverf0 = snapshot.kernels.KAp_LCDM / snapshot.cosmology.f0
        CFD3 = snapshot.kernels.KR1_LCDM
        CFD3p = snapshot.kernels.KR1p_LCDM
    else:
        A = 1
        ApOverf0 = 0
        CFD3 = 1
        CFD3p = 1

    # Calculate k-functions first time to validate results and do any JIT initialization
    Pk_in = snapshot.ps_wiggle.P
    Pk_nw_in = snapshot.ps_nowiggle.P
    fk_in = snapshot.ps_wiggle.f
    sigma2v = snapshot.sigma_values.sigma2v
    f0 = snapshot.cosmology.f0
    start_time = time.time()
    kfuncs_out = calculator.evaluate(Pk_in, Pk_nw_in, fk_in, A, ApOverf0, CFD3, CFD3p, sigma2v, f0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"First {calculator_cls.__name__}.evaluate in {1e3 * elapsed_time:.2f} ms")

    # Validate results
    if validate_kfunctions(kfuncs_out, snapshot):
        print("K-functions validated successfully against the snapshot!")
    else:
        print("K-functions validation failed!")

    # Measure time for multiple evaluations
    start_time = time.time()
    for _ in range(nruns):
        kfuncs_out = calculator.evaluate(Pk_in, Pk_nw_in, fk_in, A, ApOverf0, CFD3, CFD3p, sigma2v, f0)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Average {calculator_cls.__name__}.evaluate over {nruns} runs: {1e3 * elapsed_time / nruns:.1f} ms")
