import time

import numpy as np

from scipy.special import roots_legendre

from fkpt.types import Float64NDArray, KFunctionsIn, KFunctionsOut, KFunctionsCalculator
from fkpt.snapshot import KFunctionsSnapshot


def init_cubic_spline(x: Float64NDArray, y: Float64NDArray) -> Float64NDArray:
    """Initialize a cubic spline interpolator by precomputing 2nd derivatives."""
    n = len(x)
    y = np.moveaxis(y, -1, 0)
    y2 = np.zeros_like(y)
    u = np.zeros_like(y)
    # Forward sweep
    for i in range(1, n-1):
        sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        p = sig * y2[i-1] + 2.0
        y2[i] = (sig - 1.0) / p
        udiff = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1])
        u[i] = (6.0 * udiff / (x[i+1] - x[i-1]) - sig * u[i-1]) / p
    # Back substitution
    for k in range(n-2, -1, -1):
        y2[k] = y2[k] * y2[k+1] + u[k if k < n-1 else n-2]
    return np.moveaxis(y2, 0, -1)

def eval_cubic_spline(xa: Float64NDArray, ya: Float64NDArray, y2a: Float64NDArray, x: Float64NDArray) -> Float64NDArray:
    """Evaluate the cubic spline interpolator at given x values assuming xa increasing."""
    idx_hi = np.searchsorted(xa, x, side='right')
    idx_hi = np.clip(idx_hi, 1, xa.size - 1)
    idx_lo = idx_hi - 1
    h = xa[idx_hi] - xa[idx_lo]
    a = (xa[idx_hi] - x) / h
    b = (x - xa[idx_lo]) / h
    return np.moveaxis(
        (a * ya[...,idx_lo] + b * ya[...,idx_hi] +
        ((a**3 - a) * y2a[...,idx_lo] + (b**3 - b) * y2a[...,idx_hi]) * (h**2) / 6.0),
        0, -1)


def init_kfunctions(
        k_in: Float64NDArray,
        Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray,
        fk_in: Float64NDArray, f0: float,
        kmin: int, kmax: int, Nk: int,
        nquadSteps: int, NQ: int=10, NR: int=10
        ) -> KFunctionsIn:

    # Initialize simultaneous cubic spline interpolation for Y(k) = [P(k), P_nw(k), f(k)]
    # by precomputing 2nd derivatives Y2(k)
    X = k_in
    Y = np.vstack([Pk_in, Pk_nw_in, fk_in / f0])
    Y2 = init_cubic_spline(X, Y)

    # Initialize logarithmic output k grid
    logk_grid = np.geomspace(kmin, kmax, Nk)

    # Set up quadrature k grid
    pmin = max(k_in[0], 0.01 * kmin)
    pmax = min(k_in[-1], 16.0 * kmax)
    kk_grid = np.geomspace(pmin, pmax, nquadSteps)

    # Initialize Gauss-Legendre nodes and weights on [-1,1]
    xxQ, wwQ = roots_legendre(NQ)
    xxR, wwR = roots_legendre(NR)

    return KFunctionsIn(
        k_in, logk_grid, kk_grid, Y, Y2,
        xxQ, wwQ, xxR, wwR,
    )

def validate_kfunctions(
        X: KFunctionsOut,
        snapshot: KFunctionsSnapshot
        ) -> bool:
    """Validate that the input k-grid matches the snapshot k-grid."""

    try:
        for i, name in enumerate(("kfuncs_wiggle",)):
            block = getattr(snapshot, name)
            '''
            # Validate P22 results
            assert np.allclose(X.P22dd, block.P22dd), f"{name}.P22dd does not match!"
            assert np.allclose(X.P22du, block.P22du), f"{name}.P22du does not match!"
            assert np.allclose(X.P22uu, block.P22uu), f"{name}.P22uu does not match!"

            # Validate 3-pt correlations (A-terms)
            assert np.allclose(X.I1udd1A, block.I1udd1A), f"{name}.I1udd1A does not match!"
            assert np.allclose(X.I2uud1A, block.I2uud1A), f"{name}.I2uud1A does not match!"
            assert np.allclose(X.I2uud2A, block.I2uud2A), f"{name}.I2uud2A does not match!"
            assert np.allclose(X.I3uuu2A, block.I3uuu2A), f"{name}.I3uuu2A does not match!"
            assert np.allclose(X.I3uuu3A, block.I3uuu3A), f"{name}.I3uuu3A does not match!"

            # Validate D-terms (4-pt correlations)
            assert np.allclose(X.I2uudd1BpC, block.I2uudd1BpC), f"{name}.I2uudd1BpC does not match!"
            assert np.allclose(X.I2uudd2BpC, block.I2uudd2BpC), f"{name}.I2uudd2BpC does not match!"
            assert np.allclose(X.I3uuud2BpC, block.I3uuud2BpC), f"{name}.I3uuud2BpC does not match!"
            assert np.allclose(X.I3uuud3BpC, block.I3uuud3BpC), f"{name}.I3uuud3BpC does not match!"
            assert np.allclose(X.I4uuuu2BpC, block.I4uuuu2BpC), f"{name}.I4uuuu2BpC does not match!"
            assert np.allclose(X.I4uuuu3BpC, block.I4uuuu3BpC), f"{name}.I4uuuu3BpC does not match!"
            assert np.allclose(X.I4uuuu4BpC, block.I4uuuu4BpC), f"{name}.I4uuuu4BpC does not match!"

            # Validate bias terms
            assert np.allclose(X.Pb1b2, block.Pb1b2), f"{name}.Pb1b2 does not match!"
            assert np.allclose(X.Pb1bs2, block.Pb1bs2), f"{name}.Pb1bs2 does not match!"
            assert np.allclose(X.Pb22, block.Pb22), f"{name}.Pb22 does not match!"
            assert np.allclose(X.Pb2s2, block.Pb2s2), f"{name}.Pb2s2 does not match!"
            assert np.allclose(X.Ps22, block.Ps22), f"{name}.Ps22 does not match!"
            assert np.allclose(X.Pb2theta, block.Pb2theta), f"{name}.Pb2theta does not match!"
            assert np.allclose(X.Pbs2theta, block.Pbs2theta), f"{name}.Pbs2theta does not match!"
            '''
            # Validate P13 results
            assert np.allclose(X.P13dd[i], block.P13dd), f"{name}.P13dd does not match!"
            assert np.allclose(X.P13du[i], block.P13du), f"{name}.P13du does not match!"
            assert np.allclose(X.P13uu[i], block.P13uu), f"{name}.P13uu does not match!"

            # Validate additional results
            #assert np.allclose(X.sigma32PSL, block.sigma32PSL), f"{name}.sigma32PSL does not match!"

        return True

    except AssertionError as e:
        print(f"Validation error: {e}")
        return False

def measure_kfunctions(
        calculator: KFunctionsCalculator,
        snapshot: KFunctionsSnapshot,
        nruns: int = 10
        ) -> None:
    """Measure k-functions using the provided calculator and snapshot data."""

    # Initialize k-functions input
    k_in = snapshot.ps_wiggle.k
    Pk_in = snapshot.ps_wiggle.P
    Pk_nw_in = snapshot.ps_nowiggle.P
    fk_in = snapshot.ps_wiggle.f
    f0 = snapshot.cosmology.f0
    sigma2v = snapshot.sigma_values.sigma2v
    kmin = snapshot.k_grid.kmin
    kmax = snapshot.k_grid.kmax
    Nk = snapshot.k_grid.Nk
    nquadSteps = snapshot.numerical.nquadSteps
    NQ = 10
    NR = 10
    kfuncs_in = init_kfunctions(
        k_in, Pk_in, Pk_nw_in, fk_in,
        f0,
        kmin, kmax, Nk,
        nquadSteps, NQ, NR
    )

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
    kfuncs_out = calculator(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)

    # Validate results
    if validate_kfunctions(kfuncs_out, snapshot):
        print("K-functions validated successfully against the snapshot!")
    else:
        print("K-functions validation failed!")

    # Measure time for multiple evaluations
    start_time = time.time()
    for _ in range(nruns):
        kfuncs_out = calculator(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Average time over {nruns} runs: {1e3 * elapsed_time / nruns:.1f} ms")
