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
        snapshot: KFunctionsSnapshot,
        rtol: float = 1e-5,
        atol: float = 1e-8
        ) -> bool:
    """Validate that the input k-grid matches the snapshot k-grid."""

    ok = True

    def allclose(a, b, name: str) -> None:
        nonlocal ok
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            diff = atol + rtol * np.abs(b)
            nfail = np.where(np.abs(a - b) > diff)[0].size
            max_abs_diff = np.max(np.abs(a - b))
            max_rel_diff = np.max(np.abs(a - b) / (np.abs(b) + atol))
            print(f"{name:<26} validation fails at {nfail:3d}/{len(b)} elements: max diffs {max_abs_diff:<.3e} (abs) {max_rel_diff:<.3e} (rel)")
            ok = False

    for i, name in enumerate(("kfuncs_wiggle", "kfuncs_nowiggle")):
        B = getattr(snapshot, name)

        # Validate P22 results
        allclose(X.P22dd[i], B.P22dd, f"{name}.P22dd")
        allclose(X.P22du[i], B.P22du, f"{name}.P22du")
        allclose(X.P22uu[i], B.P22uu, f"{name}.P22uu")

        # Validate 3-pt correlations (A-terms)
        allclose(X.I1udd1A[i], B.I1udd1A, f"{name}.I1udd1A")
        allclose(X.I2uud1A[i], B.I2uud1A, f"{name}.I2uud1A")
        allclose(X.I2uud2A[i], B.I2uud2A, f"{name}.I2uud2A")
        allclose(X.I3uuu2A[i], B.I3uuu2A, f"{name}.I3uuu2A")
        allclose(X.I3uuu3A[i], B.I3uuu3A, f"{name}.I3uuu3A")

        # Validate D-terms (4-pt correlations)
        allclose(X.I2uudd1BpC[i], B.I2uudd1BpC, f"{name}.I2uudd1BpC")
        allclose(X.I2uudd2BpC[i], B.I2uudd2BpC, f"{name}.I2uudd2BpC")
        allclose(X.I3uuud2BpC[i], B.I3uuud2BpC, f"{name}.I3uuud2BpC")
        allclose(X.I3uuud3BpC[i], B.I3uuud3BpC, f"{name}.I3uuud3BpC")
        allclose(X.I4uuuu2BpC[i], B.I4uuuu2BpC, f"{name}.I4uuuu2BpC")
        allclose(X.I4uuuu3BpC[i], B.I4uuuu3BpC, f"{name}.I4uuuu3BpC")
        allclose(X.I4uuuu4BpC[i], B.I4uuuu4BpC, f"{name}.I4uuuu4BpC")

        # Validate bias terms
        allclose(X.Pb1b2[i],     B.Pb1b2, f"{name}.Pb1b2")
        allclose(X.Pb1bs2[i],    B.Pb1bs2, f"{name}.Pb1bs2")
        allclose(X.Pb22[i],      B.Pb22, f"{name}.Pb22")
        allclose(X.Pb2s2[i],     B.Pb2s2, f"{name}.Pb2s2")
        allclose(X.Ps22[i],      B.Ps22, f"{name}.Ps22")
        allclose(X.Pb2theta[i],  B.Pb2theta, f"{name}.Pb2theta")
        allclose(X.Pbs2theta[i], B.Pbs2theta, f"{name}.Pbs2theta")

        # Validate P13 results
        allclose(X.P13dd[i], B.P13dd, f"{name}.P13dd")
        allclose(X.P13du[i], B.P13du, f"{name}.P13du")
        allclose(X.P13uu[i], B.P13uu, f"{name}.P13uu")

        # Validate additional results
        allclose(X.sigma32PSL[i], B.sigma32PSL, f"{name}.sigma32PSL")
        allclose(X.pkl[i], B.pkl, f"{name}.pkl")

    return ok

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

    start_time = time.time()
    kfuncs_in = init_kfunctions(
        k_in, Pk_in, Pk_nw_in, fk_in,
        f0,
        kmin, kmax, Nk,
        nquadSteps, NQ, NR
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Initialized k-functions input in {1e3 * elapsed_time:.1f} ms")

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
