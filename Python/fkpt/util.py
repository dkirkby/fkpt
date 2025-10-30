import time

import numpy as np

from scipy.special import roots_legendre

from fkpt.types import AbsCalculator, Float64NDArray, KFunctionsInitData, KFunctionsOut
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
        calculator: AbsCalculator,
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
    calculator.initialize(kfuncs_in)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"calculator.initialize in {1e3 * elapsed_time:.2f} ms")

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
    print(f"First calculator.evaluate in {1e3 * elapsed_time:.2f} ms")

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
    print(f"Average time over {nruns} runs: {1e3 * elapsed_time / nruns:.1f} ms")
