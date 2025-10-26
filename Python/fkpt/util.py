import numpy as np

from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre

from fkpt.types import Float64NDArray, KFunctionsIn, KFunctionsOut, KFunctionsCalculator
from fkpt.snapshot import KFunctionsSnapshot, load_snapshot
from fkpt.calculate4 import calculator4


def init_kfunctions(
        k_in: Float64NDArray, Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray, fk_in: Float64NDArray,
        f0: float, sigma2v: float,
        kmin: int, kmax: int, Nk: int,
        nquadSteps: int=10, NQ: int=10, NR: int=10
        ) -> KFunctionsIn:

    # Initialize cubic spline interpolator
    interpolator = CubicSpline(k_in, np.vstack([Pk_in, Pk_nw_in, fk_in]).T, axis=0)

    # Initialize logarithmic output k grid
    logk_grid = np.geomspace(kmin, kmax, Nk)

    # Precompute f(k) on output k grid
    Pout, _, fout = interpolator(logk_grid).T
    fout /= f0  # Normalize f(k) by f0

    # Set up quadrature k grid
    pmin = max(k_in[0], 0.01 * kmin)
    pmax = min(k_in[-1], 16.0 * kmax)
    kk_grid = np.geomspace(pmin, pmax, nquadSteps)

    # Precompute P(k) and f(k) on quadrature grid
    Pkk, Pkk_nw, fkk = interpolator(kk_grid).T
    fkk /= f0  # Normalize f(k) by f0

    # Initialize Gauss-Legendre nodes and weights on [-1,1]
    xxQ, wwQ = roots_legendre(NQ)
    xxR, wwR = roots_legendre(NR)

    return KFunctionsIn(
        k_in[0], k_in[-1],
        logk_grid, Pout, fout,
        f0, sigma2v,
        kk_grid, Pkk, Pkk_nw, fkk,
        xxQ, wwQ, xxR, wwR,
        interpolator
    )

def validate_kfunctions(
        X: KFunctionsOut,
        snapshot: KFunctionsSnapshot
        ) -> bool:
    """Validate that the input k-grid matches the snapshot k-grid."""

    try:
        # Validate P22 results
        assert np.allclose(X.P22dd, snapshot.kfuncs_wiggle.P22dd), "P22dd does not match!"
        assert np.allclose(X.P22du, snapshot.kfuncs_wiggle.P22du), "P22du does not match!"
        assert np.allclose(X.P22uu, snapshot.kfuncs_wiggle.P22uu), "P22uu does not match!"

        # Validate 3-pt correlations (A-terms)
        assert np.allclose(X.I1udd1A, snapshot.kfuncs_wiggle.I1udd1A), "I1udd1A does not match!"
        assert np.allclose(X.I2uud1A, snapshot.kfuncs_wiggle.I2uud1A), "I2uud1A does not match!"
        assert np.allclose(X.I2uud2A, snapshot.kfuncs_wiggle.I2uud2A), "I2uud2A does not match!"
        assert np.allclose(X.I3uuu2A, snapshot.kfuncs_wiggle.I3uuu2A), "I3uuu2A does not match!"
        assert np.allclose(X.I3uuu3A, snapshot.kfuncs_wiggle.I3uuu3A), "I3uuu3A does not match!"

        # Validate D-terms (4-pt correlations)
        assert np.allclose(X.I2uudd1BpC, snapshot.kfuncs_wiggle.I2uudd1BpC), "I2uudd1BpC does not match!"
        assert np.allclose(X.I2uudd2BpC, snapshot.kfuncs_wiggle.I2uudd2BpC), "I2uudd2BpC does not match!"
        assert np.allclose(X.I3uuud2BpC, snapshot.kfuncs_wiggle.I3uuud2BpC), "I3uuud2BpC does not match!"
        assert np.allclose(X.I3uuud3BpC, snapshot.kfuncs_wiggle.I3uuud3BpC), "I3uuud3BpC does not match!"
        assert np.allclose(X.I4uuuu2BpC, snapshot.kfuncs_wiggle.I4uuuu2BpC), "I4uuuu2BpC does not match!"
        assert np.allclose(X.I4uuuu3BpC, snapshot.kfuncs_wiggle.I4uuuu3BpC), "I4uuuu3BpC does not match!"
        assert np.allclose(X.I4uuuu4BpC, snapshot.kfuncs_wiggle.I4uuuu4BpC), "I4uuuu4BpC does not match!"

        # Validate bias terms
        assert np.allclose(X.Pb1b2, snapshot.kfuncs_wiggle.Pb1b2), "Pb1b2 does not match!"
        assert np.allclose(X.Pb1bs2, snapshot.kfuncs_wiggle.Pb1bs2), "Pb1bs2 does not match!"
        assert np.allclose(X.Pb22, snapshot.kfuncs_wiggle.Pb22), "Pb22 does not match!"
        assert np.allclose(X.Pb2s2, snapshot.kfuncs_wiggle.Pb2s2), "Pb2s2 does not match!"
        assert np.allclose(X.Ps22, snapshot.kfuncs_wiggle.Ps22), "Ps22 does not match!"
        assert np.allclose(X.Pb2theta, snapshot.kfuncs_wiggle.Pb2theta), "Pb2theta does not match!"
        assert np.allclose(X.Pbs2theta, snapshot.kfuncs_wiggle.Pbs2theta), "Pbs2theta does not match!"

        # Validate P13 results
        assert np.allclose(X.P13dd, snapshot.kfuncs_wiggle.P13dd), "P13dd does not match!"
        assert np.allclose(X.P13du, snapshot.kfuncs_wiggle.P13du), "P13du does not match!"
        assert np.allclose(X.P13uu, snapshot.kfuncs_wiggle.P13uu), "P13uu does not match!"

        # Validate additional results
        assert np.allclose(X.sigma32PSL, snapshot.kfuncs_wiggle.sigma32PSL), "sigma32PSL does not match!"

        return True

    except AssertionError as e:
        print(f"Validation error: {e}")
        return False

def measure_kfunctions(
        calculator: KFunctionsCalculator,
        snapshot: KFunctionsSnapshot
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
        f0, sigma2v,
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

    # Calculate k-functions
    kfuncs_out = calculator(kfuncs_in, A, ApOverf0, CFD3, CFD3p)

    # Validate results
    if validate_kfunctions(kfuncs_out, snapshot):
        print("K-functions validated successfully against the snapshot!")
    else:
        warnings.warn("K-functions validation failed!")


if __name__ == "__main__":
    import warnings

    # Example usage
    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'

    print(f"Loading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)
    print("Snapshot loaded successfully!")

    # Measure k-functions using calculator4
    measure_kfunctions(calculator4, snapshot)
