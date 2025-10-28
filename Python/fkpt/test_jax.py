"""Test and benchmark JAX implementation against NumPy version."""

import sys

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    print("ERROR: JAX is not installed. Install with: pip install jax jaxlib")
    sys.exit(1)

import numpy as np

from fkpt.snapshot import load_snapshot
from fkpt.util import measure_kfunctions, init_kfunctions, validate_kfunctions
from fkpt.calculate_numpy import calculate as calculate_numpy
from fkpt.calculate_jax import calculate as calculate_jax


def test_jax_vs_numpy():
    """Test that JAX and NumPy versions produce identical results."""

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'
    print(f"\nLoading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)

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

    # Kernel constants
    A = 1
    ApOverf0 = 0
    CFD3 = 1
    CFD3p = 1

    print("\n" + "="*80)
    print("CORRECTNESS TEST: JAX vs NumPy")
    print("="*80)

    # Calculate with NumPy
    print("\nRunning NumPy calculation...")
    result_numpy = calculate_numpy(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)

    # Calculate with JAX
    print("Running JAX calculation...")
    result_jax = calculate_jax(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)

    # Compare results
    print("\nComparing results...")
    print("-"*80)

    all_close = True
    tolerance = 1e-5  # Tolerance for floating point comparison (JAX may have slight differences)

    field_names = result_numpy._fields
    for i, field in enumerate(field_names):
        np_val = result_numpy[i]
        jax_val = result_jax[i]

        # Check if arrays are close (use relative tolerance primarily)
        is_close = np.allclose(np_val, jax_val, rtol=tolerance, atol=1e-3)

        if not is_close:
            max_diff = np.max(np.abs(np_val - jax_val))
            rel_diff = max_diff / (np.max(np.abs(np_val)) + 1e-10)
            print(f"✗ {field:20s}: MAX DIFF = {max_diff:.2e}, REL DIFF = {rel_diff:.2e}")
            all_close = False
        else:
            print(f"✓ {field:20s}: PASSED")

    print("-"*80)
    if all_close:
        print("✓ ALL TESTS PASSED - JAX and NumPy produce identical results!")
    else:
        print("✗ SOME TESTS FAILED - Results differ between JAX and NumPy")

    # Validate against snapshot
    print("\n" + "="*80)
    print("VALIDATION AGAINST C SNAPSHOT")
    print("="*80)

    if validate_kfunctions(result_jax, snapshot):
        print("✓ JAX results validated successfully against C snapshot!")
    else:
        print("✗ JAX results validation failed!")

    return all_close


def benchmark_jax_vs_numpy():
    """Benchmark JAX vs NumPy performance."""

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'
    print(f"\nLoading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)

    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)

    # Benchmark NumPy
    print("\nNumPy implementation:")
    print("-"*80)
    measure_kfunctions(calculate_numpy, snapshot, nruns=100)

    # Benchmark JAX
    print("\nJAX implementation:")
    print("-"*80)
    measure_kfunctions(calculate_jax, snapshot, nruns=100)


def main():
    """Main test and benchmark routine."""

    print("="*80)
    print("JAX IMPLEMENTATION TEST SUITE")
    print("="*80)

    # Test correctness
    passed = test_jax_vs_numpy()

    #if not passed:
    #    print("\n⚠ WARNING: Correctness tests failed. Skipping benchmarks.")
    #    sys.exit(1)

    # Benchmark performance
    benchmark_jax_vs_numpy()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
