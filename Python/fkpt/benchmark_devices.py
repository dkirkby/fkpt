"""Benchmark JAX implementation on different devices (CPU vs GPU)."""

import sys
import time

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    print(f"JAX version: {jax.__version__}")
    print(f"Available JAX devices: {jax.devices()}")
except ImportError:
    print("ERROR: JAX is not installed. Install with: pip install jax jaxlib")
    sys.exit(1)

import numpy as np

from fkpt.snapshot import load_snapshot
from fkpt.util import init_kfunctions, validate_kfunctions
from fkpt.calculate_numpy import calculate as calculate_numpy
from fkpt.calculate_jax import calculate as calculate_jax, _calculate_jax_core


def benchmark_on_device(device, kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v, nruns=100, include_wrapper=False):
    """Benchmark JAX calculation on a specific device.

    Args:
        device: JAX device to run on
        kfuncs_in: Input data
        A, ApOverf0, CFD3, CFD3p, sigma2v: Parameters
        nruns: Number of benchmark iterations
        include_wrapper: If True, benchmark the full calculate() wrapper including conversions.
                        If False, benchmark only the core JIT-compiled function.

    Returns:
        List of execution times in milliseconds
    """

    if include_wrapper:
        # Benchmark the full wrapper function (like test_jax.py does)
        # Clear cache to ensure fresh conversion for this device
        from fkpt.calculate_jax import _jax_cache
        _jax_cache.clear()

        with jax.default_device(device):
            # Warm-up run
            print(f"  Warming up (JIT compilation + wrapper)...")
            _ = calculate_jax(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)

            # Benchmark runs
            print(f"  Running {nruns} benchmark iterations...")
            times = []
            for _ in range(nruns):
                t0 = time.perf_counter()
                results = calculate_jax(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # Convert to ms
    else:
        # Benchmark only the core function (pure computation)
        # Convert numpy arrays to JAX arrays with explicit float64 dtype
        k_in_jax = jnp.asarray(kfuncs_in.k_in, dtype=jnp.float64)
        logk_grid_jax = jnp.asarray(kfuncs_in.logk_grid, dtype=jnp.float64)
        kk_grid_jax = jnp.asarray(kfuncs_in.kk_grid, dtype=jnp.float64)
        Y_jax = jnp.asarray(kfuncs_in.Y, dtype=jnp.float64)
        Y2_jax = jnp.asarray(kfuncs_in.Y2, dtype=jnp.float64)  # Use pre-computed NumPy Y2
        xxQ_jax = jnp.asarray(kfuncs_in.xxQ, dtype=jnp.float64)
        wwQ_jax = jnp.asarray(kfuncs_in.wwQ, dtype=jnp.float64)
        xxR_jax = jnp.asarray(kfuncs_in.xxR, dtype=jnp.float64)
        wwR_jax = jnp.asarray(kfuncs_in.wwR, dtype=jnp.float64)

        # Move arrays to the specified device
        with jax.default_device(device):
            k_in_jax = jax.device_put(k_in_jax, device)
            logk_grid_jax = jax.device_put(logk_grid_jax, device)
            kk_grid_jax = jax.device_put(kk_grid_jax, device)
            Y_jax = jax.device_put(Y_jax, device)
            Y2_jax = jax.device_put(Y2_jax, device)
            xxQ_jax = jax.device_put(xxQ_jax, device)
            wwQ_jax = jax.device_put(wwQ_jax, device)
            xxR_jax = jax.device_put(xxR_jax, device)
            wwR_jax = jax.device_put(wwR_jax, device)

            # Warm-up run to trigger JIT compilation
            print(f"  Warming up (JIT compilation - core only)...")
            _ = _calculate_jax_core(
                k_in_jax, logk_grid_jax, kk_grid_jax, Y_jax, Y2_jax,
                xxQ_jax, wwQ_jax, xxR_jax, wwR_jax,
                A, ApOverf0, CFD3, CFD3p, sigma2v
            )
            jax.block_until_ready(_)

            # Benchmark runs
            print(f"  Running {nruns} benchmark iterations...")
            times = []
            for _ in range(nruns):
                t0 = time.perf_counter()
                results = _calculate_jax_core(
                    k_in_jax, logk_grid_jax, kk_grid_jax, Y_jax, Y2_jax,
                    xxQ_jax, wwQ_jax, xxR_jax, wwR_jax,
                    A, ApOverf0, CFD3, CFD3p, sigma2v
                )
                # Block until computation is done (important for accurate timing)
                jax.block_until_ready(results)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # Convert to ms

    return times


def main():
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
    print("DEVICE COMPARISON BENCHMARK")
    print("="*80)

    # Get available devices
    devices = jax.devices()

    # Separate CPUs and GPUs
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    gpu_devices = [d for d in devices if d.platform == 'gpu' or d.platform == 'cuda']

    # If no CPU devices available, try to get one explicitly
    if not cpu_devices:
        try:
            cpu_device = jax.devices('cpu')[0]
            cpu_devices = [cpu_device]
        except:
            pass

    print(f"\nFound {len(cpu_devices)} CPU device(s) and {len(gpu_devices)} GPU device(s)")

    results = {}

    # Benchmark NumPy (CPU baseline)
    print("\n" + "-"*80)
    print("NumPy (CPU baseline)")
    print("-"*80)
    print("  Warming up...")
    _ = calculate_numpy(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)

    print("  Running 100 benchmark iterations...")
    numpy_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = calculate_numpy(kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v)
        t1 = time.perf_counter()
        numpy_times.append((t1 - t0) * 1000)

    results['NumPy (CPU)'] = numpy_times

    # Benchmark on CPU
    if cpu_devices:
        for i, device in enumerate(cpu_devices[:1]):  # Just use first CPU
            print("\n" + "-"*80)
            print(f"JAX on CPU: {device} (with wrapper, like test_jax.py)")
            print("-"*80)
            times = benchmark_on_device(device, kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v, nruns=100, include_wrapper=True)
            results[f'JAX (CPU-{i}) wrapper'] = times

            print("\n" + "-"*80)
            print(f"JAX on CPU: {device} (core only)")
            print("-"*80)
            times = benchmark_on_device(device, kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v, nruns=100, include_wrapper=False)
            results[f'JAX (CPU-{i}) core'] = times

    # Benchmark on GPU
    if gpu_devices:
        for i, device in enumerate(gpu_devices[:1]):  # Just benchmark first GPU
            print("\n" + "-"*80)
            print(f"JAX on GPU {i}: {device} (with wrapper, like test_jax.py)")
            print("-"*80)
            times = benchmark_on_device(device, kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v, nruns=100, include_wrapper=True)
            results[f'JAX (GPU-{i}) wrapper'] = times

            print("\n" + "-"*80)
            print(f"JAX on GPU {i}: {device} (core only)")
            print("-"*80)
            times = benchmark_on_device(device, kfuncs_in, A, ApOverf0, CFD3, CFD3p, sigma2v, nruns=100, include_wrapper=False)
            results[f'JAX (GPU-{i}) core'] = times
    else:
        print("\n⚠ No GPU devices found. Skipping GPU benchmarks.")

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Device':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Speedup'}")
    print("-"*80)

    numpy_mean = np.mean(results['NumPy (CPU)'])

    for name, times in results.items():
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        speedup = numpy_mean / mean_time

        print(f"{name:<20} {mean_time:>10.2f}ms {std_time:>10.2f}ms {min_time:>10.2f}ms {max_time:>10.2f}ms {speedup:>7.1f}x")

    # If we have both CPU and GPU, show the comparison
    if cpu_devices and gpu_devices:
        print("\n" + "-"*80)
        print("KEY COMPARISONS:")
        print("-"*80)

        cpu_wrapper_mean = np.mean(results['JAX (CPU-0) wrapper'])
        cpu_core_mean = np.mean(results['JAX (CPU-0) core'])
        gpu_wrapper_mean = np.mean(results['JAX (GPU-0) wrapper'])
        gpu_core_mean = np.mean(results['JAX (GPU-0) core'])

        print(f"JAX GPU (wrapper) vs JAX CPU (wrapper): {cpu_wrapper_mean / gpu_wrapper_mean:.1f}x")
        print(f"JAX GPU (core) vs JAX CPU (core): {cpu_core_mean / gpu_core_mean:.1f}x")
        print(f"JAX GPU (wrapper) vs NumPy: {numpy_mean / gpu_wrapper_mean:.1f}x")
        print(f"JAX GPU (core) vs NumPy: {numpy_mean / gpu_core_mean:.1f}x")
        print(f"\nWrapper overhead on GPU: {gpu_wrapper_mean - gpu_core_mean:.2f} ms ({((gpu_wrapper_mean / gpu_core_mean) - 1) * 100:.1f}%)")
        print(f"Wrapper overhead on CPU: {cpu_wrapper_mean - cpu_core_mean:.2f} ms ({((cpu_wrapper_mean / cpu_core_mean) - 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
