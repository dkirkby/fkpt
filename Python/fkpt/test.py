from fkpt.snapshot import load_snapshot
from fkpt.util import measure_kfunctions
from fkpt.calculate_numpy import NumpyCalculator
from fkpt.calculate_jax import JaxCalculator
import jax

def check_gpu_available():
    """Check if GPU is available for JAX."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        return len(gpu_devices) > 0, devices
    except Exception:
        return False, []

def main():

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'
    print(f"Loading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)

    # Measure k-functions using NumPy calculator
    print(f"\nMeasuring k-functions using NumPy calculator:")
    measure_kfunctions(NumpyCalculator, snapshot, nruns=100)

    # Check GPU availability
    has_gpu, devices = check_gpu_available()
    print(f"\nGPU availability: {has_gpu}")
    print(f"JAX devices: {devices}")

    # Measure k-functions using JAX calculator on CPU
    print(f"\nMeasuring k-functions using JAX calculator (CPU):")
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    measure_kfunctions(JaxCalculator, snapshot, nruns=100)

    # Measure k-functions using JAX calculator on GPU (if available)
    if has_gpu:
        print(f"\nMeasuring k-functions using JAX calculator (GPU):")
        jax.config.update('jax_default_device', jax.devices('gpu')[0])
        measure_kfunctions(JaxCalculator, snapshot, nruns=100)
    else:
        print("\nGPU not available, skipping JAX GPU benchmark.")


if __name__ == "__main__":
    main()
