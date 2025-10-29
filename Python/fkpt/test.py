from fkpt.snapshot import load_snapshot
from fkpt.util import measure_kfunctions
from fkpt.calculate_numpy import calculate as calculate_numpy
from fkpt.calculate_jax import calculate as calculate_jax

def main():

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'
    print(f"Loading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)

    # Measure k-functions using available calculators
    for calc_name, calculator in [('NumPy', calculate_numpy), ('JAX', calculate_jax)]:
        print(f"\nMeasuring k-functions using {calc_name} calculator:")
        measure_kfunctions(calculator, snapshot, nruns=100)


if __name__ == "__main__":
    main()
