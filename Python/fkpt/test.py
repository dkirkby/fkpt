from fkpt.snapshot import load_snapshot
from fkpt.util import measure_kfunctions
from fkpt.calculate_numpy import calculate as calculate_numpy


def main():

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'
    print(f"Loading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)

    # Measure k-functions using calculate_numpy
    measure_kfunctions(calculate_numpy, snapshot, nruns=100)


if __name__ == "__main__":
    main()
