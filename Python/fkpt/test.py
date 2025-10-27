from fkpt.snapshot import load_snapshot
from fkpt.util import measure_kfunctions
from fkpt.calculate4 import calculator4


def main():

    SNAPSHOT_FILE = '../kfunctions_snapshot_new.h5'

    print(f"Loading snapshot from: {SNAPSHOT_FILE}")
    snapshot = load_snapshot(SNAPSHOT_FILE)
    print("Snapshot loaded successfully!")

    # Measure k-functions using calculator4
    measure_kfunctions(calculator4, snapshot)


if __name__ == "__main__":
    main()
