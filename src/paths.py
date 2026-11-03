"""Project locations and small file-system helpers (no Qt)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # <project>/src/paths.py
EXAMPLE_DATA = ROOT / "examples" / "test_data"
FAKE_DAQ = ROOT / "tools" / "fake_daq.py"

_RUN_FILE = re.compile(r"^RA(\d+)_(\d{8}[._]\d{6})\.csv$")


def scan_detectors(directory):
    """Sorted detector numbers that have RA<det>_<date>.<time>.csv files in directory."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted({int(m[1]) for f in d.iterdir() if (m := _RUN_FILE.match(f.name))})


def parse_run_file(path):
    """(directory, run stamp, detector number) of a DAQ file RA<det>_<date>.<time>.csv, else None."""
    p = Path(path)
    m = _RUN_FILE.match(p.name)
    return (str(p.parent), m[2], int(m[1])) if m else None


def scan_runs(directory):
    """{run stamp: sorted detector numbers} of all DAQ runs in a directory (empty if none)."""
    d = Path(directory)
    runs = {}
    if d.is_dir():
        for f in d.iterdir():
            if m := _RUN_FILE.match(f.name):
                runs.setdefault(m[2], set()).add(int(m[1]))
    return {r: sorted(v) for r, v in sorted(runs.items())}
