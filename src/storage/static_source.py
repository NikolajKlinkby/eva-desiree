"""A detector set that serves raw data loaded from a file (offline analysis, Qt-free).

Same interface as daq.data_loader.DetectorSet (snapshot()), so the whole pipeline runs on it unchanged:
coincidence search, selection, position correction, imaging, centring, Abel inversion. The data never grow.
"""
from daq.data_loader import Poller, SetSnapshot, Snapshot


class StaticDetectorSet(Poller):
    def __init__(self, run, detectors, source_files=()):
        """detectors: {det: (epoch, {column: 1D array})}"""
        self.run = run
        self.source_files = tuple(source_files)
        snaps = {}
        for det, (epoch, cols) in detectors.items():
            n = min((len(c) for c in cols.values()), default=0)
            snaps[int(det)] = Snapshot(int(epoch), int(n), {k: v[:n] for k, v in cols.items()})
        self._snap = SetSnapshot(run, snaps)

    def snapshot(self):
        return self._snap

    def poll(self):
        return 0                                  # nothing ever arrives
