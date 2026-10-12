"""The live pipeline as one object (no Qt).

    DetectorSet -> CoincidenceFilter -> SelectionFilter -> Imager
                                                  +-> PolarAnalysis (centre, circularise, polar images) -> AbelAnalysis

Each stage runs in its own worker thread and publishes immutable results; a UI
(or a script) only ever reads .coincidence() / .selection() / .imaging() and
calls the set_* methods of the stage objects (flt, sel, img), all thread-safe.

demo_source: instead of reading a live DAQ directory, replay that folder of
recorded runs with tools/fake_daq.py into a temporary directory.

offline_file: instead of reading DAQ files, take the raw data of a saved analysis file (storage/h5file.py) and
re-run the whole pipeline on them; `preset` (the settings of that file) is applied before the stages start.
"""
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from analysis.coincidence import CoincidenceFilter
from analysis.filters import SelectionFilter
from analysis.imaging import Imager, CartesianBinning
from analysis.abel_analysis import AbelAnalysis
from analysis.polar import PolarAnalysis
from analysis.poscorr_stage import PositionCorrector
from daq.data_loader import DetectorSet
from paths import FAKE_DAQ


@dataclass(frozen=True)
class SessionConfig:
    directory: str = "."
    det_a: int = 3                  # detector whose events are filtered/imaged first (e.g. VMI)
    det_b: int = 1                  # partner detector (e.g. RAEA)
    run: str = "latest"
    search_window_us: float = 2500.0
    shot_period_us: float = 1000.0
    bins: int = 256
    poll_interval: float = 0.2
    demo_source: str = None         # replay this folder instead of reading `directory`
    demo_speed: float = 20.0
    offline_file: str = None        # an analysis file (.h5) to continue working on (no live data)


class Session:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.ds = self.flt = self.sel = self.pos = self.img = self.polar = self.abel = None
        self.directory = None       # the directory actually read (temp dir in demo mode)
        self._tmp = self._proc = None
        self.preset = None          # saved settings (storage.h5file.read_meta()["settings"]) to apply on start

    @property
    def running(self):
        return self.ds is not None

    def start(self):
        if self.running:
            return
        c = self.config
        self.directory = c.directory
        if c.offline_file:
            from storage import h5file
            from storage.static_source import StaticDetectorSet
            run, dets, files = h5file.read_raw(c.offline_file)
            self.directory = c.offline_file
            self.ds = StaticDetectorSet(run, dets, files)
        elif c.demo_source:
            self._tmp = tempfile.mkdtemp(prefix="vmi_demo_")
            self._proc = subprocess.Popen(
                [sys.executable, str(FAKE_DAQ), "--src", str(c.demo_source), "--out", self._tmp,
                 "--speed", str(c.demo_speed)], stdout=subprocess.DEVNULL)
            self.directory = self._tmp
        if self.ds is None:
            self.ds = DetectorSet(self.directory, run=c.run)
        self.flt = CoincidenceFilter(self.ds, c.det_a, c.det_b, search_window_us=c.search_window_us,
                                     shot_period_us=c.shot_period_us)
        self.sel = SelectionFilter(self.ds, self.flt)
        self.pos = PositionCorrector(self.ds, self.flt, self.sel)
        self.img = Imager(self.ds, self.sel, binning=CartesianBinning(c.bins, c.bins), corrector=self.pos)
        self.polar = PolarAnalysis(self.ds, self.sel, corrector=self.pos)
        self.abel = AbelAnalysis(self.polar)
        if self.preset:
            from storage.capture import apply_settings
            apply_settings(self, self.preset, running=False)
        for stage, factor in ((self.ds, 1), (self.flt, 1), (self.sel, 2.5), (self.pos, 2.5), (self.img, 2.5), (self.polar, 2.5),
                              (self.abel, 2.5)):
            stage.start(c.poll_interval * factor)

    def stop(self):
        for stage in (self.abel, self.polar, self.img, self.pos, self.sel, self.flt, self.ds):
            if stage is not None:
                stage.stop()
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
        self.ds = self.flt = self.sel = self.pos = self.img = self.polar = self.abel = self._proc = self._tmp = None

    def restart_coincidence(self, search_window_us=None, cycle_shift="keep"):
        """Apply settings that need a full re-search (search window, forced cycle
        shift; None = auto). The other stages keep running meanwhile."""
        self.flt.stop()
        self.flt.reconfigure(search_window_us=search_window_us, cycle_shift=cycle_shift)
        self.flt.start(self.config.poll_interval)

    # -- read side --------------------------------------------------------
    def coincidence(self):
        return self.flt.result() if self.flt else None

    def selection(self):
        return self.sel.result() if self.sel else None

    def position_correction(self):
        return self.pos.result() if self.pos else None

    def imaging(self):
        return self.img.result() if self.img else None

    def polar_result(self):
        return self.polar.result() if self.polar else None

    def abel_result(self):
        return self.abel.result() if self.abel else None

    def errors(self):
        """Last exception of every worker thread that had one."""
        return [e for e in (getattr(s, "last_error", None)
                            for s in (self.ds, self.flt, self.sel, self.pos, self.img, self.polar, self.abel)) if e]

    def status_line(self):
        if not self.running:
            return "stopped"
        c = self.config
        r = self.coincidence()
        parts = []
        if self.ds is not None:
            snap = self.ds.snapshot()
            for name, det in (("A", c.det_a), ("B", c.det_b)):
                s = snap.detectors.get(det)
                parts.append(f"{name}=det {det}: {s.n if s else 0} rows")
        if r is not None:
            parts.append(f"coincidence [{r.status}] pulses {r.n_matched}, pairs {len(r.dt_us)}")
        else:
            parts.append("waiting for data from both detectors")
        if self.errors():
            parts.append("ERROR: " + self.errors()[0])
        return " | ".join(parts)