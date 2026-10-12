"""Live stage that turns the position-correlation correction into per-event positions for the
imaging chain (Qt-free).

    CoincidenceFilter -> SelectionFilter -> PositionCorrector -> Imager / PolarAnalysis -> AbelAnalysis

It fits the correction of the electron position (neutral position and dt, see poscorr.fit_correction;
optionally on the walk corrected dt', after the Sum / time cuts) and publishes, for every row of the
electron detector, the displacement (dx, dy) that the correction implies. Imagers subtract it from the
encoder positions, so centring, circularisation and the Abel inversion see the corrected image. Rows
without a partner in the window or a control window are not displaced. Off by default.
"""
import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from analysis.poscorr import apply_walk, cut_pairs, fit_correction, fit_walk, pair_displacements
from daq.data_loader import Poller


@dataclass(frozen=True)
class PosCorrSettings:
    enabled: bool = False
    side_e: str = "a"
    use_neutral: bool = True
    tau_order: int = 0
    background: bool = True          # weighted fit (accidentals subtracted) and control-window rows displaced
    walk: bool = False
    walk_sigma: float = 2.0
    walk_neutral: bool = False
    cuts: bool = True                # Sum / time cuts of the selection before the fit
    version: int = 0


@dataclass(frozen=True)
class PosCorrState:
    version: int
    enabled: bool
    side_e: str
    dx: dict                         # {'a': array or None, 'b': ...}: displacement per row
    dy: dict
    n_set: int                       # rows displaced
    correction: object               # poscorr.Correction (None if no fit was possible)
    walk: object
    run: object                      # (run, epoch_a, epoch_b) of the data the rows belong to
    coin_version: int
    sel_version: int

    @property
    def active(self):
        return self.enabled and self.correction is not None and self.dx.get(self.side_e) is not None

    def text(self):
        if not self.enabled:
            return "position correction not passed on"
        if self.correction is None:
            return "position correction: no fit possible yet (window or pairs missing)"
        return f"position correction passed on: {self.n_set} rows of detector {self.correction.det_e} displaced"


class PositionCorrector(Poller):
    def __init__(self, source, coincidence, selection):
        self.source, self.coincidence, self.selection = source, coincidence, selection
        self.det_a, self.det_b = selection.det_a, selection.det_b
        self._settings = PosCorrSettings()
        self._lock = threading.Lock()
        self._key, self._result, self._version = None, None, 0
        self._dur, self._t_done = 0.0, 0.0

    @property
    def settings(self):
        return self._settings

    def configure(self, **kw):
        """Change settings; no-op if nothing changes (so a GUI can push all of them on every edit)."""
        with self._lock:
            s = self._settings
            new = dataclasses.replace(s, **{k: v for k, v in kw.items() if k != "version"})
            if dataclasses.replace(new, version=s.version) == s:
                return
            self._settings = dataclasses.replace(new, version=s.version + 1)

    def result(self):
        return self._result

    def poll(self) -> int:
        s = self._settings
        coin, sel = self.coincidence.result(), self.selection.result()
        if sel is None or (s.enabled and coin is None):
            return 0
        key = (s.version, coin.version if coin is not None else None, sel.version)
        if key == self._key:
            return 0
        only_data = self._key is not None and self._key[0] == s.version
        if not s.enabled and only_data and self._result is not None and not self._result.enabled:
            self._key = key                                   # still off: nothing to publish
            return 0
        if s.enabled and only_data and time.monotonic() - self._t_done < max(1.0, 4.0 * self._dur):
            return 0                                          # live data: refit at most ~20% of the time
        t0 = time.monotonic()
        state = self._compute(s, coin, sel)
        self._dur = time.monotonic() - t0
        self._t_done = time.monotonic()
        self._key = key
        self._version += 1
        self._result = dataclasses.replace(state, version=self._version)
        return 1

    def _compute(self, s, coin, sel):
        run = (sel.run, sel.epoch_a, sel.epoch_b)
        empty = PosCorrState(0, s.enabled, s.side_e, {"a": None, "b": None}, {"a": None, "b": None}, 0, None, None,
                             run, coin.version if coin is not None else -1, sel.version)
        if not s.enabled or coin is None or coin.window_us is None:
            return empty
        snap = self.source.snapshot()
        da, db = self.det_a, self.det_b
        res = cut_pairs(coin, sel)[0] if s.cuts else coin
        walk = None
        res_w = res
        if s.walk:
            walk = fit_walk(res, snap, da, db, n_sigma=s.walk_sigma, use_neutral=s.walk_neutral, side_e=s.side_e)
            if walk is not None:
                res_w = apply_walk(res, snap, walk)
        window = coin.window_us
        cor = fit_correction(res_w, snap, da, db, side_e=s.side_e, window=window, use_neutral=s.use_neutral,
                             tau_order=s.tau_order, background=s.background)
        if cor is None:
            return dataclasses.replace(empty, walk=walk)
        dx, dy, n_set = pair_displacements(res, res_w, snap, da, db, cor, window, background=s.background)
        other = "b" if s.side_e == "a" else "a"
        return dataclasses.replace(empty, dx={s.side_e: dx, other: None}, dy={s.side_e: dy, other: None},
                                   n_set=n_set, correction=cor, walk=walk)
