"""Live Abel inversion of the data of the polar analysis stage (no Qt).

    PolarAnalysis.result()  --->  AbelAnalysis  --->  AbelAnalysisResult

For each detector one PolarAnalysis result holds, consistently: the encoder positions of all
events, their weights (background-subtracted, or 0/1), and the transform (centre and,
if applied, circularisation). From these this stage

  1. turns the events into r_i and cos(theta_i) about the centre, measured from the laser axis,
  2. bins them radially, d_{nr}  (abel.RadialBinner: done once per event; only new events are added),
  3. inverts d with the cached matrices (abel.AbelBasis): I_n, Legendre coefficients, I(rho), beta_n.

Defaults: angular orders 0, 1, 2. Order 1 is not expected for a mirror-symmetric distribution,
but keeping it absorbs small asymmetries of the data, so it is included.

The radial bins are those of the polar images: n_bins = nr and dr = r_max / nr.

Built for a fast, live bootstrap
--------------------------------
Everything after the per-event binning is linear in the event weights. Every published AbelSide
therefore carries what a bootstrap needs, frozen: the binned events (`binner`), the weights
(`weights`) and the matrices (`basis`). `AbelSide.resample(weights)` takes one weight vector
(E,) or a whole batch (B, E) of resampled weights and returns the inversion of each, in a few
matrix products, without touching the live state. Measured for 142k events and 3 orders: about
1 ms per resample for the binning plus 0.1 ms for the inversion, i.e. about a second for a thousand
resamples; the binning is the part a compiled kernel could still speed up.
"""
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from analysis.abel_loss import LossTerms
from analysis.abel import (AbelBasis, RadialBinner, angular_orders, cos_theta_from_xy, get_pT,
                               pT_cached)
from analysis.bootstrap import Bootstrap
from daq.data_loader import Poller


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class AbelSettings:
    n_max: int = 2                    # highest angular order
    even_only: bool = False           # drop the odd orders (n_max is then rounded down to even)
    axis_angle_deg: float = 90.0      # direction of the laser axis, counter-clockwise from +x
    tikhonov: float = 0.0             # 0 = exact inverse; > 0 regularises (see abel._inverse_operator)
    diff_order: int = 0
    min_intensity: float = 0.02       # beta is left out where I(rho) < this fraction of its maximum
    area: float = 1.0                 # area per angle (units dr^2); 1 gives absolute counts
    # further loss terms (abel_loss.LossTerms); all off = the plain / Tikhonov inversion
    weighting: bool = False           # statistical weights from the event covariance
    smooth: float = 0.0               # strength of the smoothness of I(rho); 0 = off
    smooth_order: int = 2             # 1 gradient, 2 curvature
    smooth_all: bool = False          # also smooth the higher Legendre orders
    positivity: str = "off"           # "off" | "intensity" (I >= 0) | "distribution" (f(rho, theta) >= 0)
    entropy: float = 0.0              # strength of the maximum-entropy (Poisson entropy) term; 0 = off
    entropy_floor: float = 0.01       # its default level, as a fraction of the peak

    @property
    def loss(self):
        return LossTerms(self.weighting, self.smooth, self.smooth_order, self.smooth_all, self.positivity,
                         self.entropy, self.entropy_floor)

    @property
    def orders(self):
        n = self.n_max - (self.n_max % 2) if self.even_only else self.n_max
        return angular_orders(max(n, 0), self.even_only)


@dataclass(frozen=True)
class AbelAnalysisSettings:
    a: AbelSettings = AbelSettings()
    b: AbelSettings = AbelSettings()
    version: int = 0


@dataclass(frozen=True)
class BootstrapSettings:
    n_boot: int = 2000
    unit: str = "event"               # "event" or "cycle"
    n_sigma: float = 1.0              # interval of +- n_sigma standard deviations
    auto: bool = False                # start a new run when the data grew by `growth`
    growth: float = 0.10


# ------------------------------------------------------------------- results
@dataclass(frozen=True)
class AbelSide:
    det: int
    side: str
    settings: AbelSettings
    orders: tuple
    n_bins: int
    dr: float
    rho: np.ndarray                 # radii of the bins, rho = R dr
    d: np.ndarray                   # (N, R) radially binned data d_{nr}: centred (+ circularised)
    d_centred: np.ndarray           # same, centred only (None if no circularisation is applied)
    abel: object                    # abel.AbelResult, None while the matrices are being built
    basis: object                   # abel.AbelBasis (None while being built)
    binner: object                  # frozen RadialBinner holding the events
    weights: np.ndarray             # event weights aligned with the binned events
    axis_angle: float               # rad
    n_events: int                   # events with non-zero weight
    total_weight: float             # sum of the weights inside the radial range
    circ_applied: bool
    status: str
    selection_version: int
    cycle: np.ndarray = None        # Sync-Time cycle index of every row (None: not available)

    @property
    def signature(self):
        """Everything a bootstrap result depends on besides the events: when it differs from the
        signature of the current result, the bootstrap belongs to other settings."""
        return (self.settings, self.n_bins, self.dr, self.circ_applied)

    def resample(self, weights):
        """Inversion of resampled event weights: (E,) -> AbelResult; (B, E) -> AbelResult with a
        leading batch dimension. E = number of events (len(self.weights))."""
        if self.basis is None:
            raise RuntimeError("the Abel matrices are still being built")
        d = self.binner.d(weights)
        return self.basis.invert(d, self.dr, self.settings.min_intensity)


@dataclass(frozen=True)
class AbelAnalysisResult:
    version: int
    a: AbelSide
    b: AbelSide

    def __getitem__(self, det):
        if det in ("a", self.a.det):
            return self.a
        if det in ("b", self.b.det):
            return self.b
        raise KeyError(det)


# ------------------------------------------------------------------- engine
class _Features:
    """Per-event r_i, cos(theta_i) binned into a RadialBinner; only new events are added while
    nothing that changes the coordinates (transform, axis, binning, data) has changed."""
    def __init__(self):
        self.key, self.binner, self.n = None, None, 0

    def update(self, key, transform, x, y, n, axis, n_bins, dr, orders):
        if key != self.key or self.binner is None or n < self.n:
            self.binner, self.n, self.key = RadialBinner(n_bins, dr, orders), 0, key
        if n > self.n:
            xt, yt = transform.apply(x[self.n:n], y[self.n:n])
            r, c = cos_theta_from_xy(xt, yt, axis)
            self.binner.append(r, c)
            self.n = n
        return self.binner


class _State:
    def __init__(self):
        self.main, self.centred = _Features(), _Features()
        self.basis, self.basis_key = None, None


class AbelAnalysis(Poller):
    """Live Abel inversion for both detectors. Detectors are addressed by number or 'a' / 'b'."""
    def __init__(self, polar):
        self.polar = polar
        self.det_a, self.det_b = polar.det_a, polar.det_b
        self._settings = AbelAnalysisSettings()
        self._lock = threading.Lock()
        self._state = {"a": _State(), "b": _State()}
        self._building, self._built = {}, 0          # background matrix builds / how many finished
        self._key, self._result, self._version = None, None, 0
        self._cfg_key, self._t_done, self._wait = None, 0.0, 0.0      # throttle for the costly loss terms
        self._boot = {"a": None, "b": None}          # the latest Bootstrap of each side
        self._boot_cfg = {"a": BootstrapSettings(), "b": BootstrapSettings()}

    # -- settings --------------------------------------------------------
    @property
    def settings(self):
        return self._settings

    def _side(self, det):
        if det in ("a", self.det_a):
            return "a"
        if det in ("b", self.det_b):
            return "b"
        raise KeyError(f"unknown detector {det!r}")

    def _change(self, side, **kw):
        with self._lock:
            s = self._settings
            k = self._side(side)
            self._settings = replace(s, version=s.version + 1, **{k: replace(getattr(s, k), **kw)})

    def set_orders(self, side, n_max=None, even_only=None):
        kw = {}
        if n_max is not None:
            kw["n_max"] = int(n_max)
        if even_only is not None:
            kw["even_only"] = bool(even_only)
        self._change(side, **kw)

    def set_axis_angle(self, side, degrees):
        self._change(side, axis_angle_deg=float(degrees))

    def set_regularisation(self, side, tikhonov, diff_order=0):
        self._change(side, tikhonov=float(tikhonov), diff_order=int(diff_order))

    def set_loss(self, side, **kw):
        """weighting, smooth, smooth_order, smooth_all, positivity, entropy, entropy_floor: see abel_loss.LossTerms."""
        allowed = {"weighting", "smooth", "smooth_order", "smooth_all", "positivity", "entropy", "entropy_floor"}
        if set(kw) - allowed:
            raise TypeError(f"unknown loss setting(s): {sorted(set(kw) - allowed)}")
        if "weighting" in kw:
            kw["weighting"] = bool(kw["weighting"])
        if "smooth" in kw:
            kw["smooth"] = float(kw["smooth"])
        for key in ("entropy", "entropy_floor"):
            if key in kw:
                kw[key] = float(kw[key])
        if "smooth_order" in kw:
            kw["smooth_order"] = int(kw["smooth_order"])
        if "smooth_all" in kw:
            kw["smooth_all"] = bool(kw["smooth_all"])
        LossTerms(**{k: kw[k] for k in kw})                      # validates
        self._change(side, **kw)

    def set_min_intensity(self, side, fraction):
        self._change(side, min_intensity=float(fraction))

    def set_area(self, side, area):
        self._change(side, area=float(area))

    # -- evaluation ------------------------------------------------------
    def result(self):
        return self._result

    def _ensure_matrices(self, n_bins, orders):
        """Build the (expensive) projected basis in a background thread; True when it is ready."""
        if pT_cached(n_bins, orders):
            return True
        key = (n_bins, tuple(orders))
        with self._lock:
            if key not in self._building:
                def job():
                    try:
                        get_pT(n_bins, orders)
                    finally:
                        with self._lock:
                            self._building.pop(key, None)
                            self._built += 1
                th = threading.Thread(target=job, daemon=True)
                self._building[key] = th
                th.start()
        return False

    # -- bootstrap ---------------------------------------------------------
    def bootstrap_settings(self, side):
        return self._boot_cfg[self._side(side)]

    def set_bootstrap(self, side, **kw):
        """n_boot, unit, n_sigma, auto, growth. Applies to the next run."""
        k = self._side(side)
        self._boot_cfg[k] = replace(self._boot_cfg[k], **kw)

    def start_bootstrap(self, side, seed=None):
        """Bootstrap the current Abel result of this side (replaces a run in progress). Returns the
        Bootstrap, or None if there is no inversion to resample yet."""
        k = self._side(side)
        res = self._result
        if res is None or res[k].abel is None:
            return None
        self.stop_bootstrap(k)
        cfg = self._boot_cfg[k]
        boot = Bootstrap(res[k], n_boot=cfg.n_boot, unit=cfg.unit, n_sigma=cfg.n_sigma, seed=seed)
        self._boot[k] = boot
        return boot.start()

    def stop_bootstrap(self, side):
        b = self._boot[self._side(side)]
        if b is not None:
            b.stop()

    def bootstrap_result(self, side):
        b = self._boot[self._side(side)]
        return None if b is None else b.result()

    def bootstrap_running(self, side):
        b = self._boot[self._side(side)]
        return b is not None and b.running

    def _auto_bootstrap(self):
        """Keep the bands fresh for the sides that asked for it."""
        res = self._result
        if res is None:
            return
        for k in ("a", "b"):
            cfg, cur, b = self._boot_cfg[k], res[k], self._boot[k]
            if not cfg.auto or cur.abel is None or cur.n_events == 0 or (b is not None and b.running):
                continue
            last = None if b is None else b.result()
            if (last is None or last.signature != cur.signature or last.unit != cfg.unit
                    or last.n_total != cfg.n_boot or last.n_sigma != cfg.n_sigma
                    or cur.n_events >= last.n_events * (1.0 + cfg.growth) + 1):
                self.start_bootstrap(k)

    def stop(self):
        for k in ("a", "b"):
            self.stop_bootstrap(k)
            if self._boot[k] is not None:
                self._boot[k].join(10)
        super().stop()

    def poll(self) -> int:
        self._auto_bootstrap()
        pr = self.polar.result()
        if pr is None:
            return 0
        s = self._settings
        key = (pr.version, s.version, self._built)
        if key == self._key:
            return 0
        cfg_key = (s.version, self._built)
        if cfg_key == self._cfg_key and time.monotonic() - self._t_done < self._wait:
            return 0                                   # only the data grew, and the last inversion was slow
        t0 = time.monotonic()
        sides = {k: self._side_result(k, pr[k], getattr(s, k)) for k in ("a", "b")}
        if sides["a"] is None or sides["b"] is None:
            return 0
        self._t_done = time.monotonic()
        dur = self._t_done - t0
        self._wait = 0.0 if dur < 0.1 else max(1.0, 4.0 * dur)      # refit at most ~20 % of the time
        self._cfg_key = cfg_key
        self._key = key
        self._version += 1
        self._result = AbelAnalysisResult(self._version, sides["a"], sides["b"])
        return 1

    def _side_result(self, k, v, cfg):
        if v.x is None or v.weights is None or v.transform is None:
            return None
        st = self._state[k]
        n = min(len(v.x), len(v.weights))
        orders = cfg.orders
        n_bins = v.binning.nr
        dr = v.binning.r_max / n_bins
        axis = np.radians(cfg.axis_angle_deg)
        common = (v.data_id, axis, n_bins, dr, orders)
        binner = st.main.update((v.transform,) + common, v.transform, v.x, v.y, n, axis, n_bins, dr, orders)
        w = np.asarray(v.weights[:n], float)
        d = binner.d(w)
        d_centred = None
        if v.circ_applied:
            b2 = st.centred.update((v.transform_centred,) + common, v.transform_centred, v.x, v.y, n,
                                   axis, n_bins, dr, orders)
            d_centred = b2.d(w)

        basis, abel = None, None
        status = "running"
        if self._ensure_matrices(n_bins, orders):
            loss = cfg.loss
            bkey = (n_bins, orders, cfg.tikhonov, cfg.diff_order, cfg.area, cfg.even_only, loss)
            if st.basis_key != bkey:
                st.basis = AbelBasis(n_bins, n_max=orders[-1], even_only=cfg.even_only,
                                     tikhonov=cfg.tikhonov, diff_order=cfg.diff_order, area=cfg.area,
                                     loss=None if loss.weighting else loss)
                st.basis_key = bkey
            basis = st.basis
            if loss.weighting:
                # the weights come from this measurement's own covariance: a new basis object each time
                basis = basis.with_loss(loss, binner.covariance(w))
            abel = basis.invert(d, dr, cfg.min_intensity)
        else:
            status = f"building the Abel matrices for {n_bins} bins ..."
        if not np.any(w):
            status = "no events yet"
        return AbelSide(
            det=v.det, side=k, settings=cfg, orders=orders, n_bins=n_bins, dr=dr,
            rho=np.arange(n_bins) * dr, d=d, d_centred=d_centred, abel=abel, basis=basis,
            binner=binner.snapshot(), weights=w, axis_angle=axis, n_events=int(np.count_nonzero(w)),
            total_weight=float(d[0].sum()) if orders[0] == 0 else float("nan"),
            circ_applied=v.circ_applied, status=status, selection_version=v.selection_version,
            cycle=None if v.cycle is None else v.cycle[:n])
