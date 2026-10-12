"""Centering and circularisation, then polar-binned images (no Qt).

Centering
---------
Project the image on x and on y. A projection p is symmetric about the centre c,
p(c + u) = p(c - u). Its self-convolution q(s) = sum_x p(x) p(s - x) is then
maximal at s = 2c (Cauchy-Schwarz), so c = s_peak / 2, refined to sub-bin accuracy
with a parabola. This is done for x and y separately. ('centroid' is offered as a
simple baseline.)

Circularisation
---------------
Work on the polar histogram H[k, i] (angular bin k, radial bin i). A scalar stretch
s_k per angular bin (events move r -> s_k r) is chosen so that the radial profile of
bin k matches a target, in the least-squares sense over a radial range, with a free
amplitude per bin (the angular distribution changes the intensity).
  * The reference slice (angle theta_ref, snapped to the centre of its bin) defines
    two mirror planes: theta_ref and theta_ref + 90 deg. s = 1 there.
  * From the reference we sweep 90 degrees (ccw or cw) bin by bin; each bin is matched
    to the previous, already stretched, bin: this fixes the first quadrant.
  * Every other bin is mirrored into that quadrant and matched to its mirror image there
    (independently, no chaining), so all four quadrants agree.
Stretching a profile re-bins it onto the old radial bins conserving counts.
The result is a table s(theta), applied to events by imaging.Circularise as a smooth
periodic function of the event's angle. It is therefore independent of the angular
binning of any image made afterwards, and tables made with a different number of
angular bins can be loaded as they are.
"""
import json
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from analysis.imaging import (CartesianBinning, Chain, Circularise, Imager, PolarBinning,
                                  Shift)
from daq.data_loader import Poller

SYN = "Sync Time"


# ------------------------------------------------------------------ centering
@dataclass(frozen=True)
class CenterDiag:
    method: str
    x: np.ndarray                # bin centres along x / y
    px: np.ndarray               # projection on x (sum over y), negatives clipped
    cx: float
    y: np.ndarray
    py: np.ndarray
    cy: float


def _axis_center(p, c0, dx, method):
    p = np.clip(p, 0, None)
    if p.sum() <= 0:
        return None
    if method == "centroid":
        return c0 + dx * float(np.arange(len(p)) @ p) / p.sum()
    q = np.convolve(p, p)                      # index k  <->  pair sum 2*c0 + k*dx
    k = int(np.argmax(q))
    w = max(2, len(p) // 64)
    lo, hi = max(k - w, 0), min(k + w + 1, len(q))
    a, b, _ = np.polyfit(np.arange(lo, hi) - k, q[lo:hi], 2)
    kk = k if a >= 0 else float(np.clip(k - b / (2 * a), k - w, k + w))
    return c0 + dx * kk / 2.0


def estimate_center(image, extent, method="convolution"):
    """(cx, cy, CenterDiag) from a Cartesian image [iy, ix] with extent (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = extent
    ny, nx = image.shape
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    px, py = np.clip(image, 0, None).sum(axis=0), np.clip(image, 0, None).sum(axis=1)
    cx = _axis_center(px, x0 + dx / 2, dx, method)
    cy = _axis_center(py, y0 + dy / 2, dy, method)
    if cx is None or cy is None:
        return None
    return cx, cy, CenterDiag(method, x0 + dx * (np.arange(nx) + 0.5), px, cx,
                              y0 + dy * (np.arange(ny) + 0.5), py, cy)


# ------------------------------------------------------------ circularisation
@dataclass(frozen=True)
class CircParams:
    theta_ref_deg: float = 0.0     # reference slice; snapped to the centre of its bin
    direction: int = 1             # +1 counter-clockwise, -1 clockwise
    r_lo: float = None             # radial range used for matching (None = from 5% of r_max)
    r_hi: float = None             # (None = r_max)
    s_min: float = 0.8
    s_max: float = 1.25
    n_grid: int = 401
    free_amplitude: bool = True
    smooth_bins: float = 1.0       # Gaussian smoothing of the radial profiles [bins]
    max_step: float = 0.05         # chain: s may change at most this fraction from one bin to
                                   # the next (None = unlimited). Guards against run-away on
                                   # featureless profiles.
    harmonics: int = 0             # >0: project s(theta) onto cos(2 n phi), n = 0..harmonics
                                   # (what both mirror planes allow); 0 = keep every bin as fitted


@dataclass(frozen=True)
class CircTable:
    """Stretch factor s at the bin-centre angles, with what is needed to judge it later."""
    theta: tuple                   # rad, ascending in 0..2pi
    s: tuple
    similarity: tuple              # per bin: 1 = profile matches its target perfectly
    center: tuple                  # (cx, cy) the table was made around
    detector: int
    theta_ref_deg: float
    direction: int
    r_range: tuple
    n_theta: int
    created: str
    n_events: float
    source: str = "computed"
    n_at_limit: int = 0            # bins whose s sits at the search limits (fit not trustworthy)

    def transform(self):
        return Circularise(self.theta, self.s)

    def describe(self):
        s = np.asarray(self.s)
        return (f"{self.source}: {self.n_theta} angular bins, s {s.min():.4f}..{s.max():.4f}, "
                f"centre ({self.center[0]:.4f}, {self.center[1]:.4f}), reference "
                f"{self.theta_ref_deg:.1f} deg {'ccw' if self.direction > 0 else 'cw'}, "
                f"r {self.r_range[0]:.3f}..{self.r_range[1]:.3f}, "
                f"median similarity {np.median(self.similarity):.3f}"
                + (f", WARNING {self.n_at_limit} bins at the search limits" if self.n_at_limit else ""))


def _gauss(profiles, sigma):
    if sigma <= 0:
        return profiles
    n = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(profiles, ((0, 0), (n, n)), mode="edge")
    return np.array([np.convolve(row, k, mode="valid") for row in pad])


def _fold(k, k0, d, K):
    """Quadrant-1 bin that bin k maps to under the two mirror planes."""
    phi = ((k - k0) * d) % K
    u = phi % (K // 2)
    if u > K // 4:
        u = K // 2 - u
    return (k0 + d * u) % K


def circularise_fit(H, r_edges, p: CircParams, center=(0.0, 0.0), detector=0):
    """Estimate s(theta) from a polar histogram H[k, i] (see module docstring)."""
    H = np.asarray(H, float)
    K, R = H.shape
    if K % 4:
        raise ValueError(f"number of angular bins must be a multiple of 4, got {K}")
    d = 1 if p.direction >= 0 else -1
    dth = 2 * np.pi / K
    k0 = int(np.round(np.radians(p.theta_ref_deg) / dth - 0.5)) % K
    prof = _gauss(H, p.smooth_bins)
    cum = np.concatenate([np.zeros((K, 1)), np.cumsum(prof, axis=1)], axis=1)   # counts below r_edges[j]
    r_hi = p.r_hi if p.r_hi is not None else r_edges[-1]
    r_lo = p.r_lo if p.r_lo is not None else 0.05 * r_edges[-1]
    centres = 0.5 * (r_edges[1:] + r_edges[:-1])
    sl = (centres >= r_lo) & (centres <= r_hi)
    if sl.sum() < 5:
        raise ValueError("radial fit range contains fewer than 5 bins")
    sg = np.exp(np.linspace(np.log(p.s_min), np.log(p.s_max), p.n_grid))
    dlog = np.log(sg[1] / sg[0])
    edges_grid = r_edges[None, :] / sg[:, None]                                  # (S, R+1)

    def stretched(k, s):
        """Profile of bin k after r -> s r, re-binned to the old bins (counts conserved)."""
        e = r_edges / s
        return np.diff(np.interp(e, r_edges, cum[k]))

    def fit(k, target, around=None):
        G = np.diff(np.interp(edges_grid.ravel(), r_edges, cum[k]).reshape(len(sg), R + 1), axis=1)[:, sl]
        T = target[sl]
        gg, gt, tt = (G * G).sum(1), G @ T, T @ T
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = tt - gt ** 2 / gg if p.free_amplitude else tt - 2 * gt + gg
            sim = gt ** 2 / (gg * tt)
        cost = np.where(np.isfinite(cost) & (gg > 0), cost, np.inf)
        if around is not None and p.max_step is not None:
            cost = np.where((sg >= around / (1 + p.max_step)) & (sg <= around * (1 + p.max_step)), cost, np.inf)
        i = int(np.argmin(cost))
        if not np.isfinite(cost[i]) or tt <= 0:
            return 1.0, 0.0
        s = sg[i]
        if 0 < i < len(sg) - 1 and np.isfinite(cost[i - 1]) and np.isfinite(cost[i + 1]):
            den = cost[i - 1] - 2 * cost[i] + cost[i + 1]
            if den > 0:
                s = np.exp(np.log(sg[i]) + 0.5 * (cost[i - 1] - cost[i + 1]) / den * dlog)
        return float(s), float(np.nan_to_num(sim[i]))

    s_out, sim_out = np.ones(K), np.zeros(K)
    done = np.zeros(K, bool)
    final = np.zeros((K, R))                             # stretched profiles in the reference frame
    final[k0], done[k0], sim_out[k0] = prof[k0], True, 1.0
    prev = k0
    for m in range(1, K // 4 + 1):                       # sweep 90 degrees from the reference
        k = (k0 + d * m) % K
        s_out[k], sim_out[k] = fit(k, final[prev], around=s_out[prev])
        final[k], done[k], prev = stretched(k, s_out[k]), True, k
    for k in range(K):                                   # everything else: via the mirror planes
        if not done[k]:
            s_out[k], sim_out[k] = fit(k, final[_fold(k, k0, d, K)])
    theta = (np.arange(K) + 0.5) * dth
    if p.harmonics > 0:                      # cos(2 n phi): even about both mirror planes
        phi = theta - (k0 + 0.5) * dth
        A = np.cos(2 * np.outer(phi, np.arange(p.harmonics + 1)))
        n_k = prof[:, sl].sum(1)                        # bins with few counts (angular nodes,
        w = np.sqrt(n_k / max(n_k.max(), 1e-12)) * np.clip(sim_out, 0.0, 1.0) ** 8 + 1e-6   # background) count little
        coef = np.linalg.lstsq(A * w[:, None], s_out * w, rcond=None)[0]
        s_out = A @ coef
    n_limit = int(np.sum((s_out <= p.s_min * 1.001) | (s_out >= p.s_max / 1.001)))
    return CircTable(theta=tuple(theta), s=tuple(s_out), similarity=tuple(sim_out), n_at_limit=n_limit,
                     center=(float(center[0]), float(center[1])), detector=int(detector),
                     theta_ref_deg=float(np.degrees((k0 + 0.5) * dth)), direction=d,
                     r_range=(float(r_lo), float(r_hi)), n_theta=K,
                     created=time.strftime("%Y-%m-%d %H:%M:%S"), n_events=float(H.sum()))


# ------------------------------------------------------------------ save / load
def save_circ(path, t: CircTable):
    doc = dict(format="vmi-circularisation", version=1, detector=t.detector, center=list(t.center),
               theta_ref_deg=t.theta_ref_deg, direction=t.direction, r_range=list(t.r_range),
               n_theta=t.n_theta, created=t.created, n_events=t.n_events,
               theta_deg=[float(np.degrees(v)) for v in t.theta], stretch=list(t.s),
               similarity=list(t.similarity), n_at_limit=t.n_at_limit,
               note="stretch s(theta): events at angle theta (counter-clockwise from +x, "
                    "about `center`) move r -> s*r. Interpolated periodically when applied.")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def load_circ(path) -> CircTable:
    with open(path) as f:
        d = json.load(f)
    if d.get("format") != "vmi-circularisation":
        raise ValueError("not a circularisation file")
    th = np.radians(np.asarray(d["theta_deg"], float))
    s = np.asarray(d["stretch"], float)
    if len(th) != len(s) or len(th) < 4 or not (np.all(np.isfinite(s)) and np.all((s > 0.2) & (s < 5))):
        raise ValueError("invalid stretch table")
    order = np.argsort(np.mod(th, 2 * np.pi))
    th, s = np.mod(th, 2 * np.pi)[order], s[order]
    sim = np.asarray(d.get("similarity", np.zeros(len(s))), float)[order]
    return CircTable(theta=tuple(th), s=tuple(s), similarity=tuple(sim), center=tuple(d["center"]),
                     detector=int(d.get("detector", 0)), theta_ref_deg=float(d["theta_ref_deg"]),
                     direction=int(d["direction"]), r_range=tuple(d["r_range"]),
                     n_theta=int(d["n_theta"]), created=str(d.get("created", "")),
                     n_events=float(d.get("n_events", 0)), source=f"loaded from {path}",
                     n_at_limit=int(d.get("n_at_limit", 0)))


def compatibility(t: CircTable, center, detector, binning=None, tol=0.003):
    """Human-readable warnings when a table is used under other conditions."""
    w = []
    if t.detector and detector and t.detector != detector:
        w.append(f"table was made for detector {t.detector}, used for {detector}")
    shift = float(np.hypot(center[0] - t.center[0], center[1] - t.center[1]))
    if shift > tol:
        w.append(f"current centre differs from the table's by {shift:.4f} "
                 f"(table {t.center[0]:.4f}, {t.center[1]:.4f}; now {center[0]:.4f}, {center[1]:.4f}); "
                 "the correction assumes the table's centre")
    if binning is not None and t.r_range[1] > binning.r_max * 1.001:
        w.append(f"table was fitted out to r = {t.r_range[1]:.3f}, beyond the current r_max {binning.r_max:.3f}")
    if binning is not None and binning.ntheta != t.n_theta:
        w.append(f"table has {t.n_theta} angular bins, images use {binning.ntheta}: "
                 "fine, s(theta) is interpolated at each event's angle")
    return w


# ---------------------------------------------------------------- live stage
@dataclass(frozen=True)
class SideSettings:
    center_mode: str = "auto"          # 'auto' (live estimate) or 'manual'
    center: tuple = (0.5, 0.5)         # used when manual
    estimator: str = "convolution"     # or 'centroid'
    nr: int = 200
    ntheta: int = 96                   # multiple of 4
    r_max: float = 0.5
    circ_enabled: bool = False
    circ: CircTable = None


@dataclass(frozen=True)
class PolarSettings:
    a: SideSettings = SideSettings()
    b: SideSettings = SideSettings()
    version: int = 0


@dataclass(frozen=True)
class SideView:
    det: int
    side: str
    center: tuple                      # centre in use
    center_auto: tuple                 # latest automatic estimate (or None)
    center_mode: str
    diag: CenterDiag
    binning: PolarBinning
    raw: np.ndarray                    # polar histogram, centred only      [itheta, ir]
    corr: np.ndarray                   # polar histogram, centred + circularised
    circ: CircTable
    circ_applied: bool
    warnings: tuple
    # What the next stage (Abel inversion) needs, taken from the same selection and transform as the
    # histograms above, so that one result is one consistent snapshot:
    x: np.ndarray = None               # encoder positions of all rows (before any transform)
    y: np.ndarray = None
    weights: np.ndarray = None         # event weight of every row, according to the events mode
    data_id: tuple = None              # identifies the data (run, file epoch) the rows belong to
    selection_version: int = None
    transform: object = None           # detector -> centred (+ circularised) coordinates
    transform_centred: object = None   # detector -> centred coordinates only
    mode: str = "subtracted"
    cycle: np.ndarray = None           # index of the Sync-Time cycle (about 1 s) of every row, or None


@dataclass(frozen=True)
class PolarResult:
    version: int
    a: SideView
    b: SideView

    def __getitem__(self, det):
        if det in ("a", self.a.det):
            return self.a
        if det in ("b", self.b.det):
            return self.b
        raise KeyError(det)


class _PinnedSelection:
    """Gives the three imagers of PolarAnalysis the same selection result during one poll."""
    def __init__(self, selection):
        self.det_a, self.det_b = selection.det_a, selection.det_b
        self.sel = None

    def result(self):
        return self.sel


def event_weights(sel, side, mode):
    """Weight of every row of a SelectionResult for the events mode of the imagers:
    'subtracted' -> background-subtracted weights, 'raw' -> 1 for selected rows, 'all' -> 1 for searched rows."""
    valid = getattr(sel, f"valid_{side}")
    if mode == "all":
        return valid.astype(float)
    if mode == "raw":
        return (valid & getattr(sel, f"mask_{side}")).astype(float)
    return getattr(sel, f"weight_{side}")


class PolarAnalysis(Poller):
    """Centre -> (circularise) -> polar images, live, for both detectors.

    Owns three Imagers fed by the same SelectionFilter: a fixed Cartesian one for the
    centre estimate, and two polar ones (centred only / centred + circularised).
    Detectors are addressed by number or 'a' / 'b'. Circularisation is computed
    only on request (compute_circularisation), never on every update."""
    CART = CartesianBinning(512, 512, (0.0, 1.0), (0.0, 1.0))

    def __init__(self, source, selection, mode="subtracted", corrector=None):
        self.source = source
        self.selection = selection
        self.det_a, self.det_b = selection.det_a, selection.det_b
        self.mode = mode
        self._pin = _PinnedSelection(selection)
        self.cart = Imager(source, self._pin, binning=self.CART, mode=mode, corrector=corrector)
        self.raw = Imager(source, self._pin, mode=mode, corrector=corrector)
        self.corr = Imager(source, self._pin, mode=mode, corrector=corrector)
        self._settings = PolarSettings()
        self._lock = threading.Lock()
        self._status = {"a": "not computed", "b": "not computed"}
        self._auto = {"a": None, "b": None}
        self._key, self._result, self._version = None, None, 0

    # -- settings --------------------------------------------------------
    @property
    def settings(self):
        return self._settings

    def status(self, side):
        return self._status[self._side(side)]

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

    def set_mode(self, mode):
        self.mode = mode
        for im in (self.cart, self.raw, self.corr):
            im.set_mode(mode)

    def set_binning(self, side, nr=None, ntheta=None, r_max=None):
        cur = getattr(self._settings, self._side(side))
        kw = dict(nr=int(nr or cur.nr), r_max=float(r_max or cur.r_max))
        n = int(ntheta or cur.ntheta)
        kw["ntheta"] = max(4, 4 * int(round(n / 4)))             # multiple of 4
        self._change(side, **kw)

    def set_center_mode(self, side, mode):
        if mode not in ("auto", "manual"):
            raise ValueError(mode)
        self._change(side, center_mode=mode)

    def set_center(self, side, cx, cy):
        """Manual centre (switches to manual mode)."""
        self._change(side, center_mode="manual", center=(float(cx), float(cy)))

    def lock_center(self, side):
        """Freeze the centre currently in use."""
        res = self._result
        if res is not None:
            self.set_center(side, *res[side].center)

    def set_estimator(self, side, method):
        if method not in ("convolution", "centroid"):
            raise ValueError(method)
        self._change(side, estimator=method)

    def fit_r_max(self, side, quantile=99.9, pad=1.05):
        """One-off: r_max from the radius (about the current centre) of the selected events."""
        k = self._side(side)
        pos = self.cart.positions(self.det_a if k == "a" else self.det_b)
        res = self._result
        if pos is None or res is None or len(pos[0]) < 10:
            raise RuntimeError("no data yet")
        cx, cy = res[k].center
        r = np.hypot(pos[0] - cx, pos[1] - cy)
        self.set_binning(side, r_max=float(np.nanpercentile(r, quantile)) * pad)

    # -- circularisation ----------------------------------------------------
    def compute_circularisation(self, side, params: CircParams = CircParams(), wait=False, timeout=10.0):
        """Estimate s(theta) from the centred polar histogram, in a worker thread, and
        apply the table as soon as it is ready. Uses the binning currently set: if the
        published image still has an older binning it waits for the next update."""
        k = self._side(side)
        if self._result is None:
            raise RuntimeError("no data yet")
        ss = getattr(self._settings, k)
        target = PolarBinning(ss.nr, ss.ntheta, ss.r_max)
        self._status[k] = "computing ..."

        def job():
            try:
                t0 = time.time()
                while True:
                    res = self._result
                    if res is not None and res[k].binning == target:
                        break
                    if not self.running or time.time() - t0 > timeout:
                        raise RuntimeError("the images have not been updated to the new binning")
                    time.sleep(0.05)
                v = res[k]
                t = circularise_fit(v.raw.copy(), v.binning.r_edges(), params, v.center, v.det)
                self._change(side, circ=t, circ_enabled=True)
                self._status[k] = t.describe()
            except Exception as exc:                       # report, do not kill the GUI
                self._status[k] = f"failed: {exc}"
        th = threading.Thread(target=job, daemon=True)
        th.start()
        if wait:
            th.join()

    def set_circ(self, side, table, enabled=True):
        self._change(side, circ=table, circ_enabled=bool(enabled and table is not None))
        if table is not None:
            self._status[self._side(side)] = table.describe()

    def enable_circ(self, side, on=True):
        self._change(side, circ_enabled=bool(on))

    def clear_circ(self, side):
        self._change(side, circ=None, circ_enabled=False)
        self._status[self._side(side)] = "not computed"

    def save_circ(self, side, path):
        t = getattr(self._settings, self._side(side)).circ
        if t is None:
            raise RuntimeError("no circularisation to save")
        save_circ(path, t)

    def load_circ(self, side, path):
        """Load a table and apply it. Returns the compatibility warnings."""
        t = load_circ(path)
        k = self._side(side)
        ss = getattr(self._settings, k)
        res = self._result
        center = res[k].center if res is not None else ss.center
        warns = compatibility(t, center, self.det_a if k == "a" else self.det_b,
                              PolarBinning(ss.nr, ss.ntheta, ss.r_max))
        self.set_circ(side, t, True)
        return warns

    # -- evaluation ------------------------------------------------------
    def result(self):
        return self._result

    def _cycles(self, sel):
        """Cycle index of every row of each side (constant Sync Time = one cycle), taken from the
        source. {} if the source is not at least as far as the selection (then cycle resampling
        is simply unavailable for this result)."""
        out = {}
        try:
            snap = self.source.snapshot()
            for k, det in (("a", self.det_a), ("b", self.det_b)):
                xy = self.cart.raw_xy(det)
                sync = snap.detectors[det].columns[SYN]
                if xy is not None and len(sync) >= len(xy[0]):
                    out[k] = np.cumsum(np.r_[0, np.diff(sync[:len(xy[0])]) != 0]).astype(np.int64)
        except Exception:
            return {}
        return out

    def poll(self) -> int:
        sel = self.selection.result()
        if sel is None:
            return 0
        self._pin.sel = sel                  # all imagers evaluate this very selection
        self.cart.poll()
        cart = self.cart.result()
        if cart is None:
            return 0
        s = self._settings
        centers, diags, binnings, t_raw, t_corr = {}, {}, {}, {}, {}
        for k, det in (("a", self.det_a), ("b", self.det_b)):
            ss = getattr(s, k)
            img = cart[k]
            est = estimate_center(img.image, img.binning.extent, ss.estimator)
            self._auto[k] = None if est is None else (est[0], est[1])
            diags[k] = None if est is None else est[2]
            centers[k] = ss.center if (ss.center_mode == "manual" or est is None) else self._auto[k]
            binnings[k] = PolarBinning(ss.nr, ss.ntheta, ss.r_max)
            t_raw[k] = Shift(*centers[k])
            use = ss.circ_enabled and ss.circ is not None
            t_corr[k] = Chain((Shift(*centers[k]), ss.circ.transform())) if use else t_raw[k]
        for k, det in (("a", self.det_a), ("b", self.det_b)):
            for im, tr in ((self.raw, t_raw[k]), (self.corr, t_corr[k])):
                if getattr(im.settings, f"binning_{k}") != binnings[k]:
                    im.set_binning(binnings[k], det)
                if getattr(im.settings, f"transform_{k}") != tr:
                    im.set_transform(tr, det)
        self.raw.poll()
        self.corr.poll()
        raw, corr = self.raw.result(), self.corr.result()
        if raw is None or corr is None:
            return 0
        key = (cart.version, raw.version, corr.version, s.version)
        if key == self._key:
            return 0
        views = {}
        cycles = self._cycles(sel)
        for k, det in (("a", self.det_a), ("b", self.det_b)):
            ss = getattr(s, k)
            use = ss.circ_enabled and ss.circ is not None
            warns = tuple(compatibility(ss.circ, centers[k], det, binnings[k])) if use else ()
            xy = self.cart.raw_xy(det)
            views[k] = SideView(det=det, side=k, center=centers[k], center_auto=self._auto[k],
                                center_mode=ss.center_mode, diag=diags[k], binning=binnings[k],
                                raw=raw[k].image, corr=corr[k].image, circ=ss.circ,
                                circ_applied=use, warnings=warns,
                                x=xy[0] if xy else None, y=xy[1] if xy else None,
                                weights=event_weights(sel, k, self.mode),
                                data_id=(sel.run, sel.epoch_a if k == "a" else sel.epoch_b),
                                selection_version=sel.version, transform=t_corr[k],
                                transform_centred=t_raw[k], mode=self.mode,
                                cycle=cycles.get(k))
        self._key = key
        self._version += 1
        self._result = PolarResult(self._version, views["a"], views["b"])
        return 1
