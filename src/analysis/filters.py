"""Charge (Sum) and event-time filters for the two coincidence detectors.

Cuts per detector
-----------------
  Sum   [lo, hi]   inclusive limits on the 'Sum' column (either may be None).
  time  [lo, hi]   seconds after that detector's FIRST event (first event = 0).

Where the Sum cut acts ("stage")
--------------------------------
  'pre'   before coincidence: only events passing their Sum cut can form pairs.
          An A event is kept if it passes ITS cut and has a partner in the
          window that passes ITS cut. Junk events (e.g. Sum = 0) no longer make
          accidentals, and the peak, window and accidental estimate are all
          recomputed from the surviving pairs.
  'post'  after coincidence: partners are searched among all events; the cut
          then removes events of its own detector only.
Pairs depend only on their two events, so 'pre' is done by restricting the stored
pair list. No coincidence re-run is needed and switching is instant.

The time cut is ALWAYS applied last, to the final selection of each detector.

Adaptivity
----------
  coincidence -> charge : result.histogram(det, 'sum', kind='true') is the Sum
      spectrum of true pairs (background-subtracted, independent of the current
      cuts); auto_sum() derives limits from it and keeps them up to date.
  charge -> coincidence : with stage 'pre', result.window_us, the peak and the
      accidental fraction follow the cuts (compare stats vs base_stats).

Everything is re-evaluated from immutable inputs (loader snapshot, coincidence
result, settings), so changing a limit takes effect on the next poll.
"""
import dataclasses
import threading
from dataclasses import dataclass

import numpy as np

from daq.data_loader import Poller
from analysis.coincidence import EVT

SUM = "Sum"


# ----------------------------------------------------------------- settings
@dataclass(frozen=True)
class Cuts:
    sum_lo: float = None
    sum_hi: float = None
    t_lo: float = None            # seconds after the detector's first event
    t_hi: float = None
    sum_stage: str = "pre"        # 'pre' (before coincidence) or 'post'
    sum_auto: tuple = None        # (q_lo, q_hi) quantiles of the true-pair spectrum;
                                  # either may be None (= no limit on that side)


@dataclass(frozen=True)
class Settings:
    a: Cuts = Cuts()
    b: Cuts = Cuts()
    use_coincidence: bool = True
    version: int = 0

    def describe(self, det_a, det_b):
        def one(name, c):
            sm = "auto %s" % (c.sum_auto,) if c.sum_auto else f"[{c.sum_lo}, {c.sum_hi}]"
            return (f"{name}: Sum {sm} ({c.sum_stage}), time [{c.t_lo}, {c.t_hi}] s")
        return (f"coincidence {'on' if self.use_coincidence else 'off'} | "
                + one(f"det {det_a} (A)", self.a) + " | " + one(f"det {det_b} (B)", self.b))


def _in_range(x, lo, hi):
    if lo is None and hi is None:
        return np.ones(len(x), bool)
    m = np.isfinite(x)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    return m


def _pad(arr, n, fill=0):
    if len(arr) == n:
        return arr
    out = np.full(n, fill, arr.dtype)
    out[:len(arr)] = arr
    return out


def _auto_limits(sums, weights, valid, q):
    """Limits from weighted quantiles of the true-pair Sum spectrum."""
    m = valid & np.isfinite(sums)
    if not m.any() or weights[m].sum() < 200:
        return None, None
    bins = np.linspace(0, sums[m].max() + 1, 401)
    h, _ = np.histogram(sums[m], bins, weights=weights[m])
    h = np.clip(np.convolve(h, np.ones(5) / 5, mode="same"), 0, None)
    cum = np.cumsum(h)
    cum = cum / cum[-1]
    lo = None if q[0] is None else float(np.interp(q[0], cum, bins[1:]))
    hi = None if q[1] is None else float(np.interp(q[1], cum, bins[1:]))
    return lo, hi


def ascii_hist(centres, counts, lines=20, width=50):
    """Tiny terminal histogram: bins are merged into at most `lines` rows."""
    centres, counts = np.asarray(centres), np.asarray(counts, float)
    g = int(np.ceil(len(counts) / lines))
    n = (len(counts) // g) * g
    c = counts[:n].reshape(-1, g).sum(1)
    x = centres[:n].reshape(-1, g).mean(1)
    top = max(c.max(), 1e-12)
    return "\n".join(f"{xi:12.4g} {ci:10.1f} " + "#" * max(int(width * ci / top), 0)
                     for xi, ci in zip(x, c))


# ------------------------------------------------------------------- result
@dataclass(frozen=True)
class SelectionResult:
    """Immutable snapshot of the full selection. Row i refers to loader row i."""
    version: int
    status: str
    settings: Settings
    det_a: int
    det_b: int
    n_a: int
    n_b: int
    valid_a: np.ndarray           # decision is final (searched / not pending)
    valid_b: np.ndarray
    mask_a: np.ndarray            # final selection (raw: still contains accidentals)
    mask_b: np.ndarray
    weight_a: np.ndarray          # final background-subtracted weights (0 if cut)
    weight_b: np.ndarray
    sum_limits: dict              # {'a': (lo, hi), 'b': (lo, hi)} actually used
    cutflow: tuple                # ((label, count_a, count_b), ...) in order applied
    window_us: tuple              # coincidence window used (None without coincidence)
    stats: dict                   # window stats with the cuts applied
    base_stats: dict              # same with no cuts at all
    sum_a: np.ndarray             # raw columns, for histograms
    sum_b: np.ndarray
    t_a: np.ndarray               # event time since the detector's first event [s]
    t_b: np.ndarray
    base_weight_a: np.ndarray     # background-subtracted weights, no cuts at all
    base_weight_b: np.ndarray
    run: object = None            # identifies the data these rows came from, so that
    epoch_a: int = None           # consumers can detect a file reset / new run
    epoch_b: int = None

    def histogram(self, det, quantity="sum", kind="all", bins=100, range=None):
        """(bin centres, counts) for one detector.
        quantity: 'sum' or 'time' (seconds since that detector's first event)
        kind:     'all'      every event
                  'true'     true pairs only: background-subtracted, independent
                             of the current cuts (use this to choose limits)
                  'selected' events kept by all current cuts, background-subtracted
                  'raw'      events kept by all current cuts, not subtracted"""
        side = "a" if det in (self.det_a, "a") else "b"
        x = getattr(self, ("sum_" if quantity == "sum" else "t_") + side)
        valid = getattr(self, "valid_" + side)
        if kind == "all":
            w = None
        elif kind == "true":
            w = getattr(self, "base_weight_" + side)
        elif kind == "selected":
            w = getattr(self, "weight_" + side)
        elif kind == "raw":
            w = getattr(self, "mask_" + side).astype(float)
        else:
            raise ValueError(kind)
        ok = valid & np.isfinite(x)
        if range is None:
            range = (float(x[ok].min()), float(x[ok].max())) if ok.any() else (0.0, 1.0)
            if range[0] == range[1]:
                range = (range[0], range[0] + 1.0)
        h, e = np.histogram(x[ok], bins=bins, range=range,
                            weights=None if w is None else w[ok])
        return 0.5 * (e[1:] + e[:-1]), h

    def show(self, det, quantity="sum", kind="all", bins=100, lines=20, range=None):
        c, h = self.histogram(det, quantity, kind, bins, range)
        return f"{quantity} histogram, det {det}, kind={kind}\n" + ascii_hist(c, h, lines)

    def summary(self):
        L = [f"[{self.status}] {self.settings.describe(self.det_a, self.det_b)}"]
        for k, name in (("a", self.det_a), ("b", self.det_b)):
            lo, hi = self.sum_limits[k]
            if getattr(self.settings, k).sum_auto:
                L.append(f"  det {name}: auto Sum limits -> [{lo}, {hi}]")
        L.append(f"  {'step':34s} {'det %d (A)' % self.det_a:>12s} {'det %d (B)' % self.det_b:>12s}")
        for label, ca, cb in self.cutflow:
            L.append(f"  {label:34s} {ca:12d} {cb:12d}")
        if self.stats:
            st, b = self.stats, self.base_stats
            if st.get("acc_fraction") is not None:
                L.append(f"  coincidence window [{self.window_us[0]:.2f}, {self.window_us[1]:.2f}] us: "
                         f"{st['pairs']} pairs, accidentals {100 * st['acc_fraction']:.1f}% "
                         f"-> true ~{st['true_pairs']:.0f}"
                         + (f"   (no cuts: {100 * b['acc_fraction']:.1f}%, true ~{b['true_pairs']:.0f})"
                            if b and b.get("acc_fraction") is not None else ""))
        return "\n".join(L)


# --------------------------------------------------------------- evaluation
def evaluate(settings, snap, det_a, det_b, coin):
    """Pure function: settings + loader snapshot + coincidence result -> SelectionResult
    (version 0; the filter object numbers them). `coin` may be None."""
    sa, sb = snap.detectors[det_a], snap.detectors[det_b]
    n_a, n_b = sa.n, sb.n
    sum_a, sum_b = sa.columns[SUM][:n_a], sb.columns[SUM][:n_b]
    t_a = sa.columns[EVT][:n_a] - sa.columns[EVT][0]       # first event = 0
    t_b = sb.columns[EVT][:n_b] - sb.columns[EVT][0]
    ca, cb = settings.a, settings.b
    use_coin = settings.use_coincidence and coin is not None
    zeros = lambda n: np.zeros(n)
    base_wa, base_wb = zeros(n_a), zeros(n_b)
    window, stats, base_stats = None, {}, {}
    status = "running"

    if use_coin and (coin.n_a > n_a or coin.n_b > n_b):     # coincidence from an older file
        use_coin, status = False, "coincidence result is stale"
    if settings.use_coincidence and coin is None:
        status = "waiting for coincidence"

    # -- valid rows --------------------------------------------------------
    if use_coin:
        valid_a, valid_b = _pad(coin.covered_a, n_a, False), _pad(coin.covered_b, n_b, False)
        # baseline: true-pair weights with no cuts (what the limits should be chosen from)
        bw = coin.window_us
        if bw is not None:
            wa, wb = coin.weights(bw)
            base_wa, base_wb = _pad(wa, n_a, 0.0), _pad(wb, n_b, 0.0)
            base_stats = coin.window_stats(bw)
    else:
        valid_a, valid_b = np.ones(n_a, bool), np.ones(n_b, bool)

    # -- Sum limits (manual, or auto from the true-pair spectrum) ------------
    lims = {}
    for k, c, s, bwt, v in (("a", ca, sum_a, base_wa, valid_a), ("b", cb, sum_b, base_wb, valid_b)):
        lo, hi = c.sum_lo, c.sum_hi
        if c.sum_auto and use_coin:
            alo, ahi = _auto_limits(s, bwt, v, c.sum_auto)
            lo, hi = alo, ahi
        lims[k] = (lo, hi)
    sum_ok_a, sum_ok_b = _in_range(sum_a, *lims["a"]), _in_range(sum_b, *lims["b"])
    time_ok_a, time_ok_b = _in_range(t_a, ca.t_lo, ca.t_hi), _in_range(t_b, cb.t_lo, cb.t_hi)

    flow = [("rows (decision final)", int(valid_a.sum()), int(valid_b.sum()))]

    if not use_coin:
        mask_a, mask_b = sum_ok_a & time_ok_a, sum_ok_b & time_ok_b
        flow.append(("pass Sum cut", int((sum_ok_a & valid_a).sum()), int((sum_ok_b & valid_b).sum())))
        flow.append(("pass time cut (last) = final", int((mask_a & valid_a).sum()),
                     int((mask_b & valid_b).sum())))
        wf_a, wf_b = mask_a.astype(float), mask_b.astype(float)
    else:
        pre_a, pre_b = ca.sum_stage == "pre", cb.sum_stage == "pre"
        elig_a = sum_ok_a if pre_a else np.ones(n_a, bool)
        elig_b = sum_ok_b if pre_b else np.ones(n_b, bool)
        keep = elig_a[coin.ia] & elig_b[coin.ib]
        rr = coin.restrict(keep)
        # window: manual stays; auto follows the surviving pairs (charge -> coincidence)
        window = rr.suggest_window() if coin.window_auto else coin.window_us
        if window is not None:
            stats = rr.window_stats(window)
            ma, mb = rr.masks(window)
            wa, wb = rr.weights(window)
        else:
            ma, mb = np.zeros(coin.n_a, bool), np.zeros(coin.n_b, bool)
            wa, wb = np.zeros(coin.n_a), np.zeros(coin.n_b)
        ma, mb, wa, wb = _pad(ma, n_a, False), _pad(mb, n_b, False), _pad(wa, n_a, 0.0), _pad(wb, n_b, 0.0)
        post_a = sum_ok_a if not pre_a else np.ones(n_a, bool)
        post_b = sum_ok_b if not pre_b else np.ones(n_b, bool)
        n = lambda ma_, mb_: (int((ma_ & valid_a).sum()), int((mb_ & valid_b).sum()))
        if pre_a or pre_b:
            flow.append(("pass Sum cut (before coincidence)",
                         *n(sum_ok_a if pre_a else np.ones(n_a, bool),
                            sum_ok_b if pre_b else np.ones(n_b, bool))))
        flow.append(("partner in window", *n(ma, mb)))
        if not (pre_a and pre_b):
            flow.append(("pass Sum cut (after coincidence)", *n(ma & post_a, mb & post_b)))
        fin_a, fin_b = post_a & time_ok_a, post_b & time_ok_b
        mask_a, mask_b = ma & fin_a, mb & fin_b
        wf_a, wf_b = wa * fin_a, wb * fin_b
        flow.append(("pass time cut (last) = final", *n(mask_a, mask_b)))
        flow.append(("background-subtracted weight sum", int(round(wf_a[valid_a].sum())),
                     int(round(wf_b[valid_b].sum()))))

    return SelectionResult(
        version=0, status=status, settings=settings, det_a=det_a, det_b=det_b,
        n_a=n_a, n_b=n_b, valid_a=valid_a, valid_b=valid_b,
        mask_a=mask_a & valid_a, mask_b=mask_b & valid_b,
        weight_a=wf_a * valid_a, weight_b=wf_b * valid_b, sum_limits=lims,
        cutflow=tuple(flow), window_us=window, stats=stats, base_stats=base_stats,
        sum_a=sum_a, sum_b=sum_b, t_a=t_a, t_b=t_b,
        base_weight_a=base_wa, base_weight_b=base_wb,
        run=snap.run, epoch_a=sa.epoch, epoch_b=sb.epoch)


# ------------------------------------------------------------------- object
class SelectionFilter(Poller):
    """Live Sum / time selection on top of the loader and (optionally) the
    coincidence filter. poll()/start() evaluate whenever data, the coincidence
    result or the settings changed; result() returns the latest SelectionResult.

    Detectors are addressed by their number (e.g. 3) or by 'a' / 'b'.
    """
    def __init__(self, source, coincidence=None, det_a=None, det_b=None,
                 use_coincidence=True):
        self.source, self.coincidence = source, coincidence
        self.det_a = coincidence.det_a if coincidence is not None else det_a
        self.det_b = coincidence.det_b if coincidence is not None else det_b
        if self.det_a is None or self.det_b is None:
            raise ValueError("give a coincidence filter or det_a and det_b")
        self._settings = Settings(use_coincidence=use_coincidence and coincidence is not None)
        self._lock = threading.Lock()            # serialises setters only
        self._key, self._result, self._version = None, None, 0

    # -- settings --------------------------------------------------------
    @property
    def settings(self):
        return self._settings

    def _side(self, det):
        if det in ("a", self.det_a):
            return "a"
        if det in ("b", self.det_b):
            return "b"
        raise KeyError(f"unknown detector {det!r}; use {self.det_a}, {self.det_b}, 'a' or 'b'")

    def _change(self, det=None, **kw):
        with self._lock:
            s = self._settings
            fields = {}
            if det is not None:
                k = self._side(det)
                fields[k] = dataclasses.replace(getattr(s, k), **kw)
            else:
                fields = kw
            self._settings = dataclasses.replace(s, version=s.version + 1, **fields)

    def set_sum(self, det, lo=None, hi=None, stage=None):
        """Sum limits for one detector (None = open). Switches auto limits off."""
        kw = dict(sum_lo=lo, sum_hi=hi, sum_auto=None)
        if stage:
            kw["sum_stage"] = self._stage(stage)
        self._change(det, **kw)

    def clear_sum(self, det):
        self._change(det, sum_lo=None, sum_hi=None, sum_auto=None)

    def set_time(self, det, lo=None, hi=None):
        """Event-time limits in seconds after that detector's first event."""
        self._change(det, t_lo=lo, t_hi=hi)

    def clear_time(self, det):
        self._change(det, t_lo=None, t_hi=None)

    def set_stage(self, stage, det=None):
        """'pre' (Sum cut before coincidence) or 'post' (after); both detectors
        unless det is given."""
        stage = self._stage(stage)
        for d in ([det] if det is not None else ["a", "b"]):
            self._change(d, sum_stage=stage)

    def auto_sum(self, det, quantiles=(0.02, 0.98)):
        """Keep Sum limits at these quantiles of the true-pair spectrum, updated
        live. Use None for an open side, e.g. (0.02, None). quantiles=None stops it."""
        self._change(det, sum_auto=None if quantiles is None else tuple(quantiles))

    def use_coincidence(self, on=True):
        if on and self.coincidence is None:
            raise ValueError("no coincidence filter attached")
        self._change(None, use_coincidence=bool(on))

    @staticmethod
    def _stage(stage):
        if stage not in ("pre", "post"):
            raise ValueError("stage must be 'pre' or 'post'")
        return stage

    # -- evaluation ------------------------------------------------------
    def result(self):
        return self._result

    def poll(self) -> int:
        """Re-evaluate if anything changed. Returns 1 if a new result was published."""
        snap = self.source.snapshot()
        sa, sb = snap.detectors.get(self.det_a), snap.detectors.get(self.det_b)
        if sa is None or sb is None or sa.n == 0 or sb.n == 0:
            return 0
        s = self._settings
        coin = self.coincidence.result() if (self.coincidence and s.use_coincidence) else None
        key = (snap.run, sa.epoch, sb.epoch, sa.n, sb.n, coin.version if coin else None,
               coin.window_us if coin else None, s.version)
        if key == self._key:
            return 0
        res = evaluate(s, snap, self.det_a, self.det_b, coin)
        self._key = key
        self._version += 1
        self._result = dataclasses.replace(res, version=self._version)
        return 1