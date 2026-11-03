"""Live coincidence filter between two detectors.

Convention: dt = t_B - t_A, in microseconds. A is the detector whose events you
want to filter (e.g. the VMI), B the partner (e.g. RAEA). For your old function
that is det_a=3 (VMI), det_b=2 (RAEA).

How it works
------------
1. Pulses.  Each detector's 'Sync Time' value changes once per cycle (~1 s).
   The first row of each constant-Sync-Time block gives that cycle's pulse time
   on the detector's own clock.
2. Pulse matching.  A pulses are matched to B pulses by *predicted time*
   (A - B clock offset, tracked pulse by pulse), not by cycle index. Dropped or
   spurious pulses on either side are skipped, so unequal cycle counts are fine.
   The integer ambiguity at start-up (which B pulse is the same physical pulse
   as A pulse 0?) is resolved from the data: the right shift gives a sharp peak
   in dt.
3. Clock map.  B event times are mapped onto A's clock by piecewise-linear
   interpolation between matched pulse pairs. This removes the relative clock
   drift (~6 us/s here) and the pulse-to-pulse jitter inside the map.
4. Pairs.  ALL pairs with |dt| <= search window are stored (vectorised,
   searchsorted-based). Windows are applied afterwards, so changing the window
   is instant and needs no reprocessing.
5. Accidentals.  The detectors fire on a laser-shot grid (~1 kHz here), so
   unrelated A and B events in the SAME shot fall inside the coincidence peak.
   A shift by one cycle does NOT reproduce that (one cycle is not a whole number
   of shots). The right control is a shift by whole laser shots: the same window
   moved by +-1, +-2 shots counts pairs of unrelated events from neighbouring
   shots, with the same sub-shot timing structure. The search window is wide
   enough (+-2.5 shots) that these sidebands are already in the pair list.
   The shot period is measured from the sideband positions.
6. Live.  Time is processed in slabs up to a frontier that both detectors have
   fully passed, so each event is searched exactly once and results never change
   afterwards. The worker publishes one immutable CoincidenceResult per update
   (no locks needed).
"""
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from daq.data_loader import Poller

EVT, SYN = "Event time", "Sync Time"


# ------------------------------------------------------------------ helpers
def block_starts(sync, jitter_frac=0.25):
    """Row index where each cycle (block of one Sync Time) starts.

    The Sync Time is constant within a cycle, except that electronics noise bursts
    can make it creep by a few us from row to row. Such creep must not start a new
    cycle: a step only counts when it exceeds `jitter_frac` of the typical cycle
    step (the median of the steps above 1 ms)."""
    if len(sync) == 0:
        return np.empty(0, np.int64)
    d = np.diff(sync)
    big = d[d > 1e-3]
    thr = jitter_frac * np.median(big) if len(big) else 0.0
    return np.r_[0, np.flatnonzero(d > thr) + 1]


def find_pairs(t_ref, t_oth, w):
    """All pairs with |t_oth - t_ref| <= w.  Returns (i, j, dt), j indexing t_oth.
    Fully vectorised; t_oth may be unsorted and may contain NaN (ignored)."""
    idx = np.flatnonzero(np.isfinite(t_oth))
    if len(t_ref) == 0 or len(idx) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    to = t_oth[idx]
    order = np.argsort(to, kind="stable")
    ts = to[order]
    lo = np.searchsorted(ts, t_ref - w, side="left")
    hi = np.searchsorted(ts, t_ref + w, side="right")
    n = hi - lo
    tot = int(n.sum())
    if tot == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    i = np.repeat(np.arange(len(t_ref)), n)
    k = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
    j = idx[order[np.repeat(lo, n) + k]]
    return i, j, t_oth[j] - t_ref[i]


class _Store:
    """Append-only growable columns, published as one (columns, n) tuple."""
    def __init__(self, dtypes, cap=1 << 15):
        self._cols = [np.empty(cap, d) for d in dtypes]
        self._n = 0
        self._state = (tuple(self._cols), 0)

    def append(self, *arrs):
        m = len(arrs[0])
        if m == 0:
            return
        n = self._n
        if n + m > len(self._cols[0]):
            cap = max(2 * len(self._cols[0]), n + m)
            new = [np.empty(cap, c.dtype) for c in self._cols]
            for c, nc in zip(self._cols, new):
                nc[:n] = c[:n]
            self._cols = new
        for c, a in zip(self._cols, arrs):
            c[n:n + m] = a                      # write beyond published n
        self._n = n + m
        self._state = (tuple(self._cols), self._n)   # publish last

    def views(self):
        cols, n = self._state
        return [c[:n] for c in cols]


def _peak_fwhm(dt, centre, half, nbins, bkg_density):
    counts, edges = np.histogram(dt, bins=nbins, range=(centre - half, centre + half))
    bw = edges[1] - edges[0]
    sm = np.convolve(counts, np.ones(3) / 3, mode="same")
    p = int(np.argmax(sm))
    b = bkg_density * bw
    if sm[p] < b + 5 * np.sqrt(b + 1):             # no significant peak
        return None
    level = b + 0.5 * (sm[p] - b)
    lo = p
    while lo > 0 and sm[lo - 1] >= level:
        lo -= 1
    hi = p
    while hi < nbins - 1 and sm[hi + 1] >= level:
        hi += 1
    left, right = edges[lo], edges[hi + 1]
    return 0.5 * (left + right), right - left, (lo == 0 or hi == nbins - 1)


# ------------------------------------------------------------------- result
@dataclass(frozen=True)
class CoincidenceResult:
    """Immutable snapshot of the filter. Row indices refer to loader rows."""
    version: int
    status: str
    n_a: int
    n_b: int
    covered_a: np.ndarray        # bool: row has been searched for partners
    covered_b: np.ndarray        # (False = still pending, or before first matched pulse)
    ia: np.ndarray               # pair list: row in A, row in B,
    ib: np.ndarray
    dt_us: np.ndarray            #   dt = t_B - t_A [us],
    ta: np.ndarray               #   and A-clock time [s] of the pair
    frontier: float              # A-clock time up to which pairs are final
    n_matched: int               # matched sync pulses
    n_unmatched_a: int
    search_window_us: float
    shot_period_guess_us: float = 1000.0
    bkg_shots: tuple = (-2, -1, 1, 2)
    window_us: tuple = None      # (tmin, tmax) in us in effect
    window_auto: bool = True
    pulses_a: np.ndarray = None  # matched sync pulses on each detector's own clock [s]
    pulses_b: np.ndarray = None  # (offset = pulses_a - pulses_b shows the clock drift)
    n_sigma: float = 3.0         # half-width of the automatic window, in peak sigmas
    dt_offset_us: float = 0.0    # dt = t_B - t_A - offset: the search is centred on this raw dt [us]
    peak_range_us: float = None  # the automatic window looks for the peak within +- this [us]; None = half a shot
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def restrict(self, keep):
        """New result holding only the pairs where `keep` is True, e.g. pairs whose
        two events both pass a charge cut. Pairs depend only on their two events,
        so this equals running the search on pre-filtered events. Peak, window,
        sidebands and weights are then all recomputed from the restricted pairs."""
        return dataclasses.replace(self, ia=self.ia[keep], ib=self.ib[keep],
                                   dt_us=self.dt_us[keep], ta=self.ta[keep])

    # -- peak / window / shot period ---------------------------------------
    def peak(self):
        if "peak" not in self._cache:
            self._cache["peak"] = self._compute_peak()
        return self._cache["peak"]

    def shot_period(self):
        """Period [us] of the periodic structure in the event times (laser shots,
        ion bunches, chopper ...), from the positions of the +-1 sidebands (their
        spacing is 2 periods). None if no sidebands are visible."""
        if "period" not in self._cache:
            self._cache["period"] = self._compute_shot_period()
        return self._cache["period"]

    def _compute_peak(self):
        """(centre_us, sigma_us) of the main coincidence peak, or None.
        sigma is FWHM/2.355; the peak is looked for within +-search window."""
        dt, W = self.dt_us, self.search_window_us
        if len(dt) < 50:
            return None
        nb = 200
        # The main peak is the tallest feature within +-half a shot of zero (or +-peak_range_us).
        P0 = self.shot_period_guess_us
        R = min(W, 0.5 * P0 if not self.peak_range_us else self.peak_range_us)
        counts, edges = np.histogram(dt, bins=nb, range=(-R, R))
        bw = 2 * R / nb
        bkg = np.median(counts) / bw
        p = int(np.argmax(counts))
        c, half = 0.5 * (edges[p] + edges[p + 1]), 3 * bw
        res = None
        for _ in range(6):
            r = _peak_fwhm(dt, c, half, 100, bkg)
            if r is None:
                # no peak shape in the range (a step or a shoulder, or too little data): with an
                # explicit peak range sit the window on dt = 0 and let the user drag it
                return (0.0, R / 6) if self.peak_range_us else None
            c, fwhm, touches = r
            res = (c, fwhm / 2.3548)
            if touches:
                half *= 3
            elif half > 4 * fwhm:
                half = 2.5 * fwhm                   # refine resolution
            else:
                break
        return res

    def suggest_window(self, n_sigma=None):
        n_sigma = self.n_sigma if n_sigma is None else n_sigma
        pk = self.peak()
        return None if pk is None else (pk[0] - n_sigma * pk[1], pk[0] + n_sigma * pk[1])

    def _compute_shot_period(self):
        pk = self.peak()
        if pk is None:
            return None
        c, P0, dt = pk[0], self.shot_period_guess_us, self.dt_us
        cents = []
        for sgn in (+1, -1):
            lo, hi = sorted((c + sgn * 0.97 * P0, c + sgn * 1.03 * P0))
            sel = dt[(dt >= lo) & (dt <= hi)]
            if len(sel) < 50:
                return None
            counts, edges = np.histogram(sel, bins=int((hi - lo) / 0.5), range=(lo, hi))
            sm = np.convolve(counts, np.ones(3) / 3, mode="same")
            p = int(np.argmax(sm))
            if sm[p] < 4 * np.median(counts) + 10:
                return None
            ctr = 0.5 * (edges[p] + edges[p + 1])
            near = sel[np.abs(sel - ctr) < 8.0]
            cents.append(near.mean())
        return (cents[0] - cents[1]) / 2.0

    # -- background: windows moved by whole shots (or, failing that, flat) ----
    def _bkg_offsets(self, w):
        """(method, offsets_us, labels): where to count the control windows.
        'shots': the window moved by k periods of the periodic structure.
        'flat' : no periodic structure found; windows far from the peak, which
                 assumes a flat accidental distribution."""
        W = self.search_window_us
        inside = lambda o: w[0] + o >= -W and w[1] + o <= W
        P = self.shot_period()
        if P is not None:
            ks = [k for k in self.bkg_shots if inside(k * P)]
            return "shots", [k * P for k in ks], ks
        offs = [o for o in (-0.8 * W, -0.5 * W, 0.5 * W, 0.8 * W) if inside(o)]
        return "flat", offs, [round(o) for o in offs]

    def window_stats(self, window_us=None):
        """Pairs in the window and the control-window estimate of accidentals.
        sideband_counts holds the count in each control window; they should
        agree with each other (check the scatter!). acc is their mean."""
        w = window_us or self.window_us
        if w is None:
            return None
        dt = self.dt_us
        main = int(((dt >= w[0]) & (dt <= w[1])).sum())
        method, offs, labels = self._bkg_offsets(w)
        counts = {l: int(((dt >= w[0] + o) & (dt <= w[1] + o)).sum())
                  for o, l in zip(offs, labels)}
        out = dict(pairs=main, method=method, shot_period_us=self.shot_period(),
                   sideband_counts=counts, acc=None, acc_fraction=None, true_pairs=None)
        if counts:
            acc = float(np.mean(list(counts.values())))
            out.update(acc=acc, acc_fraction=acc / main if main else float("nan"),
                       true_pairs=main - acc)
        return out

    def weights(self, window_us=None, background=True):
        """Per-row float weights (w_a, w_b) for building filtered images.
        Each pair inside the window adds +1 to both of its rows; with
        background=True each pair in a control window subtracts 1/K (K = number
        of control windows used), so sum(w_a) = true-pair estimate.
        Use only rows with covered_a / covered_b True."""
        w = window_us or self.window_us or self.suggest_window()
        wa, wb = np.zeros(self.n_a), np.zeros(self.n_b)
        if w is None:
            return wa, wb
        dt = self.dt_us
        sel = (dt >= w[0]) & (dt <= w[1])
        wa += np.bincount(self.ia[sel], minlength=self.n_a)
        wb += np.bincount(self.ib[sel], minlength=self.n_b)
        if background:
            _, offs, _ = self._bkg_offsets(w)
            for o in offs:
                s = (dt >= w[0] + o) & (dt <= w[1] + o)
                wa -= np.bincount(self.ia[s], minlength=self.n_a) / len(offs)
                wb -= np.bincount(self.ib[s], minlength=self.n_b) / len(offs)
        return wa, wb

    def masks(self, window_us=None, unique=False):
        """(mask_a, mask_b): rows having a partner with dt inside the window.
        This is the RAW selection and still contains accidentals; use
        weights() for background-subtracted images. Pending rows are False
        (see covered_a / covered_b). unique=True keeps only one-to-one pairs."""
        w = window_us or self.window_us or self.suggest_window()
        ma, mb = np.zeros(self.n_a, bool), np.zeros(self.n_b, bool)
        if w is None:
            return ma, mb
        sel = (self.dt_us >= w[0]) & (self.dt_us <= w[1])
        ia, ib = self.ia[sel], self.ib[sel]
        if unique:
            ca = np.bincount(ia, minlength=self.n_a)
            cb = np.bincount(ib, minlength=self.n_b)
            keep = (ca[ia] == 1) & (cb[ib] == 1)
            ia, ib = ia[keep], ib[keep]
        ma[ia] = True
        mb[ib] = True
        return ma, mb

    def histogram(self, bin_us=1.0, range_us=None):
        """(bin centres, pair counts) of dt over the search window."""
        W = self.search_window_us
        lo, hi = range_us or (-W, W)
        nb = max(int(round((hi - lo) / bin_us)), 1)
        h, e = np.histogram(self.dt_us, bins=nb, range=(lo, hi))
        return 0.5 * (e[1:] + e[:-1]), h

    def summary(self):
        L = [f"[{self.status}] rows A={self.n_a} B={self.n_b} | matched pulses="
             f"{self.n_matched} (unmatched A pulses: {self.n_unmatched_a})",
             f"pairs within +-{self.search_window_us:.0f} us: {len(self.dt_us)}"]
        pk = self.peak()
        if pk:
            L.append(f"main peak at {pk[0]:.2f} us, sigma {pk[1]:.2f} us")
        w = self.window_us
        if w:
            st = self.window_stats(w)
            L.append(f"window [{w[0]:.2f}, {w[1]:.2f}] us "
                     f"({'auto' if self.window_auto else 'manual'}): {st['pairs']} pairs")
            if st["acc"] is not None:
                sb = ", ".join(f"{k:+d}:{n}" for k, n in st["sideband_counts"].items())
                if st["method"] == "shots":
                    L.append(f"period {st['shot_period_us']:.3f} us; control windows moved by "
                             f"k periods [{sb}]")
                else:
                    L.append(f"no periodic structure found: flat-background fallback, "
                             f"control windows at offsets [{sb}] us")
                L.append(f"accidentals ~{st['acc']:.0f} ({100 * st['acc_fraction']:.1f}% of "
                         f"window) -> true pairs ~{st['true_pairs']:.0f}")
            else:
                L.append("no control windows fit in the search window: no accidental estimate")
            ma, mb = self.masks(w)
            fa = (ma & self.covered_a).sum() / max(self.covered_a.sum(), 1)
            fb = (mb & self.covered_b).sum() / max(self.covered_b.sum(), 1)
            L.append(f"raw A events with a partner in window: {100 * fa:.1f}% | B: "
                     f"{100 * fb:.1f}% (of {self.covered_a.sum()} / {self.covered_b.sum()} "
                     f"searched)")
        return "\n".join(L)


# ------------------------------------------------------------------- engine
class CoincidenceFilter(Poller):
    """Incremental coincidence search between detectors det_a and det_b.

    source            anything with .snapshot() -> SetSnapshot (e.g. DetectorSet)
    search_window_us  store all pairs with |dt| below this; must exceed ~2 laser
                      shots so that the accidental sidebands are included
    shot_period_us    nominal laser shot period (refined from the data)
    window_us         (tmin, tmax) in us; None = auto from the peak
    cycle_shift       force B pulse index = A pulse index + shift (None = auto)
    match_tol         max clock-offset change between pulses to still match [s]
    lag_margin        events may precede their Sync Time by this much [s]
    dt_offset_us      centre of the search: dt = t_B - t_A - dt_offset_us. Moves another peak of the
                      dt distribution (e.g. a neighbouring laser shot) to dt = 0
    """
    def __init__(self, source, det_a, det_b, search_window_us=2500.0,
                 shot_period_us=1000.0, window_us=None, bkg_shots=(-2, -1, 1, 2),
                 cycle_shift=None, max_shift=10, init_blocks=8, match_tol=1e-3,
                 lag_margin=0.1, dt_offset_us=0.0):
        self.source, self.det_a, self.det_b = source, det_a, det_b
        self.W = search_window_us * 1e-6
        self.shot_period_us, self.bkg_shots = shot_period_us, tuple(bkg_shots)
        self._window = window_us
        self.cycle_shift = cycle_shift
        self.max_shift, self.init_blocks = max_shift, init_blocks
        self.match_tol, self.lag_margin = match_tol, lag_margin
        self.offset = float(dt_offset_us) * 1e-6
        self.auto_n_sigma = 3.0
        self.peak_range_us = None
        self.cycle_shift_found = None
        self._result, self._version, self._rc = None, 0, (None, None)
        self._reset(None)

    # ------------------------------------------------------------- public
    def set_window(self, tmin_us, tmax_us):
        self._window = (float(tmin_us), float(tmax_us))

    def clear_window(self):
        self._window = None

    def set_bkg_shots(self, shots):
        """Which whole-period shifts of the window are used as control windows,
        e.g. (-1, 1) or (-2, -1, 1, 2). Takes effect immediately."""
        self.bkg_shots = tuple(int(k) for k in shots)

    def set_peak_range(self, range_us):
        """The automatic window takes the tallest feature within +- range_us of dt = 0 (None or 0:
        half a shot). Use a small range after a dt offset, so that a weaker peak at 0 wins over a
        taller one further away."""
        self.peak_range_us = float(range_us) if range_us else None

    def set_auto_sigma(self, n_sigma):
        """Half-width of the automatic window in units of the peak sigma."""
        self.auto_n_sigma = float(n_sigma)

    def set_shot_period(self, period_us):
        """Nominal period [us] of the periodic structure (refined from the data)."""
        self.shot_period_us = float(period_us)

    def reconfigure(self, search_window_us=None, cycle_shift="keep", dt_offset_us=None):
        """Change settings that need a full re-search (search window, forced cycle
        shift; None = auto, dt offset; None = keep). Stop the worker thread first, then start it again."""
        if self.running:
            raise RuntimeError("stop() the filter before reconfigure()")
        if search_window_us is not None:
            self.W = float(search_window_us) * 1e-6
        if cycle_shift != "keep":
            self.cycle_shift = cycle_shift
        if dt_offset_us is not None:
            new = float(dt_offset_us) * 1e-6
            if abs(new - self.offset) > 1e-12:
                self._window = None             # a manual window sat on the old peak: pick the window again
            self.offset = new
        self._reset(None)
        self._rc = (None, None)

    def result(self) -> CoincidenceResult:
        r = self._result
        if r is None:
            return None
        key = (r.version, self._window, self.bkg_shots, self.shot_period_us, self.auto_n_sigma, self.peak_range_us)
        if self._rc[0] != key:               # one object per key so cached
            r2 = dataclasses.replace(r, bkg_shots=self.bkg_shots,     # peak/period are reused
                                     shot_period_guess_us=self.shot_period_us,
                                     n_sigma=self.auto_n_sigma, peak_range_us=self.peak_range_us)
            w = self._window if self._window is not None else r2.suggest_window()
            r3 = dataclasses.replace(r2, window_us=w, window_auto=self._window is None)
            r3._cache.update(r2._cache)
            self._rc = (key, r3)
        return self._rc[1]

    def poll(self) -> int:
        """Process whatever the source has. Returns number of newly searched A rows."""
        return self.update(self.source.snapshot())

    def update(self, snap) -> int:
        sa, sb = snap.detectors.get(self.det_a), snap.detectors.get(self.det_b)
        if sa is None or sb is None or sa.n == 0 or sb.n == 0:
            return 0
        sig = (snap.run, sa.epoch, sb.epoch)
        if sig != self._sig:
            self._reset(sig)
        if (sa.n, sb.n) == self._last_n:
            return 0
        self._last_n = (sa.n, sb.n)
        nA, nB = sa.n, sb.n
        evA, synA = sa.columns[EVT], sa.columns[SYN]
        evB, synB = sb.columns[EVT], sb.columns[SYN]
        stA, stB = block_starts(synA), block_starts(synB)
        TA, TB = synA[stA], synB[stB]

        if self._O is None and not self._init_offset(TA, TB, stA, stB, evA, evB):
            self._publish(nA, nB, "aligning cycles (need more data / clear peak)")
            return 0
        self._match(TA, TB)
        n_new = self._process(evA, evB, stA, stB, TA, TB, nA, nB)
        self._publish(nA, nB, ("running" if len(self._mTA) >= 2 else "waiting for pulses")
                      + (" (cycle shift not determinable: assumed 0)" if self.cycle_shift_ambiguous
                         and self.cycle_shift_found == 0 else
                         " (cycle shift not determinable: assumed %+d)" % self.cycle_shift_found
                         if self.cycle_shift_ambiguous else ""))
        return n_new

    # ----------------------------------------------------------- internals
    def _reset(self, sig):
        self._sig, self._last_n = sig, None
        self._O = None                      # predicted clock offset T_A - T_B [s]
        self._next_i, self._last_j = 0, -1
        self._mTA, self._mTB = [], []       # matched pulse pairs
        self.n_unmatched_a = 0
        self._F = -np.inf                   # finalised-time frontier (A clock)
        self._pairs = _Store([np.int64, np.int64, np.float64, np.float64])
        self._cov_a = np.zeros(1 << 16, bool)
        self._cov_b = np.zeros(1 << 16, bool)
        self.cycle_shift_found = None
        self.cycle_shift_ambiguous = False

    def _init_offset(self, TA, TB, stA, stB, evA, evB):
        """Find which B pulse corresponds to which A pulse (integer shift)."""
        ca, cb = len(TA) - 1, len(TB) - 1                 # complete blocks
        forced = self.cycle_shift is not None
        if min(ca, cb) < (2 if forced else self.init_blocks):
            return False
        shifts = [self.cycle_shift] if forced else range(-self.max_shift, self.max_shift + 1)
        W_us, scores = self.W * 1e6, {}
        for s in shifts:
            i0 = max(0, -s)
            i1 = min(ca, cb - s, i0 + 60)
            if i1 - i0 < 2:
                scores[s] = 0
                continue
            ia = np.arange(i0, i1)
            ja = ia + s
            a_rows = evA[stA[i0]:stA[i1]]
            b_rows = evB[stB[ja[0]]:stB[ja[-1] + 1]]
            tau = np.interp(b_rows, TB[ja], TA[ia], left=np.nan, right=np.nan)
            dt = find_pairs(a_rows, tau, self.W)[2] * 1e6
            # sharpest 5 us feature in dt: robust against the broad sidebands
            scores[s] = 0 if len(dt) == 0 else int(np.histogram(
                dt, bins=max(int(2 * W_us / 5), 2), range=(-W_us, W_us))[0].max())
        if forced:
            best = self.cycle_shift
        else:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best, top = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0
            if top < 20:
                return False
            if top < 3 * max(second, 5):
                # The laser structure is the same in every cycle, so every shift shows
                # it and the shifts cannot be told apart. Both detectors start together,
                # so take the shift closest to 0 and say so in the status.
                spread = np.array(list(scores.values()), float)
                if spread.min() < 0.5 * top:
                    return False                     # a few shifts stand out: not clear yet
                best = min((s for s in scores if scores[s] >= 0.5 * top), key=abs)
                self.cycle_shift_ambiguous = True
        i0 = max(0, -best)
        self._O = TA[i0] - TB[i0 + best]
        self._next_i, self._last_j = i0, i0 + best - 1
        self.cycle_shift_found = best
        return True

    def _match(self, TA, TB):
        """Match A pulses to B pulses by predicted time (see module docstring)."""
        nbB, tol = len(TB), self.match_tol
        while self._next_i < len(TA):
            ta = TA[self._next_i]
            pred = ta - self._O
            j = int(np.searchsorted(TB, pred))
            best, bd = None, np.inf
            for c in (j - 1, j):
                if 0 <= c < nbB and c > self._last_j:
                    d = abs(TB[c] - pred)
                    if d < bd:
                        best, bd = c, d
            if best is not None and bd < tol:
                self._mTA.append(ta)
                self._mTB.append(TB[best])
                self._O = ta - TB[best]
                self._last_j = best
            elif pred > TB[-1] + tol:
                break                        # B has not reached this pulse yet
            else:
                self.n_unmatched_a += 1      # dropped / spurious pulse: skip
            self._next_i += 1

    @staticmethod
    def _scan_start(T, st, t_lo, margin=1.5):
        b0 = max(int(np.searchsorted(T, t_lo - margin, side="right")) - 1, 0)
        return int(st[b0])

    def _process(self, evA, evB, stA, stB, TA, TB, nA, nB):
        if len(self._mTA) < 2:
            return 0
        mTA, mTB = np.asarray(self._mTA), np.asarray(self._mTB)
        W = self.W
        # Both detectors are complete (in A-clock) below mTA[-1] - lag_margin.
        F = mTA[-1] - self.lag_margin - W
        if F <= self._F:
            return 0
        lo, hi = self._F, F
        # A events finalised in this slab: A-clock time in [lo, hi)
        r0 = self._scan_start(TA, stA, lo)
        ev = evA[r0:nA]
        m = (ev >= lo) & (ev < hi)
        iA, tA = r0 + np.flatnonzero(m), ev[m]
        # B events (mapped to the A clock) that can pair with them
        r0 = self._scan_start(TB, stB, lo - W - self._O + self.offset)
        ev = evB[r0:nB]
        tau = np.interp(ev, mTB, mTA, left=np.nan, right=np.nan) - self.offset   # dt is relative to the offset
        m = (tau >= lo - W) & (tau < hi + W)          # NaN compares False
        iB, tB = r0 + np.flatnonzero(m), tau[m]

        i, j, dt = find_pairs(tA, tB, W)
        self._pairs.append(iA[i], iB[j], dt * 1e6, tA[i])
        self._grow_cov(nA, nB)
        self._cov_a[iA[tA >= mTA[0] + W]] = True
        inB = (tB >= lo) & (tB < hi) & (tB >= mTA[0] + W)
        self._cov_b[iB[inB]] = True
        self._F = F
        return len(iA)

    def _grow_cov(self, nA, nB):
        for name, n in (("_cov_a", nA), ("_cov_b", nB)):
            arr = getattr(self, name)
            if len(arr) < n:
                new = np.zeros(max(2 * len(arr), n), bool)
                new[:len(arr)] = arr
                setattr(self, name, new)

    def _publish(self, nA, nB, status):
        self._grow_cov(nA, nB)
        self._version += 1
        ia, ib, dt, ta = self._pairs.views()
        self._result = CoincidenceResult(
            version=self._version, status=status, n_a=nA, n_b=nB,
            covered_a=self._cov_a[:nA].copy(), covered_b=self._cov_b[:nB].copy(),
            ia=ia, ib=ib, dt_us=dt, ta=ta, frontier=self._F,
            n_matched=len(self._mTA), n_unmatched_a=self.n_unmatched_a,
            search_window_us=self.W * 1e6, shot_period_guess_us=self.shot_period_us,
            bkg_shots=self.bkg_shots, pulses_a=np.asarray(self._mTA),
            pulses_b=np.asarray(self._mTB), dt_offset_us=self.offset * 1e6)