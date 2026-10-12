"""Bias-corrected bootstrap of the Abel inversion (no Qt).

What is resampled
-----------------
Every event row has a weight w_i (1 for counts, or the background-subtraction weight: positive for
events in the coincidence window, negative for events in the control windows). A bootstrap sample
multiplies each weight by an independent k ~ Poisson(1) (the Poisson bootstrap: the bootstrapped
total count itself fluctuates like a count, and it can be drawn without ever sorting or copying the
data). The rows in the control windows are resampled like all others, so the noise of the
accidental subtraction is part of the error.

  unit = "event":  one k per event row            (events are independent Poisson counts)
  unit = "cycle":  one k per Sync-Time cycle (about 1 s), shared by all rows of the cycle: keeps
                   correlations inside a cycle (intensity drifts, shot-to-shot structure), at the
                   price of only ~one unit per second of data.

The statistics
--------------
Everything after the per-event radial binning is linear in the weights (abel.RadialBinner), so a
whole batch of resamples is a few matrix products (AbelSide.resample). The statistics are

  I(rho)      the intensity,                       shape (R,)
  beta_n(rho) for every order n >= 1 in use,       shape (N-1, R)

and for beta the same bins are left out (NaN) as in the measurement, so a bin never changes its
meaning between samples. Bins that are NaN in the measurement have no interval.

Bias-corrected percentile interval (Efron 1987, without the acceleration)
------------------------------------------------------------------------
For each statistic separately, with theta the measured value and theta*_b the B bootstrap values,

  z0      = Phi^-1( #{theta*_b < theta} / B )          (ties count one half)
  lower   = quantile of theta*   at  Phi( 2 z0 - z )
  upper   = quantile of theta*   at  Phi( 2 z0 + z )

with z = 1 for the usual +-1 sigma (a 68.27 % interval; z = n_sigma in general). Quantiles use linear
interpolation between the sorted samples. Without bias (z0 = 0) this is the plain percentile interval.
The legacy C++ program did the same, with a bubble sort per bin and special cases when z0 was
infinite; here the fraction is kept inside [1/2B, 1 - 1/2B], which makes z0 finite and the
interval a clamped one-sided interval in that case, as it should.
`accel` (the acceleration constant of BCa, 0 here) is the hook for the next refinement: because
everything is linear in the weights, the influence function it needs is available exactly.
"""
import threading
import time
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()
_cdf = np.vectorize(_N.cdf, otypes=[float])
_ppf = np.vectorize(_N.inv_cdf, otypes=[float])


# ------------------------------------------------------------------ intervals
@dataclass(frozen=True)
class Interval:
    estimate: np.ndarray        # the measured statistic (shape S)
    lower: np.ndarray           # bias-corrected lower bound (S)
    upper: np.ndarray           # bias-corrected upper bound (S)
    z0: np.ndarray              # bias constant (S)
    std: np.ndarray             # standard deviation of the bootstrap values (S)
    bias: np.ndarray            # mean(bootstrap) - estimate (S)
    n_valid: np.ndarray         # number of finite bootstrap values (S)
    n_sigma: float

    @property
    def minus(self):
        return self.estimate - self.lower

    @property
    def plus(self):
        return self.upper - self.estimate


def bias_corrected_interval(estimate, samples, n_sigma=1.0, accel=0.0):
    """Bias-corrected bootstrap interval for every statistic.

    estimate: shape S. samples: shape (B, *S), NaN allowed (ignored per statistic). Returns Interval."""
    est = np.asarray(estimate, float)
    smp = np.asarray(samples, float)
    if smp.shape[1:] != est.shape:
        raise ValueError(f"samples {smp.shape} do not match the estimate {est.shape}")
    B = smp.shape[0]
    flat = smp.reshape(B, -1)
    e = est.reshape(-1)
    fin = np.isfinite(flat)
    n = fin.sum(axis=0)                                                    # valid samples per statistic
    srt = np.sort(flat, axis=0)                                            # NaN last
    with np.errstate(invalid="ignore"):
        below = ((flat < e) & fin).sum(axis=0) + 0.5 * ((flat == e) & fin).sum(axis=0)
        ok = (n >= 2) & np.isfinite(e)
        p = np.where(ok, below / np.maximum(n, 1), 0.5)
        p = np.clip(p, 0.5 / np.maximum(n, 1), 1.0 - 0.5 / np.maximum(n, 1))
        z0 = _ppf(np.where(ok, p, 0.5))
        z = float(n_sigma)
        ah = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
        al = z0 + (z0 - z) / (1.0 - accel * (z0 - z))
        lo_q, hi_q = _cdf(al), _cdf(ah)

        def quantile(q):
            pos = q * np.maximum(n - 1, 0)
            i0 = np.clip(np.floor(pos).astype(np.int64), 0, np.maximum(n - 2, 0))
            fr = pos - i0
            a = np.take_along_axis(srt, i0[None, :], 0)[0]
            b = np.take_along_axis(srt, np.minimum(i0 + 1, np.maximum(n - 1, 0))[None, :], 0)[0]
            return a + (b - a) * fr
        lower = np.where(ok, quantile(lo_q), np.nan)
        upper = np.where(ok, quantile(hi_q), np.nan)
        mean = np.where(fin, flat, 0.0).sum(axis=0) / np.maximum(n, 1)
        dev = np.where(fin, flat - mean, 0.0)
        std = np.sqrt((dev ** 2).sum(axis=0) / np.maximum(n - 1, 1))
    shape = est.shape
    return Interval(estimate=est, lower=lower.reshape(shape), upper=upper.reshape(shape),
                    z0=np.where(ok, z0, np.nan).reshape(shape), std=np.where(ok, std, np.nan).reshape(shape),
                    bias=np.where(ok, mean - e, np.nan).reshape(shape), n_valid=n.reshape(shape),
                    n_sigma=float(n_sigma))


# ------------------------------------------------------------------ the runner
@dataclass(frozen=True)
class BootstrapResult:
    side: str
    signature: tuple             # what the measurement was made with: see AbelSide.signature
    unit: str
    n_sigma: float
    n_done: int
    n_total: int
    n_events: int                # events with non-zero weight that were resampled
    n_units: int                 # resampling units (events, or cycles)
    seconds: float
    running: bool
    stopped: bool                # ended by stop() before n_total
    error: str                   # "" unless the thread failed
    rho: np.ndarray
    beta_orders: tuple
    intensity: Interval          # None until two samples are in
    beta: Interval               # None while no beta exists (only order 0 in use)
    intensity_samples: np.ndarray    # (n_done, R): the bootstrap distribution itself
    beta_samples: np.ndarray         # (n_done, N-1, R)

    @property
    def done(self):
        """Finished all samples (not running, not stopped early, no error)."""
        return not self.running and not self.stopped and not self.error and self.n_done == self.n_total

    def summary(self, at=None):
        """One or two lines of text. at: radial bin index to quote the intervals at."""
        state = ("running" if self.running else "stopped" if self.stopped else
                 "failed: " + self.error if self.error else "done")
        s = (f"bootstrap {state}: {self.n_done}/{self.n_total} samples of {self.n_units} "
             f"{self.unit}s ({self.n_events} events), {self.seconds:.1f} s, interval +-{self.n_sigma:g} sigma "
             f"({100 * (2 * _N.cdf(self.n_sigma) - 1):.1f} %)")
        if at is not None and self.intensity is not None and np.isfinite(self.intensity.lower[at]):
            i = self.intensity
            s += (f"\n  I at bin {at}: {i.estimate[at]:.4g} -{i.minus[at]:.3g} +{i.plus[at]:.3g}"
                  f"  (bias z0 = {i.z0[at]:+.2f})")
            if self.beta is not None:
                for j, n in enumerate(self.beta_orders):
                    b = self.beta
                    if np.isfinite(b.lower[j, at]):
                        s += (f"\n  beta_{n}: {b.estimate[j, at]:+.3f} -{b.minus[j, at]:.3f} "
                              f"+{b.plus[j, at]:.3f}  (z0 = {b.z0[j, at]:+.2f})")
        return s


class Bootstrap:
    """One bootstrap run on one published AbelSide, in a background thread.

    The run only reads the frozen snapshot of the AbelSide (events, weights, matrices), so the live
    pipeline carries on and nothing needs locking. Results are published while it runs
    (every `publish_every` seconds and at the end), so the bands can be shown growing."""
    def __init__(self, side, n_boot=2000, unit="event", n_sigma=1.0, seed=None, publish_every=0.5):
        if side.basis is None or side.abel is None:
            raise RuntimeError("the Abel matrices are still being built")
        if unit not in ("event", "cycle"):
            raise ValueError("unit must be 'event' or 'cycle'")
        if unit == "cycle" and side.cycle is None:
            raise ValueError("this data has no cycle information (Sync Time)")
        self.side, self.unit, self.n_boot = side, unit, int(n_boot)
        self.n_sigma, self.publish_every = float(n_sigma), float(publish_every)
        self.seed = seed
        w = side.weights
        self._nz = np.flatnonzero(w)                    # rows with weight 0 contribute nothing: leave them out
        self._w = w[self._nz]
        self._binner = side.binner.subset(self._nz)
        if unit == "cycle":
            cyc = side.cycle[:len(w)][self._nz]
            _, self._cycle = np.unique(cyc, return_inverse=True)
            self._n_units = int(self._cycle.max()) + 1 if len(cyc) else 0
        else:
            self._cycle, self._n_units = None, len(self._nz)
        R = side.n_bins
        nb = len(side.abel.beta_orders)
        self._I = np.full((self.n_boot, R), np.nan)
        self._B = np.full((self.n_boot, nb, R), np.nan)
        self._mask = np.isnan(side.abel.beta)                         # the measurement's own mask
        self._done, self._t0, self._seconds = 0, None, 0.0
        self._stop, self._thread = threading.Event(), None
        self._lock = threading.Lock()
        self._result = None
        self._running, self._stopped, self._error = False, False, ""
        self._publish()

    # -- control ------------------------------------------------------
    def start(self):
        self._running = True
        self._t0 = time.monotonic()
        self._publish()                                  # from now on the result says "running", not "done"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def result(self):
        return self._result

    @property
    def running(self):
        return self._running

    # -- the work ------------------------------------------------------
    def _run(self):
        try:
            rng = np.random.default_rng(self.seed)
            side, basis = self.side, self.side.basis
            E = max(len(self._w), 1)
            chunk = int(np.clip(2e7 // E, 8, 128))                    # about 160 MB of weights at most
            if basis.nonlinear:
                chunk = min(chunk, 32)                               # constrained: one optimisation per sample, publish often
            min_int = side.settings.min_intensity if basis.nonlinear else 0.0
            last = time.monotonic()
            while self._done < self.n_boot and not self._stop.is_set():
                c = min(chunk, self.n_boot - self._done)
                k = rng.poisson(1.0, (c, self._n_units)).astype(float)
                if self._cycle is not None:
                    k = k[:, self._cycle]
                d = self._binner.d(k * self._w)
                res = basis.invert(d, side.dr, min_int)                # beta masked with the measurement's mask below (and, if constrained, where L_0 ~ 0)
                sl = slice(self._done, self._done + c)
                self._I[sl] = res.intensity
                self._B[sl] = np.where(self._mask[None], np.nan, res.beta)
                self._done += c
                if time.monotonic() - last >= self.publish_every:
                    self._publish()
                    last = time.monotonic()
        except Exception as exc:                                       # reported in the result, not raised
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._seconds = time.monotonic() - self._t0
            self._running = False
            self._stopped = self._stop.is_set() and self._done < self.n_boot
            self._publish()

    def _publish(self):
        a, n = self.side, self._done
        t = (time.monotonic() - self._t0) if (self._t0 is not None and self._running) else self._seconds
        I = B = None
        if n >= 2:
            I = bias_corrected_interval(a.abel.intensity, self._I[:n], self.n_sigma)
            if self._B.shape[1]:
                B = bias_corrected_interval(a.abel.beta, self._B[:n], self.n_sigma)
        self._result = BootstrapResult(
            side=a.side, signature=a.signature, unit=self.unit, n_sigma=self.n_sigma, n_done=n,
            n_total=self.n_boot, n_events=len(self._nz), n_units=self._n_units, seconds=t,
            running=self._running, stopped=self._stopped, error=self._error, rho=a.rho,
            beta_orders=a.abel.beta_orders, intensity=I, beta=B,
            intensity_samples=self._I[:n], beta_samples=self._B[:n])
