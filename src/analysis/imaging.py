"""Live detector images from the a, b, c, d columns of the selected events.

Pipeline (every stage is a small swappable, immutable object):

    a,b,c,d --Encoder--> (x, y) --Transform--> (x', y') --Binning--> pixel --> image

  Encoder    QuadEncoder: x = (a + d) / S,  y = (a + b) / S,  S = a + b + c + d.
             Events with S <= 0 have no position; they are counted, never imaged.
  Transform  Identity (default) or Shift(cx, cy). This is the hook for centering,
             circularisation, ... once polar binning needs them.
  Binning    CartesianBinning(nx, ny, x_range, y_range) -> image[iy, ix].
             A polar binning only has to provide .shape and .index(x, y).

Which events enter the image (mode):
  'subtracted'  every selected pair event with its background-subtracted weight
                (the statistically correct image of true coincidences; single
                pixels can be negative). Default.
  'raw'         events passing all cuts, +1 each (still contains accidentals).
  'all'         every searched event, ignoring all cuts (for comparison).

Positions are computed once per row and cached (rows are append-only); only
the cheap binning is redone when the selection, the binning or the transform
changes.
"""
import dataclasses
import threading
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from daq.data_loader import Poller


# ------------------------------------------------------------------ encoder
@dataclass(frozen=True)
class QuadEncoder:
    """x = sum(x_num) / sum(norm), y = sum(y_num) / sum(norm), from named columns.
    Defaults are x = (a+d)/(a+b+c+d), y = (a+b)/(a+b+c+d)."""
    x_num: tuple = ("a", "d")
    y_num: tuple = ("a", "b")
    norm: tuple = ("a", "b", "c", "d")

    def encode(self, cols, lo, hi):
        """(x, y) for rows lo:hi; NaN where the normalisation is not positive."""
        S = sum(cols[k][lo:hi] for k in self.norm)
        ok = S > 0
        safe = np.where(ok, S, 1.0)
        x = np.where(ok, sum(cols[k][lo:hi] for k in self.x_num) / safe, np.nan)
        y = np.where(ok, sum(cols[k][lo:hi] for k in self.y_num) / safe, np.nan)
        return x, y


# ---------------------------------------------------------------- transform
class Transform:
    def apply(self, x, y):
        raise NotImplementedError


@dataclass(frozen=True)
class Identity(Transform):
    def apply(self, x, y):
        return x, y


@dataclass(frozen=True)
class Shift(Transform):
    """Move the origin to (cx, cy): the centering step for later polar binning."""
    cx: float = 0.0
    cy: float = 0.0

    def apply(self, x, y):
        return x - self.cx, y - self.cy


@dataclass(frozen=True)
class Chain(Transform):
    """Apply several transforms in order, e.g. Chain((Shift(cx, cy), Circularise(...)))."""
    steps: tuple = ()

    def apply(self, x, y):
        for t in self.steps:
            x, y = t.apply(x, y)
        return x, y


@dataclass(frozen=True)
class Circularise(Transform):
    """Radial stretch r -> s(theta) * r about the origin (apply Shift first).

    s is given at the bin-centre angles `theta` [rad, ascending, in 0..2pi) and is
    interpolated linearly and periodically, so it is a continuous function of the
    event's angle: the binning used later (and the one the table was made with) are
    independent. Stored as tuples so that the transform is hashable and comparable."""
    theta: tuple = ()
    s: tuple = ()

    @cached_property
    def _arrays(self):
        return np.asarray(self.theta, float), np.asarray(self.s, float)

    def stretch(self, theta):
        th, s = self._arrays
        return np.interp(np.mod(theta, 2 * np.pi), th, s, period=2 * np.pi)

    def apply(self, x, y):
        f = self.stretch(np.arctan2(y, x))
        return x * f, y * f


# ------------------------------------------------------------------ binning
class Binning:
    shape = (0, 0)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def index(self, x, y):
        """Flat bin index into image.ravel(), -1 where outside / undefined."""
        raise NotImplementedError


@dataclass(frozen=True)
class CartesianBinning(Binning):
    nx: int = 256
    ny: int = 256
    x_range: tuple = (0.0, 1.0)
    y_range: tuple = (0.0, 1.0)

    @property
    def shape(self):
        return (self.ny, self.nx)            # image[iy, ix]

    @property
    def extent(self):                        # (x0, x1, y0, y1) for plotting
        return (*self.x_range, *self.y_range)

    def x_edges(self):
        return np.linspace(*self.x_range, self.nx + 1)

    def y_edges(self):
        return np.linspace(*self.y_range, self.ny + 1)

    def index(self, x, y):
        fx = (x - self.x_range[0]) * (self.nx / (self.x_range[1] - self.x_range[0]))
        fy = (y - self.y_range[0]) * (self.ny / (self.y_range[1] - self.y_range[0]))
        # right edge belongs to the last bin (same convention as np.histogram)
        ok = (fx >= 0) & (fx <= self.nx) & (fy >= 0) & (fy <= self.ny)   # NaN -> False
        ix = np.minimum(np.where(ok, fx, 0).astype(np.int64), self.nx - 1)
        iy = np.minimum(np.where(ok, fy, 0).astype(np.int64), self.ny - 1)
        return np.where(ok, iy * self.nx + ix, -1)


@dataclass(frozen=True)
class PolarBinning(Binning):
    """Bins in (radius, angle) around the origin (apply Shift first): image[itheta, ir].
    Angle theta = atan2(y, x) in 0..2pi, counter-clockwise from +x."""
    nr: int = 200
    ntheta: int = 96
    r_max: float = 0.5

    @property
    def shape(self):
        return (self.ntheta, self.nr)

    @property
    def extent(self):                        # (r0, r1, theta0, theta1 in degrees) for plotting
        return (0.0, self.r_max, 0.0, 360.0)

    def r_edges(self):
        return np.linspace(0.0, self.r_max, self.nr + 1)

    def theta_edges(self):                   # radians
        return np.linspace(0.0, 2 * np.pi, self.ntheta + 1)

    def index(self, x, y):
        fr = np.hypot(x, y) * (self.nr / self.r_max)
        ft = np.mod(np.arctan2(y, x), 2 * np.pi) * (self.ntheta / (2 * np.pi))
        ok = np.isfinite(fr) & np.isfinite(ft) & (fr <= self.nr)          # r_max itself is inside
        ir = np.minimum(np.where(ok, fr, 0).astype(np.int64), self.nr - 1)
        it = np.minimum(np.where(ok, ft, 0).astype(np.int64), self.ntheta - 1)
        return np.where(ok, it * self.nr + ir, -1)


# ------------------------------------------------------------------- result
@dataclass(frozen=True)
class DetectorImage:
    det: int
    mode: str
    binning: Binning
    image: np.ndarray             # shape = binning.shape; counts or weights
    n_rows: int                   # rows that entered (selected / weighted / all)
    n_no_position: int            # of those: a+b+c+d <= 0
    n_outside: int                # of those: outside the binning range
    selection_version: int

    @property
    def total(self):
        return float(self.image.sum())

    def projection(self, axis):
        """Sum over the other axis: 'x' -> 1D along x, 'y' -> along y."""
        return self.image.sum(axis=0 if axis == "x" else 1)

    def centroid(self):
        """(x, y) centre of mass of the positive part of the image, in the
        coordinates of the binning (i.e. after the transform). Handy for centering."""
        pos = np.clip(self.image, 0, None)
        if pos.sum() == 0 or not isinstance(self.binning, CartesianBinning):
            return None
        xc = 0.5 * (self.binning.x_edges()[1:] + self.binning.x_edges()[:-1])
        yc = 0.5 * (self.binning.y_edges()[1:] + self.binning.y_edges()[:-1])
        return float((pos.sum(0) * xc).sum() / pos.sum()), float((pos.sum(1) * yc).sum() / pos.sum())

    def ascii(self, size=28):
        """Coarse terminal rendering (y up, x right); sqrt intensity scale."""
        img = np.clip(self.image, 0, None)
        ny, nx = img.shape
        gy, gx = max(ny // size, 1), max(nx // size, 1)
        img = img[:(ny // gy) * gy, :(nx // gx) * gx]
        img = img.reshape(img.shape[0] // gy, gy, img.shape[1] // gx, gx).sum((1, 3))
        top = max(img.max(), 1e-12)
        ramp = " .:-=+*#%@"
        return "\n".join("".join(ramp[min(int(9 * (v / top) ** 0.5), 9)] for v in row)
                         for row in img[::-1])

    def summary(self):
        c = self.centroid()
        return (f"det {self.det} [{self.mode}] image {self.image.shape[1]}x{self.image.shape[0]}: "
                f"total {self.total:.1f} from {self.n_rows} rows "
                f"({self.n_no_position} without position, {self.n_outside} outside range)"
                + (f", centroid ({c[0]:.4f}, {c[1]:.4f})" if c else ""))


@dataclass(frozen=True)
class ImagingResult:
    version: int
    status: str
    a: DetectorImage
    b: DetectorImage

    def __getitem__(self, det):
        if det in ("a", self.a.det):
            return self.a
        if det in ("b", self.b.det):
            return self.b
        raise KeyError(det)

    def summary(self):
        return self.a.summary() + "\n" + self.b.summary()


@dataclass(frozen=True)
class ImagerSettings:
    encoder: QuadEncoder = QuadEncoder()
    binning_a: Binning = CartesianBinning()
    binning_b: Binning = CartesianBinning()
    transform_a: Transform = Identity()
    transform_b: Transform = Identity()
    mode: str = "subtracted"
    version: int = 0


def _padded(a, n):
    """a cut or zero-padded to length n (rows newer than the correction are not displaced)."""
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.zeros(n - len(a))])


class _Rows:
    """Per-detector row cache: raw positions (encoder only) and pixel indices."""
    def __init__(self):
        self.pos_key = None
        self.x = self.y = np.empty(0)
        self.idx_key = None
        self.idx = np.empty(0, np.int64)


# ------------------------------------------------------------------- object
class Imager(Poller):
    """Builds one image per detector from a SelectionFilter, live.

    Detectors are addressed by number or 'a' / 'b' as in SelectionFilter."""
    def __init__(self, source, selection, encoder=None, binning=None, mode="subtracted", corrector=None):
        self.source, self.selection = source, selection
        self.corrector = corrector              # optional PositionCorrector: per-row displacement of the positions
        self.det_a, self.det_b = selection.det_a, selection.det_b
        b = binning or CartesianBinning()
        self._settings = ImagerSettings(encoder=encoder or QuadEncoder(),
                                        binning_a=b, binning_b=b, mode=self._mode(mode))
        self._lock = threading.Lock()
        self._rows = {"a": _Rows(), "b": _Rows()}
        self._key, self._result, self._version = None, None, 0
        self._latest = {}                       # inputs of the last evaluation (fit_range)
        self._raw_xy = {}                       # raw positions of the rows of the last evaluation

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

    @staticmethod
    def _mode(mode):
        if mode not in ("subtracted", "raw", "all"):
            raise ValueError("mode must be 'subtracted', 'raw' or 'all'")
        return mode

    def _change(self, **kw):
        with self._lock:
            s = self._settings
            self._settings = dataclasses.replace(s, version=s.version + 1, **kw)

    def set_mode(self, mode):
        self._change(mode=self._mode(mode))

    def set_encoder(self, encoder):
        self._change(encoder=encoder)

    def set_binning(self, binning, det=None):
        """Binning for one detector, or both if det is None."""
        sides = ("a", "b") if det is None else (self._side(det),)
        self._change(**{f"binning_{k}": binning for k in sides})

    def set_transform(self, transform, det=None):
        sides = ("a", "b") if det is None else (self._side(det),)
        self._change(**{f"transform_{k}": transform for k in sides})

    def fit_range(self, det, quantiles=(0.1, 99.9), pad=0.05, square=True, nbins=None):
        """One-off: set a Cartesian range around the currently selected events
        (percentiles of x' and y', padded). The range then stays fixed so that
        images remain comparable while data accumulate. Returns the binning."""
        k = self._side(det)
        if k not in self._latest:
            raise RuntimeError("no data yet: wait for the first image")
        x, y = self._latest[k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            raise RuntimeError("too few events to fit a range")
        x0, x1 = (float(v) for v in np.percentile(x[ok], quantiles))
        y0, y1 = (float(v) for v in np.percentile(y[ok], quantiles))
        wx, wy = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if square:
            wx = wy = max(wx, wy)
        cur = getattr(self._settings, f"binning_{k}")
        n = nbins or getattr(cur, "nx", 256)
        b = CartesianBinning(n, nbins or getattr(cur, "ny", 256),
                             (cx - wx / 2, cx + wx / 2), (cy - wy / 2, cy + wy / 2))
        self.set_binning(b, det)
        return b

    def positions(self, det):
        """(x', y') of the events selected in the last evaluation (after the
        transform), or None. Used e.g. to size a radial range."""
        return self._latest.get(self._side(det))

    def raw_xy(self, det):
        """(x, y): the positions of all rows used in the last evaluation (encoder positions, minus the
        displacement of the position correction when one is passed on; before any transform), aligned
        with the rows of the selection; None before the first evaluation."""
        return self._raw_xy.get(self._side(det))

    # -- evaluation ------------------------------------------------------
    def result(self):
        return self._result

    def poll(self) -> int:
        snap = self.source.snapshot()
        sel = self.selection.result()
        if sel is None:
            return 0
        sa, sb = snap.detectors.get(self.det_a), snap.detectors.get(self.det_b)
        if sa is None or sb is None:
            return 0
        if (sel.run, sel.epoch_a, sel.epoch_b) != (snap.run, sa.epoch, sb.epoch) \
                or sel.n_a > sa.n or sel.n_b > sb.n:
            return 0                                  # selection belongs to other data
        s = self._settings
        corr = {k: self._active_correction(k, sel) for k in ("a", "b")}
        key = (sel.version, s.version, tuple(c.version if c else 0 for c in corr.values()))
        if key == self._key:
            return 0
        imgs = {}
        for k, det, dsnap, n in (("a", self.det_a, sa, sel.n_a), ("b", self.det_b, sb, sel.n_b)):
            imgs[k] = self._image(k, det, dsnap, n, sel, s, corr[k])
        self._key = key
        self._version += 1
        self._result = ImagingResult(self._version, sel.status, imgs["a"], imgs["b"])
        return 1

    def _active_correction(self, k, sel):
        """The published position correction if it is on, for this side, and belongs to this data."""
        c = self.corrector.result() if self.corrector is not None else None
        if c is None or not c.active or c.dx.get(k) is None:
            return None
        return c if c.run == (sel.run, sel.epoch_a, sel.epoch_b) else None

    def _image(self, k, det, dsnap, n, sel, s, corr=None):
        rows = self._rows[k]
        binning, transform = getattr(s, f"binning_{k}"), getattr(s, f"transform_{k}")
        # positions: only new rows are encoded
        pk = (dsnap.epoch, s.encoder)
        if rows.pos_key != pk:
            rows.pos_key, rows.x, rows.y = pk, np.empty(0), np.empty(0)
            rows.idx_key = None
        have = len(rows.x)
        if have < n:
            x, y = s.encoder.encode(dsnap.columns, have, n)
            rows.x, rows.y = np.concatenate([rows.x, x]), np.concatenate([rows.y, y])
        # pixel indices: only new rows unless binning/transform changed
        ik = (binning, transform, corr.version if corr else 0)
        if rows.idx_key != ik:
            rows.idx_key, rows.idx = ik, np.empty(0, np.int64)
        x_all, y_all = rows.x[:n], rows.y[:n]
        if corr is not None:                                  # position correction: subtract the displacement
            x_all = x_all - _padded(corr.dx[k], n)
            y_all = y_all - _padded(corr.dy[k], n)
        done = len(rows.idx)
        if done < n:
            xt, yt = transform.apply(x_all[done:n], y_all[done:n])
            rows.idx = np.concatenate([rows.idx, binning.index(xt, yt)])
        idx = rows.idx[:n]

        valid = getattr(sel, f"valid_{k}")
        mask, weight = getattr(sel, f"mask_{k}"), getattr(sel, f"weight_{k}")
        if s.mode == "all":
            used, w = valid, None
        elif s.mode == "raw":
            used, w = valid & mask, None
        else:
            used, w = valid & (weight != 0), weight
        finite = np.isfinite(x_all)
        ok = used & (idx >= 0)
        img = np.bincount(idx[ok], weights=None if w is None else w[ok],
                          minlength=binning.size).reshape(binning.shape).astype(float)
        # what fit_range sees: transformed positions of the selected (mask) events
        pick = valid & (mask if s.mode != "all" else True)
        self._latest[k] = transform.apply(x_all[pick], y_all[pick])
        self._raw_xy[k] = (x_all, y_all)                      # views of the rows (copies when corrected)
        return DetectorImage(det=det, mode=s.mode, binning=binning, image=img,
                             n_rows=int(used.sum()), n_no_position=int((used & ~finite).sum()),
                             n_outside=int((used & finite & (idx < 0)).sum()),
                             selection_version=sel.version)
