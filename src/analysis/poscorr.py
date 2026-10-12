"""Position of the hit versus coincidence time.

For every coincidence pair (A event, B event) the hit position (x, y) on each detector is
computed from the a, b, c, d charges (the same QuadEncoder as the images) and related to
dt = t_B - t_A. Plotting x and y against dt shows e.g. whether the arrival time of the partner
depends on where the ion hit (time-of-flight spread, a tilted detector, walk of the timing).

This is a plain look at the coincidence pairs: no Sum or time cuts and no accidental
subtraction (accidentals are flat in dt and just add the mean position of a random hit).

position_correlation(...) returns one PosCorr per detector:
  binned means    mean x / mean y and their standard errors in bins of dt
  raw points      a random subsample (<= max_points) of the individual pairs

fit_correction(...) fits, inside the coincidence window, how the electron position depends on the
neutral position and on dt, and removes it (see docs/notes/main.tex, "Two-Dimensional Correction Model"):

    r_corr = r_e - M r_n - c tau - d,     tau = dt,

by least squares of (x_e, y_e) on (x_n, y_n, tau), weighted with the accidental-subtraction weights
(+1 for a pair in the window, -1/K for a pair in each of the K control windows, moved into the window
in dt), so that only the true coincidences determine the fit. d is not a free offset: it is fixed so
that the corrected image keeps the mean position of the uncorrected one.

correction_images(...) bins the weighted electron positions, before and after the correction, on a fixed
detector grid for the whole window and for equal slices of dt: a flat dt dependence of the corrected
position is what a perfect correction leaves; a remaining systematic trend is information on the
missing momentum component.
"""
from dataclasses import dataclass

import numpy as np

from analysis.filters import _in_range
from analysis.imaging import QuadEncoder




@dataclass(frozen=True)
class PosCorr:
    det: int
    side: str                  # 'a' or 'b'
    lo: float                  # dt range [us]
    hi: float
    centers: np.ndarray        # bin centres [us]
    count: np.ndarray          # pairs with a position in each bin
    mean_x: np.ndarray         # NaN in bins without pairs
    mean_y: np.ndarray
    sem_x: np.ndarray          # standard error of the mean (NaN for fewer than 2 pairs)
    sem_y: np.ndarray
    raw_dt: np.ndarray         # subsample of the individual pairs
    raw_x: np.ndarray
    raw_y: np.ndarray
    n_pairs: int               # pairs inside the dt range
    n_valid: int               # ... of which have a position on this detector
    corr_mean_x: np.ndarray = None   # the same means after the correction (electron detector only,
    corr_mean_y: np.ndarray = None   # and only when a correction was given); the uncorrected
    corr_sem_x: np.ndarray = None    # ones above are left as they are
    corr_sem_y: np.ndarray = None
    corr_count: np.ndarray = None


def _stats(idx, nb, v):
    """Mean and standard error of v per bin (idx = bin index of every value)."""
    n = np.bincount(idx, minlength=nb).astype(float)
    s1 = np.bincount(idx, weights=v, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        dev = v - mean[idx]                            # two-pass: no cancellation
        var = np.bincount(idx, weights=dev * dev, minlength=nb) / (n - 1)
        sem = np.where(n > 1, np.sqrt(var / n), np.nan)
    return n, mean, sem


def _rows_ok(res, snap, det_a, det_b):
    """bool per pair: both rows exist in the snapshot (a result can be newer than the snapshot).
    None if a detector is missing."""
    ok = np.ones(len(res.dt_us), bool)
    for det, rows in ((det_a, res.ia), (det_b, res.ib)):
        s = snap.detectors.get(det)
        if s is None:
            return None
        ok &= rows < min(len(s.columns[k]) for k in ("a", "b", "c", "d"))
    return ok


def _positions(res, snap, det_a, det_b, idx, encoder):
    """xa, ya, xb, yb of the pairs `idx` (NaN = no position)."""
    out = {}
    for side, det, rows in (("a", det_a, res.ia), ("b", det_b, res.ib)):
        cols = snap.detectors[det].columns
        r = rows[idx]
        sub = {k: np.asarray(cols[k])[r].astype(float) for k in ("a", "b", "c", "d")}
        out["x" + side], out["y" + side] = encoder.encode(sub, 0, len(r))
    return out


def _gather(res, snap, det_a, det_b, lo, hi, encoder):
    """dt and the positions of both detectors for all pairs with lo <= dt < hi (NaN = no position).
    None if a detector is missing."""
    ok = _rows_ok(res, snap, det_a, det_b)
    if ok is None:
        return None
    idx = np.nonzero(ok & (res.dt_us >= lo) & (res.dt_us < hi))[0]
    out = dict(dt=res.dt_us[idx], n_pairs=len(idx))
    out.update(_positions(res, snap, det_a, det_b, idx, encoder))
    return out


def weighted_pairs(res, snap, det_a, det_b, window, background=True, encoder=None):
    """The pairs that determine the true coincidences of `window` = (lo, hi): the pairs inside it
    (weight +1) and, with background=True, the pairs of every control window (weight -1/K, K = number
    of control windows) with their dt moved back into the window. Returns a dict with xa, ya, xb, yb,
    dt (= tau), w, n_pairs (pairs in the window itself); None if a detector is missing."""
    ok = _rows_ok(res, snap, det_a, det_b)
    if ok is None:
        return None
    lo, hi = float(window[0]), float(window[1])
    dt = res.dt_us
    parts = [(lo, hi, 0.0, 1.0)]
    if background:
        _, offs, _ = res._bkg_offsets((lo, hi))
        parts += [(lo + o, hi + o, o, -1.0 / len(offs)) for o in offs]
    idx, tau, w = [], [], []
    for a, b, o, wt in parts:
        i = np.nonzero(ok & (dt >= a) & (dt <= b))[0]
        idx.append(i); tau.append(dt[i] - o); w.append(np.full(len(i), wt))
    n_main = len(idx[0])
    idx = np.concatenate(idx)
    out = dict(dt=np.concatenate(tau), w=np.concatenate(w), n_pairs=n_main)
    out.update(_positions(res, snap, det_a, det_b, idx, encoder or QuadEncoder()))
    return out


def _electron_neutral(p, side_e):
    sn = "b" if side_e == "a" else "a"
    return p["x" + side_e], p["y" + side_e], p["x" + sn], p["y" + sn]


def _legendre(u, k):
    """Legendre polynomial P_k(u), k >= 1 (u in -1..1 across the fit window)."""
    return np.polynomial.legendre.legval(u, [0] * k + [1])


@dataclass(frozen=True)
class Correction:
    """r_corr = r_e - M r_n - f(tau) - const, fitted on the pairs inside `window` (see fit_correction).
    f is a polynomial of order tau_order in tau (Legendre basis in u = (tau - centre)/scale, scale = 3
    standard deviations of the pairs' dt; order 1 = c tau, order 0 = no dt term), shifted so that the corrected image keeps the mean electron position."""
    side_e: str                # 'a' or 'b': the detector whose positions are corrected
    det_e: int
    det_n: int
    window: tuple              # (lo, hi) dt range of the fit [us]
    n: float                   # sum of the weights = estimate of the true pairs used
    n_pairs: int               # pairs in the window with a position on both detectors
    M: np.ndarray              # 2x2: rows x_e, y_e; columns x_n, y_n
    c: np.ndarray              # (2,) linear dt term per us (the P1 coefficient / half window)
    d: np.ndarray              # (2,) offset of the order <= 1 formula (keeps the mean electron position)
    coef: np.ndarray           # 2x3 [x_n, y_n, tau] (= M | c) ...
    err: np.ndarray            # ... and its standard errors
    mean_e: np.ndarray         # mean electron position (before and after the correction)
    mean_n: np.ndarray
    mean_tau: float
    std_before: np.ndarray     # (2,) standard deviation of x_e, y_e before / after
    std_after: np.ndarray
    r2: np.ndarray             # (2,) fraction of the variance removed
    use_neutral: bool = True
    use_tau: bool = True       # tau_order >= 1
    background: bool = True    # accidentals subtracted (weighted fit)
    tau_order: int = 1
    tau_centre: float = 0.0    # u = (tau - tau_centre) / tau_scale; scale = 3 weighted standard deviations of dt
    tau_scale: float = 1.0
    poly: np.ndarray = None    # (2, tau_order) coefficients of P_1..P_K(u)
    poly_err: np.ndarray = None
    poly_mean: np.ndarray = None   # (tau_order,) weighted mean of P_k(u) over the fit pairs
    flat_before: np.ndarray = None  # (2,) chi2 of x_e, y_e being constant in dt (equal bins) ...
    flat_after: np.ndarray = None   # ... before / after the correction
    flat_dof: int = 0

    @property
    def t_values(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.coef / self.err

    def _poly_part(self, tau):
        """The dt term subtracted from (x_e, y_e): shape (2, ...) (zero for order 0)."""
        tau = np.asarray(tau, float)
        out = np.zeros((2,) + tau.shape)
        if self.tau_order:
            u = (tau - self.tau_centre) / self.tau_scale
            for k in range(1, self.tau_order + 1):
                pk = _legendre(u, k) - self.poly_mean[k - 1]
                out[0] += self.poly[0, k - 1] * pk
                out[1] += self.poly[1, k - 1] * pk
        return out

    def apply(self, xe, ye, xn, yn, tau):
        """Corrected electron position for any pairs (NaN where an input is NaN)."""
        dx = np.asarray(xn, float) - self.mean_n[0]
        dy = np.asarray(yn, float) - self.mean_n[1]
        pt = self._poly_part(tau)
        xc = np.asarray(xe, float) - (self.M[0, 0] * dx + self.M[0, 1] * dy + pt[0])
        yc = np.asarray(ye, float) - (self.M[1, 0] * dx + self.M[1, 1] * dy + pt[1])
        return xc, yc

    def shape(self, tau):
        """The fitted dt dependence of the electron position (mean + polynomial term), (x(tau), y(tau))."""
        pt = self._poly_part(tau)
        return self.mean_e[0] + pt[0], self.mean_e[1] + pt[1]

    def text(self):
        t = self.t_values
        lines = [f"correction of detector {self.side_e.upper()} (det {self.det_e}) from the neutral detector "
                 f"(det {self.det_n})",
                 f"fit window {self.window[0]:.2f} .. {self.window[1]:.2f} us, {self.n_pairs} pairs, "
                 + (f"weighted: {self.n:.0f} true pairs" if self.background else "unweighted (accidentals kept)"),
                 "r_corr = r_e - M r_n - f(tau) - const    (value +- error, [t = value/error])"
                 + (f";  f = polynomial of order {self.tau_order} in dt" if self.tau_order > 1 else ""), ""]
        for i, name in enumerate(("x_e", "y_e")):
            row = [f"{par} = {self.coef[i, j]:+.4g} +- {self.err[i, j]:.2g} [{t[i, j]:+.1f}]"
                   for j, par in enumerate((f"M{i}x", f"M{i}y", f"c{i}")) if j < 2 or self.tau_order]
            lines.append(f"{name}: " + ";  ".join(row))
        if self.tau_order > 1:
            for i, name in enumerate(("x_e", "y_e")):
                lines.append(f"{name} dt terms P1..P{self.tau_order} (x1000): "
                             + " ".join(f"{1000 * v:+.2f}" for v in self.poly[i]))
        lines += ["", f"std before: x {self.std_before[0]:.5f}  y {self.std_before[1]:.5f}",
                  f"std after:  x {self.std_after[0]:.5f}  y {self.std_after[1]:.5f}",
                  f"variance removed: x {100 * self.r2[0]:.2f} %  y {100 * self.r2[1]:.2f} %"]
        if self.flat_dof > 0:
            lines.append(f"flat in dt? chi2/dof ({self.flat_dof} dof): before x {self.flat_before[0] / self.flat_dof:.1f}"
                         f" y {self.flat_before[1] / self.flat_dof:.1f};  after x {self.flat_after[0] / self.flat_dof:.1f}"
                         f" y {self.flat_after[1] / self.flat_dof:.1f}   (1 = flat)")
        return "\n".join(lines)


def _flatness(v, tau, w, window, nbins=10, min_frac=0.005):
    """chi2 (and dof) of the weighted bin means of v being constant across the window."""
    edges = np.linspace(window[0], window[1], nbins + 1)
    ms, es = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (tau >= a) & (tau <= b if b == edges[-1] else tau < b)
        W = w[s].sum()
        if W < max(10.0, min_frac * w.sum()) or (w[s] > 0).sum() < 5:
            continue
        m = (w[s] * v[s]).sum() / W
        e = np.sqrt(((w[s] * (v[s] - m)) ** 2).sum()) / abs(W)
        if e > 0:
            ms.append(m); es.append(e)
    if len(ms) < 3:
        return np.nan, 0
    ms, es = np.array(ms), np.array(es)
    mm = (ms / es ** 2).sum() / (1 / es ** 2).sum()
    return float(((ms - mm) ** 2 / es ** 2).sum()), len(ms) - 1


def fit_correction(res, snap, det_a, det_b, side_e="a", window=None, use_neutral=True, use_tau=True,
                   background=True, encoder=None, tau_order=None, ridge=0.1):
    """Weighted least-squares fit of the electron position on the neutral position and a polynomial in dt,
    inside `window` (default: the coincidence window of the result). side_e is the electron detector,
    'a' or 'b'; the other one is the neutral detector. use_neutral switches the neutral-position term
    off; tau_order is the order of the dt polynomial (0 = none; default 1 if use_tau else 0).
    background=True weights with the accidental subtraction (weighted_pairs); the errors are the
    sandwich (robust) estimate, valid for signed weights. The columns are fitted in a basis orthonormal for |w| (stable for
    higher dt orders). None when there is too little to fit."""
    if tau_order is None:
        tau_order = 1 if use_tau else 0
    tau_order = int(max(0, tau_order))
    if res is None or snap is None or len(res.dt_us) == 0:
        return None
    window = window or res.window_us
    if window is None:
        return None
    p = weighted_pairs(res, snap, det_a, det_b, window, background, encoder)
    if p is None:
        return None
    xe, ye, xn, yn = _electron_neutral(p, side_e)
    good = np.isfinite(xe) & np.isfinite(ye) & np.isfinite(xn) & np.isfinite(yn)
    xe, ye, xn, yn, tau, w = xe[good], ye[good], xn[good], yn[good], p["dt"][good], p["w"][good]
    W = w.sum()
    n_pairs = int((w > 0).sum())
    if n_pairs < 20 or W < 20:
        return None
    wm = lambda v: float((w * v).sum() / W)
    mean_e = np.array([wm(xe), wm(ye)])
    mean_n = np.array([wm(xn), wm(yn)])
    mean_tau = wm(tau)
    lo, hi = float(window[0]), float(window[1])
    t0 = mean_tau
    h = 3.0 * float(np.sqrt(max((w * (tau - mean_tau) ** 2).sum() / W, 1e-24)))
    u = (tau - t0) / h
    P = [_legendre(u, k) for k in range(1, tau_order + 1)]
    pmean = np.array([wm(q) for q in P])
    cols = [xn - mean_n[0], yn - mean_n[1]] + [q - m for q, m in zip(P, pmean)]
    use = np.array([use_neutral, use_neutral] + [True] * tau_order)
    Z = np.column_stack(cols)
    Zu = Z[:, use]
    full, errs = np.zeros((2, Z.shape[1])), np.full((2, Z.shape[1]), np.nan)
    std_b, std_a, r2 = np.zeros(2), np.zeros(2), np.zeros(2)
    resids = []
    if Zu.shape[1]:
        # The columns (x_n, y_n, P_1..P_K) are strongly collinear and the signed accidental weights make the
        # normal matrix noisy. Fit in a basis that is orthonormal for the weights |w| (Cholesky of their
        # Gram matrix): there the signed normal matrix is ~ (net fraction) x identity and well conditioned.
        aw = np.abs(w)
        sc = np.sqrt(np.maximum((aw[:, None] * Zu * Zu).sum(axis=0) / aw.sum(), 1e-300))
        Zs = Zu / sc
        G = Zs.T @ (aw[:, None] * Zs) / aw.sum()
        T = np.linalg.inv(np.linalg.cholesky(G + 1e-12 * np.eye(len(G)))).T
        Q = Zs @ T
        A = Q.T @ (w[:, None] * Q)
        # Orders >= 2 add directions that the net (accidental-subtracted) weights barely constrain, so their
        # coordinates in this orthonormal basis are shrunk a little (ridge): no wild extrapolation into the wings.
        nneu = 2 if use_neutral else 0
        damp = np.zeros(len(A))
        damp[nneu + 1:] = ridge * aw.sum()
        Ainv = np.linalg.pinv(A + np.diag(damp))
    for i, v in enumerate((xe, ye)):
        dv = v - mean_e[i]
        resid = dv
        if Zu.shape[1]:
            b = Ainv @ (Q.T @ (w * dv))
            resid = dv - Q @ b
            meat = Q.T @ (((w * resid) ** 2)[:, None] * Q)
            full[i, use] = (T @ b) / sc
            errs[i, use] = np.sqrt(np.maximum(np.diag(T @ (Ainv @ meat @ Ainv) @ T.T), 0.0)) / sc
        resids.append(resid + mean_e[i])
        ss0, ss1 = float((w * dv * dv).sum()), float((w * resid * resid).sum())
        std_b[i], std_a[i] = np.sqrt(max(ss0 / W, 0)), np.sqrt(max(ss1 / W, 0))
        r2[i] = 1.0 - ss1 / ss0 if ss0 > 0 else 0.0
    M, c = full[:, :2].copy(), (full[:, 2] / h if tau_order else np.zeros(2))
    coef = np.column_stack([M, c])
    err = np.column_stack([errs[:, :2], errs[:, 2] / h if tau_order else np.full(2, np.nan)])
    d = -(M @ mean_n + c * mean_tau)
    fb, dof = zip(*(_flatness(v, tau, w, (lo, hi)) for v in (xe, ye)))
    fa, _ = zip(*(_flatness(v, tau, w, (lo, hi)) for v in resids))
    return Correction(side_e=side_e, det_e=det_a if side_e == "a" else det_b,
                      det_n=det_b if side_e == "a" else det_a,
                      window=(lo, hi), n=float(W), n_pairs=n_pairs,
                      M=M, c=c, d=d, coef=coef, err=err,
                      mean_e=mean_e, mean_n=mean_n, mean_tau=mean_tau, std_before=std_b, std_after=std_a,
                      r2=r2, use_neutral=bool(use_neutral), use_tau=tau_order >= 1, background=bool(background),
                      tau_order=tau_order, tau_centre=float(t0), tau_scale=float(h), poly=full[:, 2:].copy(), poly_err=errs[:, 2:].copy(), poly_mean=pmean,
                      flat_before=np.array(fb), flat_after=np.array(fa), flat_dof=int(min(dof)))


# ------------------------------------------------------------------ images
@dataclass(frozen=True)
class SliceImage:
    lo: float                  # dt slice [us]
    hi: float
    n_true: float              # sum of the weights (true pairs)
    n_pairs: int
    image: np.ndarray          # corrected electron image [iy, ix], weighted
    image_raw: np.ndarray      # the same pairs, uncorrected
    mean_corr: np.ndarray      # weighted mean corrected position (x, y)
    err_corr: np.ndarray       #   ... and its error
    mean_raw: np.ndarray       # weighted mean uncorrected position
    count: float = 0.0


@dataclass(frozen=True)
class CorrectionImages:
    extent: tuple              # (x0, x1, y0, y1) of the fixed detector grid
    raw: np.ndarray            # whole window, uncorrected
    corrected: np.ndarray      # whole window, corrected
    slices: tuple              # SliceImage per dt slice

    def text(self):
        lines = ["", "slices of dt: weighted mean position x1000 of the corrected image (uncorrected)"]
        for s in self.slices:
            lines.append(f"{s.lo:7.2f}..{s.hi:7.2f} us  n={s.n_true:8.0f}  "
                         f"x {1000 * s.mean_corr[0]:7.2f}+-{1000 * s.err_corr[0]:.2f} ({1000 * s.mean_raw[0]:7.2f})  "
                         f"y {1000 * s.mean_corr[1]:7.2f}+-{1000 * s.err_corr[1]:.2f} ({1000 * s.mean_raw[1]:7.2f})")
        return "\n".join(lines)


def _wimage(x, y, w, binning):
    """Weighted 2D histogram on a CartesianBinning, image[iy, ix]."""
    ok = np.isfinite(x) & np.isfinite(y)
    return np.histogram2d(y[ok], x[ok], bins=[binning.ny, binning.nx],
                          range=[binning.y_range, binning.x_range], weights=w[ok])[0]


def _wmean(v, w):
    W = w.sum()
    if W <= 0:
        return np.nan, np.nan
    m = (w * v).sum() / W
    return float(m), float(np.sqrt(((w * (v - m)) ** 2).sum()) / abs(W))


def correction_images(res, snap, det_a, det_b, cor, binning, n_slices=3, encoder=None):
    """Electron images, accidentals subtracted, on the fixed grid `binning` (a CartesianBinning, the
    same as the Detectors tab uses): the whole fit window before and after the correction, and the
    corrected image for each of n_slices equal slices of dt across the window. None if no data."""
    encoder = encoder or QuadEncoder()
    lo, hi = cor.window
    edges = np.linspace(lo, hi, max(1, int(n_slices)) + 1)

    def part(a, b):
        p = weighted_pairs(res, snap, det_a, det_b, (a, b), cor.background, encoder)
        if p is None:
            return None
        xe, ye, xn, yn = _electron_neutral(p, cor.side_e)
        ok = np.isfinite(xe) & np.isfinite(ye) & np.isfinite(xn) & np.isfinite(yn)
        xe, ye, xn, yn, tau, w = xe[ok], ye[ok], xn[ok], yn[ok], p["dt"][ok], p["w"][ok]
        xc, yc = cor.apply(xe, ye, xn, yn, tau)
        return xe, ye, xc, yc, w, int((w > 0).sum())

    whole = part(lo, hi)
    if whole is None:
        return None
    xe, ye, xc, yc, w, _ = whole
    slices = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = part(a, b)
        xe_, ye_, xc_, yc_, w_, npairs = r
        mx, ex = _wmean(xc_, w_); my, ey = _wmean(yc_, w_)
        rx, _ = _wmean(xe_, w_); ry, _ = _wmean(ye_, w_)
        slices.append(SliceImage(lo=float(a), hi=float(b), n_true=float(w_.sum()), n_pairs=npairs,
                                 image=_wimage(xc_, yc_, w_, binning), image_raw=_wimage(xe_, ye_, w_, binning), mean_corr=np.array([mx, my]),
                                 err_corr=np.array([ex, ey]), mean_raw=np.array([rx, ry])))
    return CorrectionImages(extent=tuple(binning.extent), raw=_wimage(xe, ye, w, binning),
                            corrected=_wimage(xc, yc, w, binning), slices=tuple(slices))


@dataclass(frozen=True)
class DtProfile:
    """Accidental-subtracted means of the hit position in bins of dt, one per detector."""
    det: int
    side: str
    lo: float
    hi: float
    centers: np.ndarray
    weight: np.ndarray         # sum of the weights per bin = estimate of the true pairs
    mean_x: np.ndarray         # NaN where the true weight is too small (< 3 pairs)
    mean_y: np.ndarray
    err_x: np.ndarray          # error of the weighted mean (accidental noise included)
    err_y: np.ndarray
    corr_mean_x: np.ndarray = None   # electron detector only, when a correction was given
    corr_mean_y: np.ndarray = None
    corr_err_x: np.ndarray = None
    corr_err_y: np.ndarray = None
    background: bool = True


def _wbin(idx, nb, w, v):
    """Weighted mean and error of v per bin; NaN where the summed weight is below 3."""
    W = np.bincount(idx, weights=w, minlength=nb)
    S = np.bincount(idx, weights=w * v, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(W >= 3.0, S / W, np.nan)
        dev = w * (v - np.nan_to_num(m)[idx])
        e = np.where(W >= 3.0, np.sqrt(np.bincount(idx, weights=dev * dev, minlength=nb)) / np.abs(W), np.nan)
    return W, m, e


def dt_profile(res, snap, det_a, det_b, half_us=20.0, bin_us=0.5, centre=None, background=True,
               correction=None, encoder=None):
    """{'a': DtProfile, 'b': DtProfile}: the mean hit position (x, y) of the TRUE coincidences in bins of
    dt. Each bin counts its pairs with +1 and the pairs of the same bin moved by every control-window
    offset with -1/K (weighted_pairs), so accidentals cancel bin by bin instead of being averaged in.
    With `correction` the electron detector also gets the corrected means (corr_*). Bins use the same
    edges as position_correlation. None if there is no result or a detector is missing."""
    if res is None or snap is None or len(res.dt_us) == 0:
        return None
    if centre is None:
        pk = res.peak()
        centre = pk[0] if pk is not None else 0.0
    lo, hi = float(centre) - half_us, float(centre) + half_us
    nb = int(min(4000, max(1, round((hi - lo) / max(bin_us, 1e-6)))))
    centers = 0.5 * (np.linspace(lo, hi, nb + 1)[:-1] + np.linspace(lo, hi, nb + 1)[1:])
    p = weighted_pairs(res, snap, det_a, det_b, (lo, hi), background, encoder)
    if p is None:
        return None
    tau, w = p["dt"], p["w"]
    inside = (tau >= lo) & (tau < hi)
    idx_all = np.minimum(((tau - lo) / (hi - lo) * nb).astype(np.int64), nb - 1)
    xy = {"a": (p["xa"], p["ya"]), "b": (p["xb"], p["yb"])}
    out = {}
    for side, det in (("a", det_a), ("b", det_b)):
        x, y = xy[side]
        good = inside & np.isfinite(x) & np.isfinite(y)
        W, mx, ex = _wbin(idx_all[good], nb, w[good], x[good])
        _, my, ey = _wbin(idx_all[good], nb, w[good], y[good])
        extra = {}
        if correction is not None and correction.side_e == side:
            other = "b" if side == "a" else "a"
            xc, yc = correction.apply(x, y, *xy[other], tau)
            gc = inside & np.isfinite(xc) & np.isfinite(yc)
            _, cx, cex = _wbin(idx_all[gc], nb, w[gc], xc[gc])
            _, cy, cey = _wbin(idx_all[gc], nb, w[gc], yc[gc])
            extra = dict(corr_mean_x=cx, corr_mean_y=cy, corr_err_x=cex, corr_err_y=cey)
        out[side] = DtProfile(det=det, side=side, lo=lo, hi=hi, centers=centers, weight=W, mean_x=mx, mean_y=my,
                              err_x=ex, err_y=ey, background=bool(background), **extra)
    return out


@dataclass(frozen=True)
class TimingWalk:
    """dt' = dt - sum_j k_j (pos_j - mean_j): the part of dt that is linear in the hit positions
    (positions j = x_a, y_a, x_b, y_b), fitted inside the peak. A position-dependent timing of a detector
    (signal propagation, walk) shows up here; what stays in the position-vs-dt' profile is not a timing effect
    of that kind."""
    window: tuple              # (lo, hi) dt range of the fit [us]
    n: float                   # true pairs (sum of the weights)
    names: tuple               # ("x_a", "y_a", "x_b", "y_b")
    used: tuple                # bool per position
    k: np.ndarray              # (4,) us per unit position (0 for unused)
    err: np.ndarray            # (4,)
    mean: np.ndarray           # (4,) weighted mean positions
    std_before: float          # standard deviation of dt in the fit window before / after [us]
    std_after: float
    det_a: int = 0
    det_b: int = 0

    @property
    def t_values(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.k / self.err

    def text(self):
        parts = [f"{nm} {1000 * self.k[j]:+.0f} +- {1000 * self.err[j]:.0f} ns [{self.t_values[j]:+.1f}]"
                 for j, nm in enumerate(self.names) if self.used[j]]
        return (f"timing walk dt' = dt - k.(pos - mean), fitted in {self.window[0]:.2f}..{self.window[1]:.2f} us "
                f"({self.n:.0f} pairs): per unit position (x1000 ns, i.e. ns per full detector width)\n  "
                + ";  ".join(parts) + f"\n  std of dt in the fit window {1000 * self.std_before:.0f} -> "
                f"{1000 * self.std_after:.0f} ns")


def fit_walk(res, snap, det_a, det_b, n_sigma=2.0, use_neutral=False, side_e="a", background=True, encoder=None):
    """Weighted (accidentals subtracted) linear fit of dt on the hit positions inside peak +- n_sigma sigma.
    By default only the electron detector's x and y are used; use_neutral adds the other detector's.
    None if there is no peak or too little data."""
    if res is None or snap is None or len(res.dt_us) == 0:
        return None
    pk = res.peak()
    if pk is None:
        return None
    win = (pk[0] - n_sigma * pk[1], pk[0] + n_sigma * pk[1])
    p = weighted_pairs(res, snap, det_a, det_b, win, background, encoder)
    if p is None:
        return None
    X = np.column_stack([p["xa"], p["ya"], p["xb"], p["yb"]])
    used = np.array([True, True, True, True]) if use_neutral else (
        np.array([True, True, False, False]) if side_e == "a" else np.array([False, False, True, True]))
    ok = np.isfinite(X[:, used]).all(axis=1)
    w, tau, X = p["w"][ok], p["dt"][ok], X[ok]
    W = w.sum()
    if (w > 0).sum() < 50 or W < 50:
        return None
    mean = (w[:, None] * X).sum(0) / W
    mt = (w * tau).sum() / W
    Z = (X - mean)[:, used]
    A = Z.T @ (w[:, None] * Z)
    Ainv = np.linalg.pinv(A)
    k_u = Ainv @ (Z.T @ (w * (tau - mt)))
    r = tau - mt - Z @ k_u
    meat = Z.T @ (((w * r) ** 2)[:, None] * Z)
    e_u = np.sqrt(np.maximum(np.diag(Ainv @ meat @ Ainv), 0.0))
    k, err = np.zeros(4), np.full(4, np.nan)
    k[used], err[used] = k_u, e_u
    return TimingWalk(window=win, n=float(W), names=("x_a", "y_a", "x_b", "y_b"), used=tuple(bool(u) for u in used),
                      k=k, err=err, mean=mean, std_before=float(np.sqrt((w * (tau - mt) ** 2).sum() / W)),
                      std_after=float(np.sqrt((w * r * r).sum() / W)), det_a=det_a, det_b=det_b)


def apply_walk(res, snap, walk, encoder=None):
    """A copy of the CoincidenceResult with dt replaced by dt' for every pair that has the positions the walk
    uses (other pairs keep dt). Everything computed from the copy (peak, window, control windows, weights,
    profiles, fits) then lives on the dt' axis."""
    import dataclasses
    ok = _rows_ok(res, snap, walk.det_a, walk.det_b)
    if ok is None:
        return res
    idx = np.nonzero(ok)[0]
    pos = _positions(res, snap, walk.det_a, walk.det_b, idx, encoder or QuadEncoder())
    X = np.column_stack([pos["xa"], pos["ya"], pos["xb"], pos["yb"]])
    used = np.array(walk.used)
    fin = np.isfinite(X[:, used]).all(axis=1)
    dt = res.dt_us.copy()
    dt[idx[fin]] = res.dt_us[idx[fin]] - ((X[fin][:, used] - walk.mean[used]) @ walk.k[used])   # NaN of unused columns stays out
    return dataclasses.replace(res, dt_us=dt)


def position_correlation(res, snap, det_a, det_b, half_us=20.0, bin_us=0.5, centre=None,
                         max_points=20000, encoder=None, seed=0, correction=None):
    """{'a': PosCorr, 'b': PosCorr} from a CoincidenceResult and the SetSnapshot it was made from
    (or a later one: rows are append-only). The dt range is centre +- half_us, the centre being the
    coincidence peak unless given. With `correction` (a Correction) the electron detector's PosCorr also
    holds the means after the correction (corr_*); the uncorrected means are not touched.
    None if there is no result or a detector is missing."""
    if res is None or snap is None or len(res.dt_us) == 0:
        return None
    encoder = encoder or QuadEncoder()
    if centre is None:
        pk = res.peak()
        centre = pk[0] if pk is not None else 0.0
    lo, hi = float(centre) - half_us, float(centre) + half_us
    nb = int(min(4000, max(1, round((hi - lo) / max(bin_us, 1e-6)))))
    edges = np.linspace(lo, hi, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    g = _gather(res, snap, det_a, det_b, lo, hi, encoder)
    if g is None:
        return None
    dt = g["dt"]
    xy = {"a": (g["xa"], g["ya"]), "b": (g["xb"], g["yb"])}
    rng = np.random.default_rng(seed)
    out = {}
    for side, det in (("a", det_a), ("b", det_b)):
        x, y = xy[side]
        good = np.isfinite(x) & np.isfinite(y)
        dt_g, x_g, y_g = dt[good], x[good], y[good]
        idx = np.minimum(((dt_g - lo) / (hi - lo) * nb).astype(np.int64), nb - 1)
        n, mx, sx = _stats(idx, nb, x_g)
        _, my, sy = _stats(idx, nb, y_g)
        if len(dt_g) > max_points:
            pick = np.sort(rng.choice(len(dt_g), max_points, replace=False))
            raw = dt_g[pick], x_g[pick], y_g[pick]
        else:
            raw = dt_g, x_g, y_g
        extra = {}
        if correction is not None and correction.side_e == side:
            other = "b" if side == "a" else "a"
            xc, yc = correction.apply(x, y, *xy[other], dt)
            gc = np.isfinite(xc) & np.isfinite(yc)
            ic = np.minimum(((dt[gc] - lo) / (hi - lo) * nb).astype(np.int64), nb - 1)
            nc, cx, csx = _stats(ic, nb, xc[gc])
            _, cy, csy = _stats(ic, nb, yc[gc])
            extra = dict(corr_mean_x=cx, corr_mean_y=cy, corr_sem_x=csx, corr_sem_y=csy, corr_count=nc)
        out[side] = PosCorr(det=det, side=side, lo=lo, hi=hi, centers=centers, count=n,
                            mean_x=mx, mean_y=my, sem_x=sx, sem_y=sy,
                            raw_dt=raw[0], raw_x=raw[1], raw_y=raw[2],
                            n_pairs=g["n_pairs"], n_valid=int(good.sum()), **extra)
    return out


def cut_pairs(res, sel, sum_cut=True, time_cut=True):
    """Coincidence result restricted to pairs whose two events pass the Detectors-tab
    cuts of `sel` (a SelectionResult): the Sum (charge) limits actually in use and the
    time-since-start limits. Applied regardless of the Sum stage ('pre'/'post'): the
    position analysis wants clean events on both sides. Rows the selection has not
    seen yet are dropped. Returns (restricted result, n_before, n_after); `sel` None
    or no active cut returns `res` unchanged."""
    n0 = len(res.ia)
    if sel is None or res is None or not (sum_cut or time_cut):
        return res, n0, n0

    def ok(vals, lims, tlo, thi, n):
        k = min(n, len(vals[0]), len(vals[1]))        # unseen rows: not usable yet
        m = np.ones(k, bool)
        if sum_cut:
            m &= _in_range(vals[0][:k], *lims)
        if time_cut:
            m &= _in_range(vals[1][:k], tlo, thi)
        out = np.zeros(n, bool)
        out[:k] = m
        return out

    st = sel.settings
    oka = ok((sel.sum_a, sel.t_a), sel.sum_limits["a"], st.a.t_lo, st.a.t_hi, res.n_a)
    okb = ok((sel.sum_b, sel.t_b), sel.sum_limits["b"], st.b.t_lo, st.b.t_hi, res.n_b)
    keep = oka[res.ia] & okb[res.ib]
    return res.restrict(keep), n0, int(keep.sum())


def pair_displacements(res_cut, res_w, snap, det_a, det_b, cor, window, background=True, encoder=None):
    """Per-row displacement of the electron detector for a Correction `cor`: the correction applied to
    the pairs of the coincidence window (their electron row gets d = r_e - r_corr) and, with background=True,
    to the pairs of the control windows (dt moved back into the window, as in weighted_pairs). A row that is
    in a pair of the window itself takes that one; otherwise it takes the displacement of its control pair.

    The pairs are chosen with the raw dt of `res_cut` (the same windows that give the rows their
    accidental-subtraction weights); the dt term of the correction is evaluated at the (walk corrected) dt
    of `res_w`, which holds the same pairs in the same order. Returns (dx, dy, n_rows_set): arrays over the
    rows of the electron detector in the snapshot (0 = untouched)."""
    ok = _rows_ok(res_cut, snap, det_a, det_b)
    det_e = det_a if cor.side_e == "a" else det_b
    n = len(snap.detectors[det_e].columns["a"]) if det_e in snap.detectors else 0
    dx, dy = np.zeros(n), np.zeros(n)
    if ok is None or len(res_cut.dt_us) != len(res_w.dt_us):
        return dx, dy, 0
    lo, hi = float(window[0]), float(window[1])
    parts = [(lo, hi, 0.0)]
    if background:
        _, offs, _ = res_cut._bkg_offsets((lo, hi))
        parts += [(lo + o, hi + o, o) for o in offs]
    rows_e = res_cut.ia if cor.side_e == "a" else res_cut.ib
    encoder = encoder or QuadEncoder()
    n_set = 0
    for a, b, o in reversed(parts):                         # control windows first, the window itself last
        i = np.nonzero(ok & (res_cut.dt_us >= a) & (res_cut.dt_us <= b))[0]
        if not len(i):
            continue
        pos = _positions(res_cut, snap, det_a, det_b, i, encoder)
        xe, ye, xn, yn = _electron_neutral(pos, cor.side_e)
        xc, yc = cor.apply(xe, ye, xn, yn, res_w.dt_us[i] - o)
        ddx, ddy = xe - xc, ye - yc
        good = np.isfinite(ddx) & np.isfinite(ddy)
        r = rows_e[i][good]
        keep = r < n
        dx[r[keep]], dy[r[keep]] = ddx[good][keep], ddy[good][keep]
    n_set = int(np.count_nonzero((dx != 0) | (dy != 0)))
    return dx, dy, n_set
