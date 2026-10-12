"""Turn pipeline results into the arrays each plot needs (no Qt, fully testable).

The Qt widgets only draw what these functions return and forward user input to
the pipeline objects. Everything here is a pure function of immutable results.
"""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from analysis.abel import distribution_slice
from analysis.imaging import CartesianBinning, PolarBinning

Hist = namedtuple("Hist", "x y")            # bin centres, counts (or weights)
EMPTY = Hist(np.zeros(0), np.zeros(0))


# ---------------------------------------------------------- detector panel
@dataclass(frozen=True)
class DetectorPanel:
    det: int
    side: str                     # 'a' or 'b'
    image: np.ndarray             # [iy, ix] or None
    extent: tuple                 # (x0, x1, y0, y1) in image coordinates, or None
    levels: tuple                 # colour scale (0 .. ~99.8th percentile)
    time_all: Hist                # event time since the detector's first event [s]
    time_sel: Hist                #   ... of events kept by all cuts (background-subtracted)
    sum_all: Hist                 # Sum column
    sum_true: Hist                #   ... of true coincidence pairs only (independent of cuts)
    sum_sel: Hist                 #   ... of events kept by all cuts
    t_max: float
    sum_max: float
    n_valid: int
    n_selected: int
    cut_sum: tuple                # Sum limits actually applied (lo, hi), None = open
    cut_time: tuple
    centroid: tuple               # of the image, or None
    image_summary: str


def detector_panel(sel, imgres, det, bins_time=100, bins_sum=200):
    """Data for one detector's row on the Detectors tab. None until data exist."""
    if sel is None:
        return None
    side = "a" if det == sel.det_a else "b"
    valid = getattr(sel, f"valid_{side}")
    if not valid.any():
        return None
    t, s = getattr(sel, f"t_{side}")[valid], getattr(sel, f"sum_{side}")[valid]
    t_max = float(np.nanmax(t)) or 1.0
    s_max = float(np.nanmax(s)) or 1.0
    t_rng, s_rng = (0.0, t_max), (0.0, s_max)

    def hist(quantity, kind, bins, rng):
        c, h = sel.histogram(det, quantity, kind, bins=bins, range=rng)
        return Hist(c, h)

    cuts = getattr(sel.settings, side)
    img = imgres[det] if imgres is not None else None
    image = extent = centroid = None
    levels = (0.0, 1.0)
    if img is not None and isinstance(img.binning, CartesianBinning):
        image, extent, centroid = img.image, img.binning.extent, img.centroid()
        hi = float(np.percentile(image[image > 0], 99.8)) if (image > 0).any() else 1.0
        levels = (0.0, max(hi, 1e-9))
    lims = sel.sum_limits[side]
    return DetectorPanel(
        det=det, side=side, image=image, extent=extent, levels=levels,
        time_all=hist("time", "all", bins_time, t_rng), time_sel=hist("time", "selected", bins_time, t_rng),
        sum_all=hist("sum", "all", bins_sum, s_rng), sum_true=hist("sum", "true", bins_sum, s_rng),
        sum_sel=hist("sum", "selected", bins_sum, s_rng), t_max=t_max, sum_max=s_max,
        n_valid=int(valid.sum()), n_selected=int((getattr(sel, f"mask_{side}") & valid).sum()),
        cut_sum=tuple(lims), cut_time=(cuts.t_lo, cuts.t_hi), centroid=centroid,
        image_summary=img.summary() if img is not None else "")


# ------------------------------------------------------- coincidence panel
@dataclass(frozen=True)
class CoincidencePanel:
    status: str
    search_window_us: float
    window: tuple                 # (lo, hi) in effect, or None
    window_auto: bool
    peak: tuple                   # (centre, sigma) us, or None
    period_us: float              # measured period of the beam structure, or None
    full: Hist                    # dt over +-search window
    zoom: Hist                    # dt around the window
    zoom_range: tuple
    controls: tuple               # ((lo, hi, label), ...) control windows (shifted copies)
    bars: tuple                   # (labels, counts): main window and each control window
    acc: float                    # mean control count (accidental estimate) or None
    acc_fraction: float
    true_pairs: float
    method: str                   # 'shots' | 'flat' | ''
    rate_main: Hist               # pairs per second in the window vs time
    rate_acc: Hist                # mean of the control windows, same binning
    clock_t: np.ndarray           # matched pulses: time [s] ...
    clock_residual_us: np.ndarray #   ... clock offset minus its linear fit [us]
    drift_ppm: float              # relative clock rate difference
    n_pairs: int
    n_matched: int
    text: str


def _hist(values, lo, hi, bins):
    h, e = np.histogram(values, bins=bins, range=(lo, hi))
    return Hist(0.5 * (e[1:] + e[:-1]), h.astype(float))


def coincidence_panel(res, n_full_bins=800, n_zoom_bins=120, rate_bin_s=None):
    """All arrays for the Coincidence tab from a CoincidenceResult. None if no result."""
    if res is None:
        return None
    W = res.search_window_us
    dt = res.dt_us
    w = res.window_us
    pk = res.peak() if len(dt) >= 50 else None
    full = _hist(dt, -W, W, n_full_bins) if len(dt) else EMPTY

    # zoom: the window plus generous margins (or the peak / a default span)
    if w is not None:
        half = max(1.5 * (w[1] - w[0]), 6.0)
        c = 0.5 * (w[0] + w[1])
    elif pk is not None:
        half, c = max(8 * pk[1], 6.0), pk[0]
    else:
        half, c = min(50.0, W), 0.0
    zoom_range = (c - half, c + half)
    zoom = _hist(dt, *zoom_range, n_zoom_bins) if len(dt) else EMPTY

    controls, bars, acc, acc_fraction, true_pairs, method, period = (), ((), ()), None, None, None, "", None
    rate_main = rate_acc = EMPTY
    if w is not None and len(dt):
        st = res.window_stats(w)
        method, period = st["method"], st["shot_period_us"]
        _, offs, labels = res._bkg_offsets(w)
        controls = tuple((w[0] + o, w[1] + o, str(l)) for o, l in zip(offs, labels))
        names = ["window"] + [f"{l:+d}" if isinstance(l, int) else f"{l}us" for l in labels]
        counts = [st["pairs"]] + [st["sideband_counts"][l] for l in labels]
        bars = (tuple(names), tuple(counts))
        acc, acc_fraction, true_pairs = st["acc"], st["acc_fraction"], st["true_pairs"]
        # stationarity: pair rate in the window vs the control windows, over time
        if len(res.ta):
            t0 = float(res.ta.min())
            span = float(res.ta.max()) - t0
            bw = rate_bin_s or max(1.0, span / 200.0)
            nb = max(int(np.ceil(span / bw)), 1)
            rng = (0.0, nb * bw)
            sel = (dt >= w[0]) & (dt <= w[1])
            rate_main = _hist(res.ta[sel] - t0, *rng, nb)
            rate_main = Hist(rate_main.x, rate_main.y / bw)
            if offs:
                accum = np.zeros(nb)
                for o in offs:
                    s = (dt >= w[0] + o) & (dt <= w[1] + o)
                    accum += np.histogram(res.ta[s] - t0, bins=nb, range=rng)[0]
                rate_acc = Hist(rate_main.x, accum / len(offs) / bw)

    clock_t, clock_res, drift = np.zeros(0), np.zeros(0), None
    if res.pulses_a is not None and len(res.pulses_a) >= 3:
        pa, pb = res.pulses_a, res.pulses_b
        clock_t = pa - pa[0]
        off_us = ((pa - pb) - (pa[0] - pb[0])) * 1e6
        slope, icpt = np.polyfit(clock_t, off_us, 1)
        clock_res = off_us - (slope * clock_t + icpt)
        drift = float(slope)                      # us per s  ==  ppm

    return CoincidencePanel(
        status=res.status, search_window_us=W, window=w, window_auto=res.window_auto, peak=pk,
        period_us=period, full=full, zoom=zoom, zoom_range=zoom_range, controls=controls,
        bars=bars, acc=acc, acc_fraction=acc_fraction, true_pairs=true_pairs, method=method,
        rate_main=rate_main, rate_acc=rate_acc, clock_t=clock_t, clock_residual_us=clock_res,
        drift_ppm=drift, n_pairs=len(dt), n_matched=res.n_matched, text=res.summary())


# ------------------------------------------------------------------ cuts
def apply_cut(sel, det, kind, enabled, lo, hi):
    """Push one draggable-line cut to the SelectionFilter.
    kind: 'sum' or 'time'. enabled=False clears the cut."""
    lo, hi = (min(lo, hi), max(lo, hi)) if enabled else (None, None)
    if kind == "sum":
        sel.set_sum(det, lo, hi)
    elif kind == "time":
        sel.set_time(det, lo, hi)
    else:
        raise ValueError(kind)


# ------------------------------------------------------------ polar / centering tab
def polar_to_cartesian(H, binning, size=256, per_area=True):
    """Resample a polar array H[itheta, ir] onto a Cartesian grid for display.
    per_area=True: H holds counts per bin, shown as counts per unit area (density), so that a ring
    does not look brighter at large r. per_area=False: H already is a density (e.g. a slice of the
    reconstructed distribution) and is only resampled.
    Returns (image[iy, ix], extent (-r_max, r_max, -r_max, r_max))."""
    nth, nr = H.shape
    rmax = binning.r_max
    r_edges = binning.r_edges()
    area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2) * (2 * np.pi / nth)
    dens = H / area[None, :] if per_area else H
    c = (np.arange(size) + 0.5) / size * 2 * rmax - rmax
    X, Y = np.meshgrid(c, c)
    r = np.hypot(X, Y)
    th = np.mod(np.arctan2(Y, X), 2 * np.pi)
    ok = r < rmax
    ir = np.minimum((r / rmax * nr).astype(int), nr - 1)
    it = np.minimum((th / (2 * np.pi) * nth).astype(int), nth - 1)
    img = np.where(ok, dens[it, ir], 0.0)
    return img, (-rmax, rmax, -rmax, rmax)


@dataclass(frozen=True)
class PolarPanel:
    det: int
    side: str
    view: str                     # 'polar' or 'cartesian'
    image: np.ndarray
    extent: tuple
    levels: tuple
    aspect_locked: bool
    xlabel: str
    ylabel: str
    title: str
    r: np.ndarray                 # radial bin centres
    prof_raw: Hist                # counts per radial bin summed over angle, centred only
    prof_corr: Hist               #   ... centred + circularised
    center: tuple
    center_auto: tuple            # latest automatic estimate or None
    center_mode: str
    cx_proj: Hist                 # projection on x vs distance from the centre, and its mirror image
    cx_mirror: Hist
    cy_proj: Hist
    cy_mirror: Hist
    circ_theta_deg: np.ndarray    # s(theta) of the table in use (empty if none)
    circ_s: np.ndarray
    circ_similarity: np.ndarray
    mirror_planes_deg: tuple      # angles of the four half-planes
    circ_applied: bool
    circ_info: str
    warnings: tuple


def _mirrored(x, p, c):
    """(projection, its mirror image) as functions of u = x - c, both scaled to max 1."""
    m = p.max() if p.max() > 0 else 1.0
    u = x - c
    return Hist(u, p / m), Hist(u, np.interp(2 * c - x, x, p, left=0.0, right=0.0) / m)


def polar_panel(res, side, view="polar", cart_size=256):
    """Everything the centering / circularisation tab draws. None until data exist."""
    if res is None:
        return None
    v = res[side]
    H = v.corr
    if not np.isfinite(H).all() or H.size == 0:
        return None
    b = v.binning
    if view == "cartesian":
        image, extent = polar_to_cartesian(H, b, cart_size)
        aspect, xl, yl = True, "x - centre", "y - centre"
    else:
        image, extent = H, b.extent
        aspect, xl, yl = False, "radius r", "angle (deg)"
    pos = image[image > 0]
    levels = (0.0, float(np.percentile(pos, 99.8)) if pos.size else 1.0)
    rc = 0.5 * (b.r_edges()[1:] + b.r_edges()[:-1])
    d = v.diag
    empty = (EMPTY, EMPTY)
    cxp, cxm = _mirrored(d.x, d.px, d.cx) if d is not None else empty
    cyp, cym = _mirrored(d.y, d.py, d.cy) if d is not None else empty
    t = v.circ
    if t is not None:
        th, s, sim = np.degrees(np.asarray(t.theta)), np.asarray(t.s), np.asarray(t.similarity)
        planes = tuple((t.theta_ref_deg + 90.0 * i) % 360.0 for i in range(4))
        info = t.describe()
    else:
        th = s = sim = np.zeros(0)
        planes, info = (), "no circularisation: rings are only centred"
    return PolarPanel(
        det=v.det, side=side, view=view, image=image, extent=extent, levels=levels,
        aspect_locked=aspect, xlabel=xl, ylabel=yl,
        title="centred + circularised" if v.circ_applied else "centred (not circularised)",
        r=rc, prof_raw=Hist(rc, v.raw.sum(axis=0)), prof_corr=Hist(rc, H.sum(axis=0)),
        center=v.center, center_auto=v.center_auto, center_mode=v.center_mode,
        cx_proj=cxp, cx_mirror=cxm, cy_proj=cyp, cy_mirror=cym,
        circ_theta_deg=th, circ_s=s, circ_similarity=sim, mirror_planes_deg=planes,
        circ_applied=v.circ_applied, circ_info=info, warnings=v.warnings)


# ------------------------------------------------------------------ Abel inversion
@dataclass(frozen=True)
class AbelPanel:
    det: int
    side: str
    status: str
    orders: tuple
    rho: np.ndarray
    bins_after: tuple             # ((n, Hist of d_n(r)), ...) centred + circularised
    bins_before: tuple            # same for the centred data; None if no circularisation is applied
    intensity: Hist               # I(rho), None while the matrices are being built
    betas: tuple                  # ((n, Hist of beta_n(rho)), ...)
    slice_image: np.ndarray       # reconstructed image: the distribution in the plane of the laser axis
    slice_extent: tuple
    slice_levels: tuple
    slice_aspect: bool
    slice_xlabel: str
    slice_ylabel: str
    text: str
    intensity_band: tuple = None  # (x, lower, upper) bias-corrected bootstrap band, or None
    beta_bands: tuple = ()        # ((n, (x, lower, upper)), ...)
    boot_text: str = ""           # one line for the controls


def _boot_panel(a, boot):
    """(intensity band, beta bands, one-line status, extra summary lines) of a bootstrap result for
    the current Abel result `a`. The bands are only drawn when the bootstrap belongs to the current
    settings; otherwise the summary says so."""
    if boot is None:
        return None, (), "no bootstrap yet", []
    fresh = boot.signature == a.signature
    state = ("running" if boot.running else "stopped" if boot.stopped else
             "failed" if boot.error else "done")
    line = f"{state}: {boot.n_done}/{boot.n_total} samples, {boot.seconds:.1f} s"
    if not fresh:
        line += "; belongs to other settings (not drawn)"
    elif boot.n_events != a.n_events and boot.n_events:
        line += f"; made with {boot.n_events} events, now {a.n_events}"
    if boot.error:
        line += f" ({boot.error})"
    at = int(np.argmax(a.abel.intensity)) if a.abel is not None else None
    extra = ["", boot.summary(at)] if at is not None else ["", boot.summary()]
    if not fresh or boot.intensity is None:
        return None, (), line, extra
    ib = (boot.rho, boot.intensity.lower, boot.intensity.upper)
    bb = ()
    if boot.beta is not None:
        bb = tuple((n, (boot.rho, boot.beta.lower[j], boot.beta.upper[j]))
                   for j, n in enumerate(boot.beta_orders))
    return ib, bb, line, extra


def abel_panel(ares, side, binning, view="polar", cart_size=256, boot=None):
    """Everything the Abel page draws. binning: the PolarBinning of the side (for the angular grid
    of the reconstructed image). None until the Abel stage has published."""
    if ares is None:
        return None
    a = ares[side]
    bins_after = tuple((n, Hist(a.rho, a.d[i])) for i, n in enumerate(a.orders))
    bins_before = (None if a.d_centred is None
                   else tuple((n, Hist(a.rho, a.d_centred[i])) for i, n in enumerate(a.orders)))
    intensity, betas, image, extent, levels = None, (), None, None, (0.0, 1.0)
    aspect, xl, yl = False, "radius rho", "angle (deg)"
    lines = [f"detector {a.det} ({side.upper()}): {a.status}",
             f"angular orders {list(a.orders)}, {a.n_bins} radial bins of {a.dr:.5g}, laser axis "
             f"{np.degrees(a.axis_angle):.1f} deg",
             f"events with weight: {a.n_events}, total weight in range: {a.total_weight:.1f}"]
    if a.abel is not None:
        ab = a.abel
        intensity = Hist(a.rho, ab.intensity)
        betas = tuple((n, Hist(a.rho, ab.beta[j])) for j, n in enumerate(ab.beta_orders))
        ntheta = binning.ntheta
        theta = (np.arange(ntheta) + 0.5) * 2 * np.pi / ntheta
        f = distribution_slice(ab.I_n, a.orders, a.dr, (np.arange(a.n_bins) + 0.5) * a.dr, theta, a.axis_angle)
        grid = PolarBinning(a.n_bins, ntheta, a.n_bins * a.dr)
        if view == "cartesian":
            image, extent = polar_to_cartesian(f, grid, cart_size, per_area=False)
            aspect, xl, yl = True, "x - centre", "y - centre"
        else:
            image, extent = f, grid.extent
        pos = image[image > 0]
        levels = (0.0, float(np.percentile(pos, 99.8)) if pos.size else 1.0)
        i = int(np.argmax(ab.intensity))
        lines.append(f"integral of I(rho): {ab.total():.1f};  maximum at rho = {a.rho[i]:.4g}")
        if a.n_events:
            lines.append("beta at the maximum: " + ", ".join(
                f"b{n} = {ab.beta[j, i]:.3f}" for j, n in enumerate(ab.beta_orders)))
        lines.append(f"regularisation: {'none (exact inverse)' if not np.any(a.basis.tikhonov) else a.basis.tikhonov.max()}")
        if a.basis.joint is not None:
            lines.append("loss terms: " + a.basis.loss_text())
    ib, bb, boot_line, extra = _boot_panel(a, boot)
    lines += extra
    return AbelPanel(det=a.det, side=side, status=a.status, orders=a.orders, rho=a.rho,
                     bins_after=bins_after, bins_before=bins_before, intensity=intensity, betas=betas,
                     slice_image=image, slice_extent=extent, slice_levels=levels, slice_aspect=aspect,
                     slice_xlabel=xl, slice_ylabel=yl, text="\n".join(lines),
                     intensity_band=ib, beta_bands=bb, boot_text=boot_line)
