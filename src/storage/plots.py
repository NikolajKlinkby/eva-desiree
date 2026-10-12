"""The data of every plot of the GUI, with names (Qt-free).

build(session, gui) returns a nested dict that goes under /plots in the analysis file. One group per plot; the
arrays in it are named after what they are (`dt_us`, `pairs`, `radius_rho`, `intensity`, `lower`, `upper`, ...)
and the attributes say what the plot is: `title`, `xlabel`, `ylabel`, `gui_location`. The numbers are the very
arrays the GUI draws (they come from the same functions, gui/plotdata.py and analysis/poscorr.py).

  /plots/detectors/detector_<n>_<A|B>/   image, event_time, charge_sum, cuts           Detectors tab
  /plots/coincidence/timing/             dt_full, dt_zoom, window_vs_control, pair_rate,
                                         clock_offset_residual, summary                 Coincidence tab, Timing
  /plots/coincidence/position_correlation/<A|B>_detector_<n>/                           Coincidence tab, Position Correlation
                                         raw_pairs, binned_means, fitted_dt_shape
  /plots/coincidence/position_correlation/electron_images/   uncorrected / corrected, whole window and slices
  /plots/analysis/<A|B>_detector_<n>/    centring/{x_projection, y_projection, centre}, image_centred_only,
                                         image_centred_circularised, radial_profile, circularisation,
                                         radial_bins_centred, radial_bins_centred_circularised,
                                         abel/{intensity, beta_<n>, reconstructed_image, summary}   Analysis tab

An image is `counts` (or `density`), with its axes next to it: `x_centres`, `y_centres` (Cartesian, [y][x]) or
`radius` and `angle_deg` (polar, [angle][radius]).
"""
import numpy as np

from gui import plotdata
from analysis.imaging import CartesianBinning

WHERE_DET = "Detectors tab"
WHERE_TIM = "Coincidence tab, page Timing"
WHERE_POS = "Coincidence tab, page Position Correlation"
WHERE_CEN = "Analysis tab, page centring and circularisation"
WHERE_ABL = "Analysis tab, page Abel inversion"


def _plot(title, xlabel, ylabel, where, **arrays):
    d = {"title": title, "xlabel": xlabel, "ylabel": ylabel, "gui_location": where}
    for k, v in arrays.items():
        a = None if v is None else np.asarray(v)
        d[k] = None if a is not None and a.size == 0 else a          # no empty arrays: the plot has no such curve
    return d


def _image(title, xlabel, ylabel, where, data, name="counts", **axes):
    d = {"title": title, "xlabel": xlabel, "ylabel": ylabel, "gui_location": where, name: np.asarray(data)}
    d.update({k: np.asarray(v) for k, v in axes.items()})
    return d


def _centres(extent, shape):
    x0, x1, y0, y1 = extent
    ny, nx = shape
    return x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx, y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny


def _guard(out, name, fn, *a, **k):
    """Run one section; a failure is written into the file instead of stopping the whole save."""
    try:
        r = fn(*a, **k)
        if r:
            out[name] = r
    except Exception as exc:
        out[name] = {"error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------ Detectors tab
def detectors(sel, img):
    out = {}
    if sel is None:
        return out
    for det, label in ((sel.det_a, "A"), (sel.det_b, "B")):
        p = plotdata.detector_panel(sel, img, det)
        if p is None:
            continue
        g = {}
        if p.image is not None:
            x, y = _centres(p.extent, p.image.shape)
            g["image"] = _image(f"detector {det} ({label}): image", "x", "y", WHERE_DET, p.image, x_centres=x, y_centres=y)
            g["image"]["centroid"] = None if p.centroid is None else np.array(p.centroid)
        g["event_time"] = _plot(f"detector {det} ({label}): event time", "time since first event (s)", "events", WHERE_DET,
                                time_s=p.time_all.x, all_events=p.time_all.y, kept=p.time_sel.y)
        g["charge_sum"] = _plot(f"detector {det} ({label}): charge sum", "Sum", "events", WHERE_DET,
                                sum=p.sum_all.x, all_events=p.sum_all.y, true_pairs=p.sum_true.y, kept=p.sum_sel.y)
        g["cuts"] = {"sum_low": p.cut_sum[0], "sum_high": p.cut_sum[1], "time_low_s": p.cut_time[0],
                     "time_high_s": p.cut_time[1], "events_valid": p.n_valid, "events_kept": p.n_selected,
                     "image_summary": p.image_summary}
        out[f"detector_{det}_{label}"] = g
    return out


# ------------------------------------------------------------------ Coincidence tab: Timing
def timing(res):
    if res is None:
        return {}
    c = plotdata.coincidence_panel(res)
    w = None if c.window is None else np.array(c.window)
    g = {"dt_full": _plot("dt, full search window", "dt = t_B - t_A (us)", "pairs per bin", WHERE_TIM,
                          dt_us=c.full.x, pairs=c.full.y, window_us=w),
         "dt_zoom": _plot("dt, zoom", "dt (us)", "pairs per bin", WHERE_TIM, dt_us=c.zoom.x, pairs=c.zoom.y,
                          window_us=w, control_windows_us=np.array([r[:2] for r in c.controls]) if c.controls else None)}
    names, counts = c.bars
    g["window_vs_control"] = _plot("window vs control windows", "", "pairs", WHERE_TIM, counts=np.array(counts, float))
    g["window_vs_control"]["names"] = list(names)
    g["window_vs_control"]["accidentals_mean"] = c.acc
    g["pair_rate"] = _plot("pair rate (stationarity check)", "time (s)", "pairs / s", WHERE_TIM, time_s=c.rate_main.x,
                           window=c.rate_main.y, control_windows_mean=c.rate_acc.y)
    g["clock_offset_residual"] = _plot("clock offset B-A, residual to linear drift", "time (s)", "us", WHERE_TIM,
                                       time_s=c.clock_t, residual_us=c.clock_residual_us)
    g["clock_offset_residual"]["drift_ppm"] = c.drift_ppm
    g["summary"] = {"text": c.text, "status": c.status, "pairs": c.n_pairs, "matched_pulses": c.n_matched,
                    "window_us": w, "window_automatic": c.window_auto, "search_window_us": c.search_window_us,
                    "peak_centre_sigma_us": None if c.peak is None else np.array(c.peak),
                    "period_us": c.period_us, "accidentals_mean": c.acc, "accidentals_fraction": c.acc_fraction,
                    "true_pairs": c.true_pairs, "method": c.method}
    return g


# ------------------------------------------------------------------ Coincidence tab: Position Correlation
def _gui(gui, key, default):
    v = (gui or {}).get(f"tab_coin.{key}")
    return default if not v else v.get("value", default)


def position_correlation(session, gui):
    from analysis.poscorr import (apply_walk, correction_images, cut_pairs, dt_profile, fit_correction, fit_walk,
                                      position_correlation as pcorr)
    s = session
    ps = s.pos.settings
    res, sel = s.coincidence(), s.selection()
    if res is None or res.window_us is None or not (ps.enabled or _gui(gui, "pc_enable", False)):
        return {}
    half, binw, nsl = _gui(gui, "pc_half", 20.0), _gui(gui, "pc_bin", 0.5), int(_gui(gui, "pc_slices", 1))
    snap = s.ds.snapshot()
    da, db = s.flt.det_a, s.flt.det_b
    out = {"parameters": {"electron_side": ps.side_e, "use_neutral_position": ps.use_neutral, "dt_polynomial_order": ps.tau_order,
                          "accidentals_subtracted": ps.background, "walk_correction": ps.walk, "apply_cuts": ps.cuts,
                          "dt_range_half_us": half, "bin_width_us": binw, "n_slices": nsl}}
    if ps.cuts and sel is not None:
        res = cut_pairs(res, sel)[0]
    walk = fit_walk(res, snap, da, db, n_sigma=ps.walk_sigma, use_neutral=ps.walk_neutral, side_e=ps.side_e) if ps.walk else None
    if walk is not None:
        res = apply_walk(res, snap, walk)
    cor = fit_correction(res, snap, da, db, side_e=ps.side_e, use_neutral=ps.use_neutral, tau_order=ps.tau_order,
                         background=ps.background)
    pc = pcorr(res, snap, da, db, half_us=half, bin_us=binw, correction=cor)
    prof = dt_profile(res, snap, da, db, half_us=half, bin_us=binw, correction=cor) if ps.background else None
    xl = "dt' = dt - walk(position) (us)" if walk is not None else "dt = t_B - t_A (us)"
    for side, label in (("a", "A"), ("b", "B")):
        r = pc[side] if pc else None
        if r is None:
            continue
        g = {"raw_pairs": _plot(f"detector {r.det} ({label}): hit position of individual pairs (subsample)", xl,
                                "x, y", WHERE_POS, dt_us=r.raw_dt, x=r.raw_x, y=r.raw_y)}
        q = prof[side] if prof else None
        if q is not None:
            bm = _plot(f"detector {r.det} ({label}): mean hit position vs dt, accidentals subtracted", xl, "x, y", WHERE_POS,
                       dt_us=q.centers, x=q.mean_x, y=q.mean_y, x_error=q.err_x, y_error=q.err_y)
            if q.corr_mean_x is not None:
                bm.update(x_corrected=np.asarray(q.corr_mean_x), y_corrected=np.asarray(q.corr_mean_y),
                          x_corrected_error=np.asarray(q.corr_err_x), y_corrected_error=np.asarray(q.corr_err_y))
        else:
            bm = _plot(f"detector {r.det} ({label}): mean hit position vs dt", xl, "x, y", WHERE_POS, dt_us=r.centers,
                       x=r.mean_x, y=r.mean_y, x_error=r.sem_x, y_error=r.sem_y, pairs=r.count)
            if r.corr_mean_x is not None:
                bm.update(x_corrected=np.asarray(r.corr_mean_x), y_corrected=np.asarray(r.corr_mean_y),
                          x_corrected_error=np.asarray(r.corr_sem_x), y_corrected_error=np.asarray(r.corr_sem_y))
        g["binned_means"] = bm
        if cor is not None and cor.side_e == side and cor.tau_order >= 1:
            tt = np.linspace(cor.window[0], cor.window[1], 120)
            xs, ys = cor.shape(tt)
            g["fitted_dt_shape"] = _plot(f"detector {r.det} ({label}): fitted dt shape", xl, "x, y", WHERE_POS,
                                         dt_us=tt, x=xs, y=ys)
        g["pairs_in_range"] = int(r.n_valid)
        out[f"{label}_detector_{r.det}"] = g
    if cor is not None:
        img = s.imaging()
        b = img[cor.det_e].binning if img is not None else None
        b = b if isinstance(b, CartesianBinning) else CartesianBinning(256, 256)
        ci = correction_images(res, snap, da, db, cor, b, n_slices=nsl)
        ei = {"fit_text": cor.text() + (ci.text() if ci is not None else "") + (("\n\n" + walk.text()) if walk else "")}
        if ci is not None:
            for name, arr in (("uncorrected_full_window", ci.raw), ("corrected_full_window", ci.corrected)):
                x, y = _centres(ci.extent, arr.shape)
                ei[name] = _image(f"electron image, detector {cor.det_e}: {name.replace('_', ' ')}", "x", "y", WHERE_POS,
                                  arr, name="weights", x_centres=x, y_centres=y)
            for i, sl in enumerate(ci.slices, 1):
                x, y = _centres(ci.extent, sl.image.shape)
                ei[f"slice_{i}"] = {
                    "dt_low_us": sl.lo, "dt_high_us": sl.hi, "true_pairs": sl.n_true, "pairs": sl.n_pairs,
                    "mean_corrected": np.asarray(sl.mean_corr), "mean_uncorrected": np.asarray(sl.mean_raw),
                    "corrected": _image(f"slice {i}: corrected electron image", "x", "y", WHERE_POS, sl.image,
                                        name="weights", x_centres=x, y_centres=y),
                    "uncorrected": _image(f"slice {i}: uncorrected electron image", "x", "y", WHERE_POS, sl.image_raw,
                                          name="weights", x_centres=x, y_centres=y)}
        out["electron_images"] = ei
    return out


# ------------------------------------------------------------------ Analysis tab
def _hists(prefix, pairs, rho):
    d = {"radius": np.asarray(rho)}
    for n, h in pairs:
        d[f"{prefix}{n}"] = np.asarray(h.y)
    return d


def analysis(pol, ab, boot_of):
    out = {}
    if pol is None:
        return out
    for side, label in (("a", "A"), ("b", "B")):
        p = plotdata.polar_panel(pol, side, "polar")
        if p is None:
            continue
        v = pol[side]
        b = v.binning
        g = {}
        # --- page centring and circularisation
        cen = {"x": p.center[0], "y": p.center[1], "mode": p.center_mode,
               "automatic_estimate": None if p.center_auto is None else np.array(p.center_auto)}
        cen["x_projection"] = _plot("centring: projection on x and its mirror image", "distance from the centre",
                                    "projection (max = 1)", WHERE_CEN, distance=p.cx_proj.x, projection=p.cx_proj.y,
                                    mirrored=p.cx_mirror.y)
        cen["y_projection"] = _plot("centring: projection on y and its mirror image", "distance from the centre",
                                    "projection (max = 1)", WHERE_CEN, distance=p.cy_proj.x, projection=p.cy_proj.y,
                                    mirrored=p.cy_mirror.y)
        g["centring"] = cen
        ang = np.degrees(0.5 * (b.theta_edges()[1:] + b.theta_edges()[:-1]))
        g["image_centred_only"] = _image(f"detector {v.det}: centred", "radius r", "angle (deg)", WHERE_CEN, v.raw,
                                         radius=p.r, angle_deg=ang)
        g["image_centred_circularised"] = _image(f"detector {v.det}: {p.title}", "radius r", "angle (deg)",
                                                 f"{WHERE_CEN}; {WHERE_ABL} (measured image)", v.corr, radius=p.r,
                                                 angle_deg=ang)
        cimg, ext = plotdata.polar_to_cartesian(v.corr, b, 256)
        cx, cy = _centres(ext, cimg.shape)
        g["image_centred_circularised"]["density_cartesian"] = cimg
        g["image_centred_circularised"]["x_centres"], g["image_centred_circularised"]["y_centres"] = cx, cy
        g["radial_profile"] = _plot("angle-integrated radial profile", "radius r", "counts per radial bin", WHERE_CEN,
                                    radius=p.r, centred=p.prof_raw.y, centred_circularised=p.prof_corr.y)
        circ = _plot("circularisation: stretch s(angle)", "angle (deg)", "s", WHERE_CEN, angle_deg=p.circ_theta_deg,
                     stretch_s=p.circ_s, similarity=p.circ_similarity)
        circ["mirror_planes_deg"] = np.array(p.mirror_planes_deg)
        circ["applied"] = p.circ_applied
        circ["info"] = p.circ_info
        g["circularisation"] = circ
        # --- page Abel inversion
        if ab is not None:
            ap = plotdata.abel_panel(ab, side, b, "polar", boot=boot_of(side))
            if ap is not None:
                orders = list(ap.orders)
                g["radial_bins_centred_circularised"] = _plot(
                    "radial bins d_n(r), centred + circularised", "radius r", "d_n(r)", f"{WHERE_CEN}; {WHERE_ABL}",
                    **{"radius": ap.rho, **{f"order_{n}": h.y for n, h in ap.bins_after}})
                if ap.bins_before is not None:
                    g["radial_bins_centred"] = _plot("radial bins d_n(r), centred only", "radius r", "d_n(r)", WHERE_CEN,
                                                     **{"radius": ap.rho, **{f"order_{n}": h.y for n, h in ap.bins_before}})
                a = {"summary": {"text": ap.text, "status": ap.status, "orders": np.array(orders),
                                 "bootstrap_status": ap.boot_text}}
                if ap.intensity is not None:
                    ib = ap.intensity_band
                    a["intensity"] = _plot("intensity I(rho)", "radius rho", "counts per unit radius", WHERE_ABL,
                                           radius_rho=ap.rho, intensity=ap.intensity.y,
                                           lower=None if ib is None else ib[1], upper=None if ib is None else ib[2])
                    bb = dict(ap.beta_bands)
                    for n, h in ap.betas:
                        bd = bb.get(n)
                        a[f"beta_{n}"] = _plot(f"anisotropy beta_{n}(rho)", "radius rho", f"beta_{n}", WHERE_ABL,
                                               radius_rho=ap.rho, beta=h.y, lower=None if bd is None else bd[1],
                                               upper=None if bd is None else bd[2])
                    rho = (np.arange(len(ap.rho)) + 0.5) * (ap.rho[1] - ap.rho[0] if len(ap.rho) > 1 else 1.0)
                    f = ap.slice_image
                    rec = _image("reconstructed image: distribution in the plane of the laser axis", "radius rho",
                                 "angle (deg)", WHERE_ABL, f, name="density", radius=rho,
                                 angle_deg=np.degrees((np.arange(f.shape[0]) + 0.5) * 2 * np.pi / f.shape[0]))
                    a["reconstructed_image"] = rec
                g["abel"] = a
        out[f"{label}_detector_{v.det}"] = g
    return out


# ------------------------------------------------------------------ everything
def build(session, gui=None):
    s = session
    out = {}
    _guard(out, "detectors", detectors, s.selection(), s.imaging())
    coin = {}
    _guard(coin, "timing", timing, s.coincidence())
    _guard(coin, "position_correlation", position_correlation, s, gui)
    if coin:
        out["coincidence"] = coin
    _guard(out, "analysis", analysis, s.polar_result(), s.abel_result(), lambda side: s.abel.bootstrap_result(side))
    return out


def index(plots, prefix=""):
    """Lines 'path: title  [where]' of every plot (a group with a title) in a nested plots dict."""
    lines = []
    for k, v in plots.items():
        if not isinstance(v, dict):
            continue
        path = f"/plots/{prefix}{k}"
        if "title" in v and "gui_location" in v:
            lines.append(f"{path}: {v['title']}  [{v['gui_location']}]")
        else:
            lines += index(v, f"{prefix}{k}/")
    return lines
