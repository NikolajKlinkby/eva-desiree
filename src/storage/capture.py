"""Collect the state of a running Session for the analysis file, and put saved settings back (Qt-free)."""
import dataclasses
import platform
import subprocess
import sys

from paths import ROOT
from storage import h5file, plots


# ------------------------------------------------------------------ settings
def coincidence_settings(flt):
    return {"search_window_us": flt.W * 1e6, "shot_period_us": flt.shot_period_us, "bkg_shots": tuple(flt.bkg_shots),
            "window_us": flt._window,                     # None = automatic
            "auto_n_sigma": flt.auto_n_sigma, "cycle_shift": flt.cycle_shift,
            "cycle_shift_found": flt.cycle_shift_found, "match_tol": flt.match_tol, "lag_margin": flt.lag_margin,
            "max_shift": flt.max_shift}


def stage_settings(session):
    s = session
    return {"coincidence": coincidence_settings(s.flt), "selection": s.sel.settings, "poscorr": s.pos.settings,
            "imager": s.img.settings, "polar": s.polar.settings, "abel": s.abel.settings,
            "abel_bootstrap": {k: s.abel.bootstrap_settings(k) for k in ("a", "b")}}


def _set(stage, settings):
    """Replace the settings object of a stage (what its set_* methods do, for all fields at once)."""
    with stage._lock:
        stage._settings = dataclasses.replace(settings, version=stage._settings.version + 1)


def apply_settings(session, saved, running):
    """Put saved settings (the "settings" of read_meta) into the stages of a session.

    running=False: before Session.start(); also applies what needs a stopped coincidence filter (search window,
    forced cycle shift). running=True: afterwards, on the live stages (everything else). Missing entries are skipped."""
    s = session
    c = saved.get("coincidence")
    if c and s.flt is not None:
        if not running:
            s.flt.reconfigure(search_window_us=c.get("search_window_us"), cycle_shift=c.get("cycle_shift"))
        s.flt.set_shot_period(c["shot_period_us"])
        s.flt.set_bkg_shots(c["bkg_shots"])
        s.flt.set_auto_sigma(c["auto_n_sigma"])
        if c.get("window_us") is None:
            s.flt.clear_window()
        else:
            s.flt.set_window(*c["window_us"])
    for name, stage in (("selection", s.sel), ("poscorr", s.pos), ("imager", s.img), ("polar", s.polar),
                        ("abel", s.abel)):
        obj = saved.get(name)
        if obj is not None and stage is not None and dataclasses.is_dataclass(obj):
            _set(stage, obj)
    pol = saved.get("polar")
    if pol is not None and s.polar is not None and dataclasses.is_dataclass(pol):
        for k in ("a", "b"):
            circ = getattr(pol, k).circ
            if circ is not None:
                s.polar._status[k] = circ.describe()
    for k, cfg in (saved.get("abel_bootstrap") or {}).items():
        if s.abel is not None and dataclasses.is_dataclass(cfg):
            s.abel._boot_cfg[k] = cfg


class LoadedBootstrap:
    """A bootstrap result read from a file, in the place of a Bootstrap run (so that the bands are shown)."""
    running = False

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def stop(self):
        pass

    def join(self, timeout=None):
        pass


def restore_bootstrap(session, bootstrap):
    for k, r in (bootstrap or {}).items():
        if r is not None and hasattr(r, "n_done") and session.abel is not None:
            session.abel._boot[k] = LoadedBootstrap(r)


# ------------------------------------------------------------------ the file content
def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "h5py", "PyQt6.QtCore", "pyqtgraph"):
        try:
            m = __import__(mod, fromlist=["x"])
            out[mod.split(".")[0]] = str(getattr(m, "__version__", getattr(m, "PYQT_VERSION_STR", "?")))
        except Exception:
            pass
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                           text=True, timeout=2).stdout.strip() or None
    except Exception:
        out["git_commit"] = None
    return out


def source_files(ds):
    """[{det, file, bytes}] of the DAQ files behind the data set."""
    if getattr(ds, "source_files", None):
        return list(ds.source_files)
    import os
    out = []
    for det, loader in sorted(getattr(ds, "_state", (None, {}))[1].items()):
        try:
            out.append({"det": det, "file": os.path.basename(loader.path), "bytes": os.path.getsize(loader.path)})
        except OSError:
            out.append({"det": det, "file": os.path.basename(getattr(loader, "path", "?")), "bytes": None})
    return out


def capture(session, gui=None, comment="", created=None, n_saves=1, source=None):
    """The dict h5file.write_file stores. Reads only immutable results, so it can run in any thread."""
    s = session
    snap = s.ds.snapshot()
    raw = {det: {"run": snap.run, "epoch": sn.epoch, "columns": dict(sn.columns)} for det, sn in snap.detectors.items()}
    cfg = dataclasses.asdict(s.config)
    bootstrap = {k: s.abel.bootstrap_result(k) for k in ("a", "b")}
    meta = {"created": created or h5file.now(), "saved": h5file.now(), "n_saves": n_saves, "comment": comment,
            "run": snap.run, "live": source is None, "loaded_from": source, "status": s.status_line(),
            "source_files": source_files(s.ds), "software": _versions(), "program_format": h5file.VERSION}
    try:
        plot_data = plots.build(s, gui)
    except Exception as exc:                              # never lose a save because of a plot
        plot_data = {"error": f"{type(exc).__name__}: {exc}"}
    return {"meta": meta, "config": cfg, "gui": gui or {}, "raw": raw, "plots": plot_data,
            "settings": stage_settings(s),
            "results": {"coincidence": s.coincidence(), "selection": s.selection(), "poscorr": s.position_correction(),
                        "imaging": s.imaging(), "polar": s.polar_result(), "abel": s.abel_result(),
                        "bootstrap": {k: v for k, v in bootstrap.items() if v is not None} or None}}
