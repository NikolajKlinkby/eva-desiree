"""The analysis file (HDF5, Qt-free).  Run `python examples/05_read_h5.py file.h5` to see the layout of a file.

  /README                    this layout and an index of all plots, as text inside the file
  /meta                      when saved, software versions, run, DAQ files, status text, comment
  /config                    session configuration (detectors, bins, search window ...)
  /gui                       the value of every input widget of the three tabs (restores the window)
  /settings/<stage>          every stage's settings, as the real objects (see tree.py):
                               coincidence, selection, poscorr, imager, polar, abel, abel_bootstrap
  /raw/detector_<n>/<col>    the raw event columns of detector n, one dataset per DAQ column (Sum, Event time, ...)
  /results/<stage>           what the stages computed, with the field names of the program:
                               coincidence, selection, poscorr, imaging, polar, abel, bootstrap
  /plots/...                 the data of every plot of the GUI, one group per plot, with x and y arrays named
                               after what they are, and attributes title, xlabel, ylabel, gui_location
                               (see plots.py)

Everything under /results and /plots is recomputable from /raw and /settings; it is stored so that the file is
readable without this program (h5py, HDFView, MATLAB ...). Opening the file in the program re-runs the pipeline
from /raw with /settings, so the analysis can be continued (offline analysis).

The file is written to <name>.part and renamed when complete: a crash while saving never damages the previous
version. Format version 1 files (JSON blobs, datasets a0 / c0) are still read.
"""
import datetime
import json
import os
import re

import numpy as np

from storage.tree import read_tree, safe_name, write_tree

FORMAT = "vmi-analysis"
VERSION = 2
RAW_COMPRESS = dict(compression="gzip", compression_opts=1, shuffle=True)


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("saving and loading analysis files needs the h5py package (pip install h5py)") from exc
    return h5py


def _get_json(group, name):                                    # format version 1
    raw = group[name][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _attr_str(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def column_dataset_names(columns):
    """Unique HDF5 names for the DAQ column names (same order); the names themselves where possible."""
    out, seen = [], set()
    for c in columns:
        name, n = safe_name(c), 1
        while name in seen:
            n += 1
            name = f"{safe_name(c)}~{n}"
        seen.add(name)
        out.append(name)
    return out


def layout_text(snap):
    """The text of /README: where things are, plus the index of the plots."""
    from storage import plots
    lines = [doc_line for doc_line in __doc__.strip().splitlines()[1:]]
    lines += ["", "PLOTS (group: title  [where in the program])", ""] + plots.index(snap.get("plots") or {})
    return "\n".join(lines)


def write_file(path, snap):
    """snap: {"meta": dict, "config": dict, "gui": dict, "settings": {name: obj}, "results": {name: obj},
              "plots": nested dict (plots.build), "raw": {det: {"run": str, "epoch": int, "columns": {name: array}}}}"""
    h5py = _h5py()
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    meta = dict(snap.get("meta", {}))
    meta.setdefault("saved", now())
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = VERSION
        f.attrs["created"] = meta.get("created", meta["saved"])
        f.attrs["saved"] = meta["saved"]
        f.attrs["n_saves"] = int(meta.get("n_saves", 1))
        f.attrs["description"] = "EVA analysis file: raw data, settings, results and plots; see the dataset /README"
        f.create_dataset("README", data=layout_text(snap), dtype=h5py.string_dtype())
        write_tree(f, "meta", meta)
        write_tree(f, "config", snap.get("config", {}))
        write_tree(f, "gui", snap.get("gui", {}))
        g = f.create_group("settings")
        for name, obj in snap.get("settings", {}).items():
            write_tree(g, name, obj)
        raw = f.create_group("raw")
        for det, d in snap.get("raw", {}).items():
            gd = raw.create_group(f"detector_{int(det)}")
            gd.attrs["detector"] = int(det)
            gd.attrs["run"] = "" if d.get("run") is None else str(d["run"])
            gd.attrs["epoch"] = int(d.get("epoch", 0))
            cols = d["columns"]
            gd.attrs["n_rows"] = int(min((len(c) for c in cols.values()), default=0))
            gd.attrs["columns"] = json.dumps(list(cols))        # the original names, in file order
            for name, a in zip(column_dataset_names(cols), cols.values()):
                a = np.ascontiguousarray(a)
                gd.create_dataset(name, data=a, **RAW_COMPRESS) if a.size else gd.create_dataset(name, data=a)
        res = f.create_group("results")
        for name, obj in snap.get("results", {}).items():
            if obj is not None:
                write_tree(res, name, obj)
        if snap.get("plots"):
            write_tree(f, "plots", snap["plots"])
    os.replace(tmp, path)


def _check(f):
    if _attr_str(f.attrs.get("format", "")) != FORMAT:
        raise ValueError("not an EVA analysis file")
    if int(f.attrs.get("format_version", 0)) > VERSION:
        raise ValueError(f"the file was written by a newer version of the program (format {int(f.attrs['format_version'])})")


def _is_dataset(obj):
    return hasattr(obj, "dtype")


def read_meta(path):
    """Everything but the raw data and the results: {"meta", "config", "gui", "settings": {name: obj}}."""
    h5py = _h5py()
    with h5py.File(os.fspath(path), "r") as f:
        _check(f)
        if _is_dataset(f["config"]):                           # format version 1
            meta, config, gui = _get_json(f["meta"], "info"), _get_json(f, "config"), _get_json(f, "gui")
        else:
            meta, config, gui = read_tree(f["meta"]), read_tree(f["config"]), read_tree(f["gui"])
        return {"meta": meta, "config": config, "gui": gui,
                "settings": {k: read_tree(f["settings"][k]) for k in f["settings"]}}


def read_raw(path):
    """(run, {det: (epoch, {column: array})}, [source files])"""
    h5py = _h5py()
    out, run = {}, None
    with h5py.File(os.fspath(path), "r") as f:
        _check(f)
        for key in f["raw"]:
            gd = f["raw"][key]
            names = json.loads(_attr_str(gd.attrs["columns"]))
            run = _attr_str(gd.attrs["run"]) or None
            if int(f.attrs["format_version"]) == 1:           # datasets c0, c1, ...
                cols = {n: np.asarray(gd[f"c{i}"][()]) for i, n in enumerate(names)}
            else:
                cols = {n: np.asarray(gd[ds][()]) for n, ds in zip(names, column_dataset_names(names))}
            out[int(re.search(r"(\d+)$", key).group(1))] = (int(gd.attrs["epoch"]), cols)
        info = _get_json(f["meta"], "info") if _is_dataset(f["config"]) else read_tree(f["meta"])
        files = info.get("source_files", [])
    return run, out, files


def read_results(path, names=None):
    """{stage: object} of /results (all, or only the given names)."""
    h5py = _h5py()
    with h5py.File(os.fspath(path), "r") as f:
        _check(f)
        return {k: read_tree(f["results"][k]) for k in f["results"] if names is None or k in names}


def read_plots(path):
    """The nested dict of /plots (arrays, strings, numbers), {} if the file has none."""
    h5py = _h5py()
    with h5py.File(os.fspath(path), "r") as f:
        _check(f)
        return read_tree(f["plots"]) if "plots" in f else {}
