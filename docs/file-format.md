# The analysis file (HDF5)

EVA writes everything it knows about an analysis to **one HDF5 file**: the raw data, every setting, the results, and
the data behind every plot of the window – under readable names. The file can be opened again in EVA for **offline
analysis**, and read without EVA in anything that reads HDF5 (Python/h5py, HDFView, MATLAB, ...).

**Contents**

1. [Saving and opening](#1-saving-and-opening)
2. [Layout](#2-layout)
3. [Where is the data of a plot?](#3-where-is-the-data-of-a-plot)
4. [Reading a file](#4-reading-a-file)
5. [Notes](#5-notes)

---

## 1. Saving and opening

The second strip of the main window:

| Control | What it does |
|---|---|
| **save analysis to file** (toggle) | switches saving on; file name and folder of your choice (empty: `eva_analysis_<date>.h5` in the data folder; `browse...` opens the file dialog) |
| autosave interval | while data come in the file is rewritten every N seconds (default 60), and only if something changed |
| **save now** | writes immediately |
| **open saved file (offline analysis)...** | loads a file and runs the whole pipeline again on its raw data: coincidence search, cuts, position correction, images, centring, circularisation, Abel inversion, bootstrap bands |

* The file is also written when the session stops (also when the window is closed).
* A save goes to `<name>.part` and is renamed when complete, so a crash never damages the previous version.
* Opening restores every setting and every box in the three tabs; everything can be changed and saved again. Saving
  back to the opened file asks first; the automatic save on stop never overwrites it (default target:
  `<name>_reanalysed.h5`).

In code, see [`examples/04_save_h5.py`](../examples/04_save_h5.py) and [`06_reanalyse_h5.py`](../examples/06_reanalyse_h5.py),
and [section 6 of the GUI guide](gui.md#6-saving-and-offline-analysis).

## 2. Layout

```text
analysis.h5
├── README                      text: this layout and an index of every plot (path, title, where in the program)
├── meta/                       when saved, versions (python, numpy, scipy, h5py, git commit), run, DAQ files, comment
├── config/                     detectors, bins, search window ...
├── gui/                        the value of every input widget (restored on opening)
├── settings/<stage>/           coincidence, selection (cuts), position correction, imager, polar, abel, bootstrap
├── raw/detector_<n>/<column>   the raw event columns, with their DAQ names
├── results/<stage>/...         what was computed, with the field names of the program
└── plots/...                   the data of every plot in the GUI, named after the plot
```

| Group | Content |
|---|---|
| `/README` | the layout and the index of all plots |
| `/meta` | saved time, software versions, run, DAQ files with sizes, status text, comment |
| `/config`, `/gui` | detectors, bins, search window ...; the value of every input widget |
| `/settings/<stage>` | coincidence, selection (cuts), position correction, imager, polar (centre, circularisation table), abel (all loss terms), bootstrap; field names as in the program |
| `/raw/detector_<n>/<column>` | `Device ID`, `Event ID`, `a`, `b`, `c`, `d`, `Sum`, `Event time`, ... (gzip) |
| `/results/<stage>/...` | pairs and window, masks / weights / cut flow, images, polar images, $d_{nr}$, $I_n(\rho)$, $I(\rho)$, $\beta_n(\rho)$, bootstrap bands and samples |
| `/plots/...` | **the data of every plot in the GUI** (next section) |

Groups and datasets carry the structure: a dataclass becomes a group, an array a dataset, a number an attribute.
So `f["results/abel/a/abel/intensity"]` is an array and `f["settings/abel/a"].attrs["tikhonov"]` a number.

## 3. Where is the data of a plot?

Every plot group has the attributes `title`, `xlabel`, `ylabel` and `gui_location` (tab and page).

| Tab / page | Path under `/plots` | Content |
|---|---|---|
| Detectors | `detectors/detector_<n>_<A\|B>/` | `image`, `event_time`, `charge_sum`, `cuts` |
| Coincidence, timing | `coincidence/timing/` | `dt_full`, `dt_zoom`, `window_vs_control`, `pair_rate`, `clock_offset_residual`, `summary` |
| Coincidence, position correlation | `coincidence/position_correlation/` | `parameters`; per detector `raw_pairs`, `binned_means`, `fitted_dt_shape`; `electron_images/` (only if the correction is on) |
| Analysis | `analysis/<A\|B>_detector_<n>/` | `centring/`, `image_centred_only`, `image_centred_circularised`, `radial_profile`, `circularisation`, `radial_bins_*` |
| Analysis, Abel inversion | `analysis/<A\|B>_detector_<n>/abel/` | `summary`, `intensity` ($\rho$, $I$, lower, upper), `beta_<n>` ($\rho$, $\beta$, lower, upper), `reconstructed_image` |

## 4. Reading a file

**With plain h5py** – no EVA needed:

```python
import h5py

with h5py.File("analysis.h5", "r") as f:
    print(f["README"][()].decode()[:600])                       # the layout and the index of all plots

    raw = f["raw/detector_3"]
    print(list(raw), raw.attrs["n_rows"])                        # column names, number of rows
    total = raw["Sum"][()]                                       # one raw column as an array

    ab = f["plots/analysis/A_detector_3/abel/intensity"]
    rho, intensity = ab["radius_rho"][()], ab["intensity"][()]
    print(ab.attrs["title"], "|", ab.attrs["gui_location"])      # what it is and where the GUI shows it
```

**With EVA's reader** – returns the real objects (settings dataclasses, results) and nested dictionaries:

```python
import sys; sys.path.insert(0, "src")
from storage import h5file

info = h5file.read_meta("analysis.h5")             # meta, config, widget state and the settings of every stage
run, detectors, files = h5file.read_raw("analysis.h5")
results = h5file.read_results("analysis.h5")       # the results, by stage
plots = h5file.read_plots("analysis.h5")           # the /plots tree as nested dictionaries
print(list(plots))
```

[`examples/05_read_h5.py`](../examples/05_read_h5.py) walks through all of it and draws a picture.

## 5. Notes

* `/results` and `/plots` can be recomputed from `/raw` and `/settings`; they are stored so that the file is useful
  without EVA. Per-event arrays of the pipeline and the Abel matrices are not stored – they are rebuilt from the raw data.
* Files of format version 1 (older, with `a0` / `c0` dataset names) can still be opened.
* Compression: arrays above 1024 elements use gzip level 1 with shuffle.
* `*.h5`, `*.hdf5` and `*.h5.part` are in `.gitignore`.
