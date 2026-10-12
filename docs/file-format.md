# The analysis file (HDF5)

The second strip of the window writes everything to one HDF5 file (`h5py`; the tabs and the analysis stay as
before):

* **save analysis to file** (toggle) with a file name and a folder of your choice (empty: `eva_analysis_<date>.h5` in
  the data folder; `browse...` opens the file dialog). While data come in the file is rewritten every N seconds
  (default 60, only if something changed), by `save now`, and when the session stops (also when the window is closed).
  A save is written to `<name>.part` and renamed when complete, so a crash never damages the previous version.
* **open saved file (offline analysis)...** loads such a file and runs the whole pipeline again on its raw data:
  coincidence search, cuts, position correction, images, centring, circularisation, Abel inversion, bootstrap bands.
  Every setting and every box in the three tabs comes back as it was; everything can be changed and saved again.
  Saving back to the opened file asks first; the automatic save on stop never overwrites it (default target:
  `<name>_reanalysed.h5`).

What the file contains (the file itself has a `/README` text with this layout and an index of every plot):

| group | content |
|---|---|
| `/README` | the layout, the index of all plots (path, title, where in the program) |
| `/meta` | when saved, software versions (python, numpy, scipy, h5py, git commit), run, DAQ files with sizes, status text, comment |
| `/config`, `/gui` | detectors, bins, search window ...; the value of every input widget |
| `/settings/<stage>` | coincidence, selection (cuts), position correction, imager, polar (centre, circularisation table), abel (all loss terms), bootstrap; field names as in the program |
| `/raw/detector_<n>/<column>` | the raw event columns with their DAQ names (`Device ID`, `Event ID`, `a`, `b`, `c`, `d`, `Sum`, `Event time`), gzip |
| `/results/<stage>/...` | what was computed, with the field names of the program: pairs and window, masks / weights / cut flow, images, polar images, d_nr, I_n(rho), I(rho), beta_n(rho), bootstrap bands and samples |
| `/plots/...` | **the data of every plot in the GUI**, named after the plot (below) |

Groups and datasets carry the structure; every plot group has the attributes `title`, `xlabel`, `ylabel` and
`gui_location` (tab and page). Plots, as they are in the program:

* `/plots/detectors/detector_<n>_<A|B>/` `image`, `event_time`, `charge_sum`, `cuts` (Detectors tab)
* `/plots/coincidence/timing/` `dt_full`, `dt_zoom`, `window_vs_control`, `pair_rate`, `clock_offset_residual`, `summary`
* `/plots/coincidence/position_correlation/` `parameters`, per detector `raw_pairs`, `binned_means`, `fitted_dt_shape`, and `electron_images/` (only if the correction is on)
* `/plots/analysis/<A|B>_detector_<n>/` `centring/`, `image_centred_only`, `image_centred_circularised`, `radial_profile`, `circularisation`, `radial_bins_*`, and `abel/` with `summary`, `intensity` (rho, I, lower, upper), `beta_<n>` (rho, beta, lower, upper), `reconstructed_image`

Read it with any HDF5 tool (h5py, HDFView, MATLAB, ...); `examples/05_read_h5.py` shows plain h5py. In Python with
this package: `from storage import h5file` (with `src/` on the path); `h5file.read_meta(path)`, `read_raw`, `read_results`, `read_plots`
return the real objects / nested dictionaries.

`/results` and `/plots` can be recomputed from `/raw` and `/settings`; they are stored so the file is useful without
this program. Per-event arrays of the pipeline and the Abel matrices are not stored; they are rebuilt from the raw
data. Files of format version 1 (older, `a0`/`c0` names) can still be opened.
