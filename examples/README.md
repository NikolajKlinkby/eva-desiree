# Examples (no GUI)

Plain Python scripts that use the same pipeline as the program, without a window. Run them from the project
folder (`python examples/<name>.py`; each has `--help` or a docstring). They use the recorded runs in
`examples/test_data/` unless you point them at your own folder. Pictures and files go to `examples/output/`.
`common.py` holds the small helpers (starting a session, waiting for the stages, printing a report).

| script | shows | needs |
|---|---|---|
| `01_online_analysis.py` | follow a DAQ folder while the files grow (or `--demo` replay), print status, autosave to an analysis file | – |
| `02_offline_raw_analysis.py` | analyse finished CSV files: coincidences, cuts, images, centre, Abel inversion, optional picture | matplotlib for the picture |
| `03_change_settings.py` | change cuts, coincidence window and the Abel corrections through the stage setters and compare | – |
| `04_save_h5.py` | write the analysis file (raw data, settings, results, plots) | h5py |
| `05_read_h5.py` | read such a file with plain h5py: layout, raw columns, settings, the data of every plot | h5py |
| `06_reanalyse_h5.py` | open a file as offline analysis, change the analysis (maximum entropy, Sum cut), save a new file | h5py |
| `07_bootstrap.py` | error bands of I(rho) and beta_n(rho), saved to the file | h5py for `--save` |
| `08_abel_inversion_synthetic.py` | the Abel inversion alone on a synthetic ring: plain vs. regularised vs. maximum entropy | – |
| `09_coincidence_lowlevel.py` | the building blocks without a `Session`: `DetectorSet`, `CoincidenceFilter`, `SelectionFilter`, `Imager` | – |

Typical order: `04` makes `examples/output/example_run.h5`, which `05` and `06` then read.
