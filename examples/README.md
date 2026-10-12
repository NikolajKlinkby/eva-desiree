# EVA examples (no GUI)

Plain Python scripts that use the same pipeline as the program, **without a window**. They are the quickest way to
see how the core objects fit together; [`docs/gui.md`](../docs/gui.md) explains the same operations tab by tab.

```bash
python examples/02_offline_raw_analysis.py            # from the project folder; each script has --help or a docstring
```

* They use the recorded runs in [`test_data/`](test_data) (detectors 1 and 3) unless you point them at your own
  folder (`--dir`).
* Pictures and files are written to `examples/output/` (git-ignored).
* [`common.py`](common.py) holds the small helpers: starting a session, waiting for the stages, printing a report.

## The scripts

| Script | What it shows | Needs |
|---|---|---|
| [`01_online_analysis.py`](01_online_analysis.py) | follow a DAQ folder while the files grow (or `--demo` replay), print the status, autosave to an analysis file | – |
| [`02_offline_raw_analysis.py`](02_offline_raw_analysis.py) | analyse finished CSV files: coincidences, cuts, images, centre, Abel inversion, optional picture | matplotlib for the picture |
| [`03_change_settings.py`](03_change_settings.py) | change cuts, coincidence window and the Abel corrections through the stage setters, and compare the results | – |
| [`04_save_h5.py`](04_save_h5.py) | write the analysis file (raw data, settings, results, plots) | h5py |
| [`05_read_h5.py`](05_read_h5.py) | read such a file with plain h5py: layout, raw columns, settings, the data of every plot | h5py |
| [`06_reanalyse_h5.py`](06_reanalyse_h5.py) | open a file as offline analysis, change the analysis (maximum entropy, Sum cut), save a new file | h5py |
| [`07_bootstrap.py`](07_bootstrap.py) | error bands of $I(\rho)$ and $\beta_n(\rho)$, saved to the file | h5py for `--save` |
| [`08_abel_inversion_synthetic.py`](08_abel_inversion_synthetic.py) | the Abel inversion alone on a synthetic ring: plain vs. regularised vs. maximum entropy | – |
| [`09_coincidence_lowlevel.py`](09_coincidence_lowlevel.py) | the building blocks without a `Session`: `DetectorSet`, `CoincidenceFilter`, `SelectionFilter`, `Imager` | – |

## Which one do I need?

| I want to ... | Start with |
|---|---|
| watch a measurement while it runs | `01` (`python examples/01_online_analysis.py --dir <daq folder> --a 3 --b 1`) |
| analyse a finished run in a script | `02`, then `03` to vary the analysis |
| keep an analysis and continue it later | `04` → `05` (read) or `06` (continue) |
| get error bars | `07` |
| understand the Abel corrections | `08` |
| build my own pipeline from the parts | `09` |

Typical order for the file examples: `04` writes `examples/output/example_run.h5`, which `05` and `06` then read.
