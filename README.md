# EVA: Electron–neutral VMI Analysis

Live and offline analysis for the electron–neutral coincidence velocity-map-imaging (VMI) experiment at the DESIREE
facility in Stockholm. EVA reads data recorded with an MCP and resistive anode encoder:
detector CSV files written by the DAQ (`RA<det>_<date>.<time>.csv`) are followed while they grow, or read
afterwards. Python/Qt successor of the author's C++/ROOT program (see [Acknowledgements](#acknowledgements)).

**What it does**

* coincidence search between two detectors (laser-shot sidebands for the accidentals), Sum and event-time cuts,
  electron position correction, detector images
* centring and circularisation of the image
* Abel inversion (rBASEX) to I(rho) and the anisotropy parameters beta_n(rho), with optional corrections
  (statistical weights, smoothness, positivity, maximum entropy)
* bias-corrected bootstrap error bands
* everything saved to one HDF5 file (raw data, settings, results, the data of every plot) that can be opened
  again for offline analysis
* a GUI (PyQt6 + pyqtgraph) and a plain Python API for scripts

## Install and run

    pip install -r requirements.txt           # numpy, scipy, pandas, h5py, PyQt6, pyqtgraph
    python main.py --demo                     # replays examples/test_data, no hardware needed
    python main.py --demo --theme light       # dark is the default
    python main.py --dir <daq folder> --a 3 --b 1 --start
    python main.py

Without a window: see [`examples/`](examples/README.md) (online analysis, offline analysis of raw files,
analysis files, bootstrap, the Abel inversion on synthetic data).

## Documentation

| | |
|---|---|
| [`docs/gui.md`](docs/gui.md) | the tabs, the Abel loss terms, bootstrap, centring and circularisation |
| [`docs/file-format.md`](docs/file-format.md) | saving, offline analysis and the layout of the HDF5 file |
| [`examples/README.md`](examples/README.md) | non-GUI example scripts |
| [`docs/notes/main.pdf`](docs/notes/main.pdf) | the notes: the mathematics (rBASEX, imaging, errors, calibration); source `docs/notes/main.tex`, references `Bibliography.bib` |

The notes are built in `docs/notes` with `pdflatex main.tex -> biber main -> pdflatex main.tex` (biblatex with the biber backend; needs a full TeX Live:
biblatex-phys, dsfont, bbm, mathdots, physics).

## Layout

    main.py                   starts the GUI
    src/
      daq/data_loader.py      tails the CSV files (live, partial lines, resets) -> snapshots
      analysis/coincidence.py pulse matching, clock map, pair search, accidentals (laser-shot sidebands)
      analysis/filters.py     Sum / event-time cuts, before or after coincidence, true-pair weights
      analysis/imaging.py     a,b,c,d -> x,y -> transform -> binning -> image (Cartesian and polar binning)
      analysis/polar.py       centring, circularisation (estimate, save/load), polar images; live stage
      analysis/abel.py        Abel inversion in the notation of docs/notes/main.tex: d -> H -> (p^T)^-1 -> C^-1; I(rho), beta_n
      analysis/abel_loss.py   further Abel loss terms: statistical weights, smoothness of I(rho), non-negativity (scipy)
      analysis/abel_analysis.py  live Abel stage: events -> radial bins d_n(r) -> I(rho), beta_n(rho); runs the bootstraps
      analysis/poscorr.py     hit position (x, y) of the coincidence pairs versus dt (binned means, raw subsample)
      analysis/poscorr_stage.py live stage: the electron correction as per-event displacements for the imagers (Analysis tab, Abel)
      analysis/bootstrap.py   bias-corrected bootstrap of I(rho) and every beta_n(rho) (Poisson weights, background thread)
      session.py              owns the live pipeline; demo mode; offline mode (raw data from an analysis file); no Qt
      storage/h5file.py       the analysis file (HDF5): layout, write (atomic), read, the /README text in the file
      storage/plots.py        the data of every plot of the program under readable names (/plots)
      storage/tree.py         settings / results (dataclasses, arrays) <-> HDF5 groups, no class-by-class code
      storage/capture.py      Session -> content of the file; saved settings -> stages
      storage/static_source.py  the raw data of a file as a data source for the same pipeline
      storage/saver.py        background saving, only when something changed
      gui/plotdata.py         results -> arrays for each plot; no Qt
      gui/widgets.py          ImagePlot, HistPlot, CutPlot (draggable cut lines)
      gui/tab_detectors.py    tab 1
      gui/tab_coincidence.py  tab 2
      gui/tab_polar.py        tab 3 "Analysis": centring, circularisation and Abel inversion (two pages of plots)
      gui/theme.py            dark / light palette (Qt + pyqtgraph)
      gui/gui_state.py        value of every input widget <-> dict (stored in the analysis file)
      gui/main_window.py      source strip, file strip (save / open), tabs, status bar, refresh timer
    tools/fake_daq.py         replays recorded runs as a live DAQ (for testing / demos)
    examples/                 non-GUI examples; examples/test_data/: two recorded runs (detectors 1 and 3)
    docs/gui.md, file-format.md  the GUI guide, the analysis file
    docs/notes/               main.tex / main.pdf: the mathematics (rBASEX, imaging, errors, calibration), Bibliography.bib


## How the pieces talk

    DetectorSet -> CoincidenceFilter -> SelectionFilter -> Imager        (one worker thread each)
         |                |                  |                |
         +---- immutable results, read with .result() / .snapshot() ----+
                                  |
                       QTimer (200 ms) in the GUI thread -> plotdata.py -> pyqtgraph

No locks and no Qt signals across threads: workers publish immutable results, the GUI
polls them. User input goes the other way through small thread-safe setters
(`sel.set_sum`, `sel.set_time`, `flt.set_window`, `img.set_mode`, ...).

## Notation (main.tex and abel.py)

The mathematics is written up in `docs/notes/main.tex`.
`src/analysis/abel.py` uses its notation, so formulas and code can be read side by side:

| `main.tex` | `abel.py` |
|---|---|
| `rho, theta'` (3D), `r, theta` (detector) | `rho`, `r`, angle from the laser axis |
| `n`, angular order (power of `cos`) | `orders` |
| `d_{nr}`, radially binned data | `d[n, r]`, from `RadialBinner.d(weights)` |
| `H_{rni}`, inverse Hankel matrix per radius | `AbelBasis.H[r, n, i]` (`hankel_matrix` gives the Hankel part `K`) |
| `p_n^T`, `(p_n^T)^{-1}` | `AbelBasis.pT[n, r, R]`, `AbelBasis.pT_inv[n, R, r]` |
| `I_{n rho}`, `sum_j (p^T)^-1 sum_i H d` | `get_angular_intensity(d)` -> `I_n` |
| `C`, `C^{-1}` | `AbelBasis.C`, `AbelBasis.C_inv`; `get_legendre_coefficients(I_n)` |
| `I(rho) = 4 pi rho^2 ...`, `beta_n` | `AbelResult.intensity`, `AbelResult.beta` |

## Citing and license

Released under the [MIT License](LICENSE): free to use, modify and distribute, as long as the copyright notice
is kept. If you use the program for a publication, please also cite it (see [`CITATION.cff`](CITATION.cff); GitHub
shows a "Cite this repository" button) and the methods papers below.

## Acknowledgements

This program continues
**[VMI-RAE-MCP-coincidense-DA-program](https://github.com/NikolajKlinkby/VMI-RAE-MCP-coincidense-DA-program)**
by Nikolaj Klinkby (`git clone git@github.com:NikolajKlinkby/VMI-RAE-MCP-coincidense-DA-program.git`),
the C++/ROOT analysis of velocity-map images recorded with an MCP and resistive anode encoder, from
which this Python/Qt version grew. The analysis chain here is that program's design: centring from
the projections by convolution, radial circularisation, an rBASEX-type Abel inversion built on
Hankel matrices, and the bias-corrected bootstrap for the errors. What is new is that it runs live, combines the stages in one program with a GUI, and stores the whole analysis in one file.
The methods are described in `docs/notes/main.tex` and rest on:

* M. Ryazanov, *Development and implementation of methods for sliced velocity map imaging*,
  PhD thesis, University of Southern California (2012): the rBASEX basis set.
* B. Efron, *Better Bootstrap Confidence Intervals*, J. Am. Stat. Assoc. (1987),
  doi:10.1080/01621459.1987.10478410: the bias-corrected bootstrap.