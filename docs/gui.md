# GUI guide

The three tabs, the Abel loss terms, the bootstrap and centring / circularisation in detail.

Tab "Detectors": one row per detector: image | event time | charge sum. The time and Sum
plots have two draggable vertical lines (tick "cut", drag, or type numbers). Grey = all
events, orange = true coincidence pairs (background-subtracted, independent of the cuts),
blue = events kept. Controls: image mode, Sum cut before/after coincidence, coincidence
on/off, fit image range.

Tab "Coincidence": dt overview with window and control windows, dt zoom (drag the window
there), window vs control-window counts, pair rate vs time, clock drift residual, text
summary. Controls: search window / cycle shift (re-searches everything), nominal period
and control windows (live), automatic or manual window. A second page, "Position Correlation",
has one plot per detector: the hit position x (left axis, blue) and y (right axis, orange) of the
coincidence pairs against dt, as binned means with standard errors plus a subsample of the individual
pairs. Controls: dt range around the peak, bin width, show pairs / window. These are all pairs in the
range, without Sum or time cuts and without accidental subtraction (accidentals are flat in dt).
The right column fits, inside the coincidence window (automatic width defaults to 6 sigma), how
the electron detector's position depends on the neutral detector's position and on dt,
r_corr = r_e - M r_n - c tau - d (docs/notes/main.tex). The fit is weighted: pairs in the window count +1,
pairs in each control window -1/K (moved into the window in dt), so only true coincidences determine
M and c (uncheck 'weighted fit' to keep the accidentals); d keeps the mean position. Shown: the electron
image of the window before and after the correction, and the corrected image for 1-6 equal slices of dt
(all on the fixed detector grid of the Detectors tab, coordinates x1000, colour scale 0..99.8th
percentile, so they fill in place as data come), the fitted M, c with errors and t-values, and the
weighted mean corrected position of every slice. 'also show corrected position' adds dashed
corrected means to the electron detector's x/y vs dt plot without touching the uncorrected ones: a
trend left there is the residual dt dependence. 'dt polynomial order' (default 0 = neutral position
only) adds a polynomial in dt to the model (order 1 = c tau; higher orders are shrunk slightly so they stay
stable with the accidental subtraction) and, with the dashed curves on, draws the fitted dt shape as a dotted
line. 'subtract accidentals in the means' (default on) draws the dt-binned means of x and y of the true
coincidences only: every bin counts its pairs +1 and the pairs of the same bin moved by each control-window
offset -1/K, so accidentals cancel bin by bin (bins with fewer than 3 true pairs are left empty). 'correct dt for the hit positions' (timing walk) fits dt = const + k.(position) inside the peak (+-2 sigma by
default; electron x, y, optionally also the neutral's) with the accidentals subtracted and replaces dt by
dt' = dt - k.(pos - mean) for the whole page (means, fits, images, dt axis). What the position-vs-dt' profile
still shows is not a linear position-dependent timing effect. The text box
reports 'flat in dt? chi2/dof' of x_e and y_e before and after (10 equal dt bins, 1 = flat). The electron detector is chosen with a combo (default
A, the detector with the earlier arrival time in the example data). 'apply Sum / time cuts of the Detectors tab' (default on)
keeps only pairs whose two events pass the charge-sum and time limits currently set there (whatever the Sum stage),
before anything on this page is computed.

The whole feature has a master switch, 'enable position correlation' (default off, its settings sections start folded): off, the page computes nothing and nothing is passed on. 'pass the correction on to the Analysis tab' (electron correction section, default off) runs the same correction in the pipeline
(`analysis/poscorr_stage.py`, between the selection and the imagers): with the Sum/time cuts, the walk correction, the
neutral term and the dt polynomial set on this page, every electron row that sits in a pair of the window (or, for the
accidental subtraction, of a control window) is displaced by the correction before the Detectors-tab image, centring,
circularisation and the Abel inversion see it. A row in both takes the displacement of its window pair; rows without a partner
are not displaced; the neutral detector's image is untouched. The Position Correlation page itself always shows the raw
positions. The fit is refreshed at most ~20% of the time while data come in.

Tab "Analysis": settings for every operation on the left (collapsible sections: detector and events,
polar binning, centring, circularisation, Abel inversion); on the right two pages to switch between.

* Page "centring and circularisation": image after centring (+ circularisation), angle-integrated radial
  profile, radial bins d_n(r) after and before circularisation, centring diagnostics, circularisation s(angle).
* Page "Abel inversion": measured (angular binned) image, reconstructed image, radial bins d_n(r),
  intensity I(rho), anisotropy beta_n(rho), and a summary.

d_n(r) are the radial bins of angular order n, by default n = 0, 1, 2. Order 1 is not expected for a
mirror-symmetric distribution but is kept: it absorbs small asymmetries of the data. The Abel controls
(highest order, even orders only, laser axis angle, regularisation, further loss terms, beta threshold) are
per detector.

## Loss terms of the Abel inversion

The plain (rBASEX) inverse reproduces the measured moments exactly, noise included, which shows up as
ringing: a negative dip of I(rho) in front of a peak. The regularisation (Tikhonov, per angular order) is a
general damping. All corrections are in the section "Regularisation (corrections to the inversion)" between "Abel
inversion" and "Bootstrap"; a bold line at its top states in words which of them are applied. Numbered as there, 1 is
the Tikhonov damping, 2 to 5 act on all orders and radii together (`analysis/abel_loss.py`, needs scipy):

* **statistical weights.** The data are weighted with their covariance, estimated from the events (including
  the noise of the accidental subtraction). Without a penalty or constraint this changes nothing (the fit is
  exact); with one, noisy radii and angular orders are smoothed more than well-determined ones. The weights
  are those of the measurement and stay fixed for all bootstrap samples.
* **smoothness of I(rho).** Penalty on the curvature (or gradient) of I(rho) = 4 pi rho^2 L_0(rho), optionally
  also of the other Legendre orders (the anisotropy). The strength is relative, in the units of the
  regularisation; compare by eye on the live curve.
* **non-negativity.** "I(rho) >= 0" forbids the dip; "whole distribution >= 0" requires the full angular
  distribution sum_n I_n cos^n(theta) to be non-negative at every radius and 16 angles. This is a constrained
  least-squares problem (ADMM; each resample of a bootstrap runs through it). Where the constraint is active,
  L_0 is zero and beta is undefined: the "beta shown above" threshold hides those bins.

* **maximum entropy (Poisson entropy MEVIR).** Adds alpha^2 F^2 sum w s(f/F) with s(x) = x ln(x/m) - x + m, the entropy of
  a positive additive distribution (it follows from Poisson counting statistics), on the angular distribution f at
  every radius and 16 angles; F is its peak, m the "default level" (percent of the peak; where the data say nothing
  f relaxes to it). It is a soft positivity: the curvature 1/f makes weak tails stiff and strong peaks soft, as counting
  noise is. It replaces the hard constraint (the two cannot be combined; the dropdown is then disabled) and combines
  with 1 to 3. The data term stays the chi2 with the event covariance: the Abel moments are weighted sums with negative
  accidental-subtraction weights, not counts, so a Poisson likelihood of the data is undefined. One damped Newton
  optimisation per sample (about 15 steps); a bootstrap of 2000 resamples takes about 30 s to a few minutes.
  Where f reaches the floor numerically (a pronounced dip), beta is undefined as for the hard constraint.

Weights and smoothness keep the inversion linear (a matrix applied to the data), so a bootstrap costs the same
as before. Non-negativity makes every sample an optimisation: 256 resamples with "whole distribution" take about
10 s more, with "I(rho) >= 0" next to nothing. The bootstrap always uses the loss terms of the Abel result it
resamples.

## Bootstrap (bias corrected)

The section "Bootstrap" of the Analysis tab (and `AbelAnalysis.start_bootstrap`) resamples the data of the
current Abel result and shows a bias-corrected interval of I(rho) and of every beta_n(rho) as a shaded band
on the two plots; the text summary quotes the interval at the maximum of I(rho).

* **Resampling.** Every event weight (1, or the background-subtraction weight: positive in the coincidence
  window, negative in the control windows) is multiplied by an independent Poisson(1) number, so the noise
  of the accidental subtraction is part of the error. "event" draws one number per event, "cycle" one per
  Sync-Time cycle (about 1 s) shared by all its events, which keeps correlations inside a cycle.
* **Interval.** For each radial bin separately: z0 = Phi^-1(fraction of bootstrap values below the measured
  one), then the quantiles at Phi(2 z0 -+ z) of the sorted bootstrap values (z = 1 for +-1 sigma, 68.3 %; the
  interval is a setting). It is asymmetric around the measurement, and the acceleration constant of BCa is
  not used (as in the C++ program). Bins where beta is not defined in the measurement (too little intensity)
  have no interval.
* **Speed and liveness.** The run works on a frozen copy of the events, weights and matrices of the published
  result, in a background thread, and publishes its progress while running (the bands grow). Events with
  weight 0 are left out. About 1000 resamples of 100k weighted events take 5 s. The bands are only drawn
  when the settings of the Abel result are those of the run, and the label says when the data has grown
  since. "re-run when the data grew by 10 %" starts a new run by itself.
* **Memory.** All samples are kept (n x bins x (1 + number of betas) numbers, 2000 samples of 200 bins and
  orders 0, 1, 2 are about 10 MB); the number of samples is limited to 50000 in the interface.
* **In code.** `Bootstrap(side, n_boot, unit, n_sigma, seed)` for one run and
  `bias_corrected_interval(estimate, samples, n_sigma)` for the interval itself; `result.resample(weights)`
  of every published Abel result inverts any number of resampled weight vectors without touching the
  live state.

## Centring and circularisation

Centring: project on x and y; the self-convolution of a projection that is symmetric about c
peaks at 2c; the peak is refined with a parabola. Automatic (live) or manual/locked.

Circularisation (on request, never on every update): a stretch factor s per angular bin
(events move r -> s r) is fitted on the polar histogram. The reference slice defines two mirror
planes; the first quadrant is fixed by chaining each bin to its predecessor, all other bins
are matched to their mirror image in that quadrant. Safeguards you can switch off: limit on the
change per bin, optional smoothing onto cos(2n phi) harmonics, fit similarity per bin and a
count of bins stuck at the search limits.

The result is a table s(angle). It is applied to every event's angle (smooth, periodic
interpolation), so it does not depend on the angular binning of any image: you can change the
binning freely, and load a table made with another number of angular bins. What does matter is
the centre (stored in the table; the tab warns if it differs) and the detector.

Saved file (JSON, human readable): format/version, detector, centre, reference angle and sweep
direction, fit range, n_theta, theta_deg[], stretch[], similarity[].
