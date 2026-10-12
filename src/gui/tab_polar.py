"""Tab 3, "Analysis": centring, circularisation and Abel inversion of the polar-binned data.

Left: settings for every operation, each in a collapsible section (click the header). The section
"Regularisation" holds every correction of the Abel inversion (Tikhonov, statistical weights, smoothness,
hard positivity, maximum entropy) and states which are applied.
Right: two pages to switch between (2 x 3 plots each).

  Page "centring and circularisation"
    image after centring (+ circularisation) | angle-integrated radial profile | radial bins d_n(r), after
    centring diagnostics                     | circularisation s(angle)         | radial bins d_n(r), before
  Page "Abel inversion"
    measured (angular binned) image          | reconstructed image              | radial bins d_n(r)
    intensity I(rho)                         | anisotropy beta_n(rho)           | summary

d_n(r) are the radial bins of angular order n (the data of the inversion), by default n = 0, 1, 2.
Order 1 is not expected for a mirror-symmetric distribution, but is kept: it absorbs small asymmetries.

Centring: projections on x and y, self-convolution of each, peak = 2 x centre.
Circularisation: stretch factor per angular bin, estimated on request (never on every
update), saved / loaded as a table s(angle). It acts on each event's angle, so it does not
depend on the angular binning of the images.
"""
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from analysis.polar import CircParams
from gui.plotdata import abel_panel, polar_panel
from gui.widgets import (COLOR, CollapsibleSection, HistPlot, ImagePlot, set_order_bands,
                             set_order_curves, set_quiet, spin)


def _group(title, expanded=True):
    """A collapsible section and its form layout."""
    g = CollapsibleSection(title, expanded)
    return g, g.form


class AnalysisTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self.side = "a"
        self._last = None
        self._busy = False
        self._abel_ready = False
        self._beta_range_set = False

        # ---------------- detector / view ----------------
        self.side_combo = QtWidgets.QComboBox()
        self.events = QtWidgets.QComboBox()
        self.events.addItems(["subtracted", "raw", "all"])
        self.view = QtWidgets.QComboBox()
        self.view.addItems(["polar map (r, angle)", "cartesian"])
        self.side_combo.currentIndexChanged.connect(self._side_changed)
        self.events.currentTextChanged.connect(self._events_changed)
        self.view.currentIndexChanged.connect(lambda *_: self._force())
        g0, f0 = _group("detector and events")
        f0.addRow("detector", self.side_combo)
        f0.addRow("events", self.events)
        f0.addRow("image view", self.view)

        # ---------------- polar binning ----------------
        self.nr = QtWidgets.QSpinBox(); self.nr.setRange(10, 4000); self.nr.setValue(80)
        self.nth = QtWidgets.QSpinBox(); self.nth.setRange(4, 1440); self.nth.setSingleStep(4); self.nth.setValue(80)
        self.rmax = spin(0.005, 10.0, 0.5, step=0.01, decimals=3)
        self.fit_btn = QtWidgets.QPushButton("fit r max to the events")
        for w in (self.nr, self.nth, self.rmax):
            w.valueChanged.connect(lambda *_: self._binning_changed())
        self.fit_btn.clicked.connect(lambda *_: self._fit_rmax())
        g1, f1 = _group("polar binning", expanded=True)
        f1.addRow("radial bins", self.nr)
        f1.addRow("angular bins (x4)", self.nth)
        f1.addRow("r max", self.rmax)
        f1.addRow(self.fit_btn)

        # ---------------- centring ----------------
        self.auto = QtWidgets.QCheckBox("automatic (live estimate)")
        self.auto.setChecked(True)
        self.estimator = QtWidgets.QComboBox()
        self.estimator.addItems(["convolution", "centroid"])
        self.cx = spin(-5.0, 5.0, 0.5, step=0.001, decimals=4); self.cx.setEnabled(False)
        self.cy = spin(-5.0, 5.0, 0.5, step=0.001, decimals=4); self.cy.setEnabled(False)
        self.lock_btn = QtWidgets.QPushButton("lock current centre")
        self.center_label = QtWidgets.QLabel("centre: -")
        self.auto.toggled.connect(self._auto_toggled)
        self.estimator.currentTextChanged.connect(self._estimator_changed)
        self.cx.valueChanged.connect(lambda *_: self._center_typed())
        self.cy.valueChanged.connect(lambda *_: self._center_typed())
        self.lock_btn.clicked.connect(lambda *_: self.auto.setChecked(False))
        g2, f2 = _group("centring", expanded=True)
        f2.addRow(self.auto)
        f2.addRow("method", self.estimator)
        f2.addRow("centre x", self.cx)
        f2.addRow("centre y", self.cy)
        f2.addRow(self.lock_btn)
        f2.addRow(self.center_label)

        # ---------------- circularisation ----------------
        self.circ_apply = QtWidgets.QCheckBox("apply correction")
        self.ref = spin(-360.0, 360.0, 0.0, step=5.0, decimals=1, suffix=" deg")
        self.direction = QtWidgets.QComboBox(); self.direction.addItems(["counter-clockwise", "clockwise"])
        self.fit_lo = spin(0.0, 100.0, 5.0, step=1.0, decimals=0, suffix=" %")
        self.fit_hi = spin(0.0, 100.0, 95.0, step=1.0, decimals=0, suffix=" %")
        self.s_min = spin(0.2, 1.0, 0.8, step=0.05, decimals=2)
        self.s_max = spin(1.0, 5.0, 1.25, step=0.05, decimals=2)
        self.step = spin(0.0, 100.0, 5.0, step=1.0, decimals=1, suffix=" %")
        self.harm = QtWidgets.QSpinBox(); self.harm.setRange(0, 12)
        self.smooth = spin(0.0, 20.0, 1.0, step=0.5, decimals=1, suffix=" bins")
        self.free_amp = QtWidgets.QCheckBox("free amplitude per angle"); self.free_amp.setChecked(True)
        self.compute_btn = QtWidgets.QPushButton("compute from current data")
        self.save_btn = QtWidgets.QPushButton("save...")
        self.load_btn = QtWidgets.QPushButton("load...")
        self.clear_btn = QtWidgets.QPushButton("clear")
        self.circ_label = QtWidgets.QLabel("not computed")
        self.circ_label.setWordWrap(True)
        self.circ_apply.toggled.connect(self._apply_toggled)
        self.compute_btn.clicked.connect(lambda *_: self._compute())
        self.save_btn.clicked.connect(lambda *_: self._save())
        self.load_btn.clicked.connect(lambda *_: self._load())
        self.clear_btn.clicked.connect(lambda *_: self._clear())
        g3, f3 = _group("circularisation (on request)", expanded=False)
        f3.addRow(self.circ_apply)
        f3.addRow("reference slice", self.ref)
        f3.addRow("sweep", self.direction)
        f3.addRow("fit r from", self.fit_lo)
        f3.addRow("fit r to", self.fit_hi)
        f3.addRow("stretch min", self.s_min)
        f3.addRow("stretch max", self.s_max)
        f3.addRow("max step / bin (0 = off)", self.step)
        f3.addRow("harmonics (0 = off)", self.harm)
        f3.addRow("profile smoothing", self.smooth)
        f3.addRow(self.free_amp)
        f3.addRow(self.compute_btn)
        row = QtWidgets.QHBoxLayout()
        for b in (self.save_btn, self.load_btn, self.clear_btn):
            row.addWidget(b)
        f3.addRow(row)
        f3.addRow(self.circ_label)

        # ---------------- Abel inversion ----------------
        self.n_max = QtWidgets.QSpinBox(); self.n_max.setRange(0, 8); self.n_max.setValue(2)
        self.even_only = QtWidgets.QCheckBox("even orders only")
        self.axis = spin(-360.0, 360.0, 90.0, step=5.0, decimals=1, suffix=" deg")
        self.tik = spin(0.0, 10.0, 0.0, step=0.01, decimals=3)
        self.tik_diff = QtWidgets.QSpinBox(); self.tik_diff.setRange(0, 3)
        self.beta_cut = spin(0.0, 100.0, 0.0, step=1.0, decimals=1, suffix=" %")
        # further loss terms (analysis/abel_loss.py): they act on all orders and radii together
        self.weighting = QtWidgets.QCheckBox("statistical weights")
        self.weighting.setChecked(True)
        self.weighting.setToolTip("Weight the data by their estimated covariance: noisy radii and angular orders\n"
                                  "are smoothed more. Only matters together with a penalty or a constraint.")
        self.smooth = spin(0.0, 100.0, 0.0, step=0.05, decimals=3)
        self.smooth.setToolTip("Strength of the smoothness penalty on I(rho) (0 = off); same relative units as the regularisation.")
        self.smooth_kind = QtWidgets.QComboBox()
        self.smooth_kind.addItems(["curvature", "gradient"])
        self.smooth_all = QtWidgets.QCheckBox("also the anisotropy (all Legendre orders)")
        self.positivity = QtWidgets.QComboBox()
        self.positivity.addItems(["off", "I(rho) >= 0", "whole distribution >= 0"])
        self.positivity.setToolTip("Constrained fit: forbids the negative dip in front of a peak. 'whole distribution' also\n"
                                   "requires the angular distribution to be positive at every radius and angle.\n"
                                   "The bootstrap runs the same constrained fit for every resample (slower).")
        self.entropy = spin(0.0, 100.0, 0.0, step=0.05, decimals=3)
        self.entropy.setToolTip("Maximum-entropy (Poisson entropy) term, strength relative to the data term; 0 = off.\n"
                                "Soft positivity: keeps the angular distribution above the default level without a hard\n"
                                "constraint, and is stiff in the weak tails and soft at the peaks, as counting noise is.\n"
                                "Replaces the hard positivity constraint. Bootstrap: one optimisation per resample (slower).")
        self.entropy_floor = spin(0.01, 50.0, 1.0, step=0.1, decimals=2, suffix=" %")
        self.entropy_floor.setToolTip("Default level of the distribution (the value it relaxes to where the data say nothing),\n"
                                      "in percent of its peak.")
        self.reg_label = QtWidgets.QLabel()
        self.reg_label.setWordWrap(True)
        self.reg_label.setStyleSheet("QLabel { font-weight: bold; }")
        self.reg_reset = QtWidgets.QPushButton("switch all corrections off")
        self.abel_label = QtWidgets.QLabel("waiting for data")
        self.abel_label.setWordWrap(True)
        for w in (self.n_max, self.axis, self.tik, self.tik_diff, self.beta_cut, self.smooth,
                  self.entropy, self.entropy_floor):
            w.valueChanged.connect(lambda *_: self._abel_changed())
        self.reg_reset.clicked.connect(lambda *_: self._reg_clear())
        for w in (self.even_only, self.weighting, self.smooth_all):
            w.toggled.connect(lambda *_: self._abel_changed())
        for w in (self.smooth_kind, self.positivity):
            w.currentIndexChanged.connect(lambda *_: self._abel_changed())
        g4, f4 = _group("Abel inversion", expanded=False)
        f4.addRow("highest order n", self.n_max)
        f4.addRow(self.even_only)
        f4.addRow("laser axis angle", self.axis)
        f4.addRow("beta shown above", self.beta_cut)
        f4.addRow(self.abel_label)
        g4.set_content_enabled(False)                    # becomes active when the first result arrives

        # ---------------- regularisation: every correction applied to the inversion ----------------
        def heading(text, tip):
            lab = QtWidgets.QLabel(text)
            lab.setStyleSheet("QLabel { font-weight: bold; padding-top: 6px; }")
            lab.setToolTip(tip)
            return lab
        gr, fr = _group("Regularisation (corrections to the inversion)", expanded=False)
        fr.addRow(self.reg_label)
        fr.addRow(heading("1  Damping (Tikhonov)", "Penalises the size (or roughness) of the solution, order by order."))
        fr.addRow("strength (0 = off)", self.tik)
        fr.addRow("penalises (0 = amplitude)", self.tik_diff)
        fr.addRow(heading("2  Statistical weights", "Weights the data by their estimated noise."))
        fr.addRow(self.weighting)
        fr.addRow(heading("3  Smoothness", "Penalises curvature / gradient of I(rho)."))
        fr.addRow("strength (0 = off)", self.smooth)
        fr.addRow("of the", self.smooth_kind)
        fr.addRow(self.smooth_all)
        fr.addRow(heading("4  Hard positivity", "Constrained fit: the solution may not go negative."))
        fr.addRow("constraint", self.positivity)
        fr.addRow(heading("5  Maximum entropy (Poisson)", "Soft positivity from the entropy of counting statistics."))
        fr.addRow("strength (0 = off)", self.entropy)
        fr.addRow("default level", self.entropy_floor)
        fr.addRow(self.reg_reset)
        gr.set_content_enabled(False)
        self._reg_sync()

        # ---------------- bootstrap ----------------
        self.boot_n = QtWidgets.QSpinBox(); self.boot_n.setRange(50, 50000); self.boot_n.setSingleStep(500)
        self.boot_n.setValue(2000)
        self.boot_unit = QtWidgets.QComboBox()
        self.boot_unit.addItems(["event", "cycle"])
        self.boot_sigma = spin(0.5, 4.0, 1.0, step=0.5, decimals=1, suffix=" sigma")
        self.boot_auto = QtWidgets.QCheckBox("re-run when the data grew by 10 %")
        self.boot_show = QtWidgets.QCheckBox("show the bands")
        self.boot_show.setChecked(True)
        self.boot_run = QtWidgets.QPushButton("run bootstrap")
        self.boot_stop = QtWidgets.QPushButton("stop")
        self.boot_label = QtWidgets.QLabel("no bootstrap yet")
        self.boot_label.setWordWrap(True)
        for w in (self.boot_n, self.boot_sigma):
            w.valueChanged.connect(lambda *_: self._boot_changed())
        self.boot_unit.currentIndexChanged.connect(lambda *_: self._boot_changed())
        self.boot_auto.toggled.connect(lambda *_: self._boot_changed())
        self.boot_show.toggled.connect(lambda *_: self._force())
        self.boot_run.clicked.connect(lambda *_: self._boot_start())
        self.boot_stop.clicked.connect(lambda *_: self._boot_halt())
        g5, f5 = _group("Bootstrap (bias corrected)", expanded=False)
        f5.addRow("samples", self.boot_n)
        f5.addRow("resample", self.boot_unit)
        f5.addRow("interval", self.boot_sigma)
        f5.addRow(self.boot_auto)
        f5.addRow(self.boot_show)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.boot_run)
        row.addWidget(self.boot_stop)
        f5.addRow(row)
        f5.addRow(self.boot_label)
        g5.set_content_enabled(False)
        self.sections = (g0, g1, g2, g3, g4, gr, g5)

        left = QtWidgets.QVBoxLayout()
        for g in (g0, g1, g2, g3, g4, gr, g5):
            left.addWidget(g)
        left.addStretch(1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lw)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(360)

        # ---------------- page 1: centring and circularisation ----------------
        self.p_img = ImagePlot("image", "r", "angle (deg)")
        self.p_prof = HistPlot("angle-integrated radial profile", "radius r", "counts per radial bin", legend=True)
        self.p_prof.add_curve("raw", COLOR["all"], filled=False, label="centred", width=2)
        self.p_prof.add_curve("corr", COLOR["sel"], label="centred + circularised")
        self.p_bins = HistPlot("radial bins d_n(r), centred + circularised", "radius r", "d_n(r)", legend=True)
        self.p_center = HistPlot("centring: projections and mirror images", "distance from centre",
                                 "projection (max = 1)", legend=True)
        self.p_center.add_curve("x", COLOR["sel"], filled=False, label="x projection", width=2)
        self.p_center.add_curve("xm", COLOR["sel"], filled=False, label="x mirrored", dash=True)
        self.p_center.add_curve("y", COLOR["true"], filled=False, label="y projection", width=2)
        self.p_center.add_curve("ym", COLOR["true"], filled=False, label="y mirrored", dash=True)
        self.p_circ = HistPlot("circularisation: stretch s(angle)", "angle (deg)", "s")
        self.p_circ.add_curve("s", COLOR["sel"], filled=False, symbol="o")
        self.planes = []
        for _ in range(4):
            line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(COLOR["acc"], style=QtCore.Qt.PenStyle.DashLine))
            line.setVisible(False)
            self.p_circ.getPlotItem().addItem(line)
            self.planes.append(line)
        self.p_bins_before = HistPlot("radial bins d_n(r), centred only", "radius r", "d_n(r)", legend=True)

        # ---------------- page 2: Abel inversion ----------------
        self.p_img2 = ImagePlot("measured image", "r", "angle (deg)")
        self.p_slice = ImagePlot("reconstructed image", "radius rho", "angle (deg)")
        self.p_bins2 = HistPlot("radial bins d_n(r)", "radius r", "d_n(r)", legend=True)
        self.p_intensity = HistPlot("intensity I(rho)", "radius rho", "counts per unit radius")
        self.p_intensity.add_band("I", COLOR["sel"])
        self.p_intensity.add_curve("I", COLOR["sel"])
        self.p_betas = HistPlot("anisotropy beta_n(rho)", "radius rho", "beta_n", legend=True)
        self.zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(COLOR["all"], style=QtCore.Qt.PenStyle.DashLine))
        self.p_betas.getPlotItem().addItem(self.zero_line)
        self.abel_text = QtWidgets.QPlainTextEdit()
        self.abel_text.setReadOnly(True)
        self.abel_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

        def page(plots):
            w, grid = QtWidgets.QWidget(), QtWidgets.QGridLayout()
            for i, plot in enumerate(plots):
                grid.addWidget(plot, i // 3, i % 3)
            w.setLayout(grid)
            return w
        self.page_center = page((self.p_img, self.p_prof, self.p_bins,
                                 self.p_center, self.p_circ, self.p_bins_before))
        self.page_abel = page((self.p_img2, self.p_slice, self.p_bins2,
                               self.p_intensity, self.p_betas, self.abel_text))
        self.pages = QtWidgets.QTabWidget()
        self.pages.addTab(self.page_center, "Centring and Circularisation")
        self.pages.addTab(self.page_abel, "Abel Inversion")
        self.pages.currentChanged.connect(lambda *_: self._force())
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(self.pages, 1)

    # ------------------------------------------------------------ session
    @property
    def pa(self):
        return self.session.polar if self.session is not None else None

    def attach(self, session):
        self.session, self._last, self.side = session, None, "a"
        c = session.config
        self._busy = True
        self.side_combo.blockSignals(True)
        self.side_combo.clear()
        self.side_combo.addItems([f"A: detector {c.det_a}", f"B: detector {c.det_b}"])
        self.side_combo.setCurrentIndex(0)
        self.side_combo.blockSignals(False)
        self._busy = False
        self.pa.set_mode(self.events.currentText())
        for side in "ab":                                  # push the binning shown in the boxes
            self.pa.set_binning(side, nr=self.nr.value(), ntheta=self.nth.value(), r_max=self.rmax.value())
            self.pa.set_estimator(side, self.estimator.currentText())
            self._push_abel(side)
            self._push_boot(side)
        self._sync_controls()

    def detach(self):
        self.session = None
        self._abel_ready = False
        for i in (4, 5, 6):
            self.sections[i].set_content_enabled(False)

    def _sync_controls(self):
        """Show the settings of the selected side (after switching side / loading)."""
        pa = self.pa
        if pa is None:
            return
        ss = getattr(pa.settings, self.side)
        self._busy = True
        set_quiet(self.nr, int(ss.nr))
        set_quiet(self.nth, int(ss.ntheta))
        set_quiet(self.rmax, ss.r_max)
        for w, on in ((self.auto, ss.center_mode == "auto"), (self.circ_apply, ss.circ_enabled)):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        self.cx.setEnabled(ss.center_mode == "manual")
        self.cy.setEnabled(ss.center_mode == "manual")
        if ss.center_mode == "manual":
            set_quiet(self.cx, ss.center[0])
            set_quiet(self.cy, ss.center[1])
        if self.session is not None and self.session.abel is not None:
            ab = getattr(self.session.abel.settings, self.side)
            set_quiet(self.n_max, ab.n_max)
            self.even_only.blockSignals(True)
            self.even_only.setChecked(ab.even_only)
            self.even_only.blockSignals(False)
            set_quiet(self.axis, ab.axis_angle_deg)
            set_quiet(self.tik, ab.tikhonov)
            set_quiet(self.tik_diff, ab.diff_order)
            set_quiet(self.smooth, ab.smooth)
            set_quiet(self.entropy, ab.entropy)
            set_quiet(self.entropy_floor, ab.entropy_floor * 100.0)
            for w, on in ((self.weighting, ab.weighting), (self.smooth_all, ab.smooth_all)):
                w.blockSignals(True)
                w.setChecked(bool(on))
                w.blockSignals(False)
            for w, i in ((self.smooth_kind, 0 if ab.smooth_order == 2 else 1),
                         (self.positivity, ("off", "intensity", "distribution").index(ab.positivity))):
                w.blockSignals(True)
                w.setCurrentIndex(i)
                w.blockSignals(False)
            set_quiet(self.beta_cut, ab.min_intensity * 100.0)
            self._reg_sync()
            bs = self.session.abel.bootstrap_settings(self.side)
            set_quiet(self.boot_n, bs.n_boot)
            set_quiet(self.boot_sigma, bs.n_sigma)
            self.boot_unit.blockSignals(True)
            self.boot_unit.setCurrentIndex(0 if bs.unit == "event" else 1)
            self.boot_unit.blockSignals(False)
            self.boot_auto.blockSignals(True)
            self.boot_auto.setChecked(bs.auto)
            self.boot_auto.blockSignals(False)
        self._busy = False

    def _force(self):
        self._last = None

    # ------------------------------------------------------------ controls
    def _side_changed(self, i):
        if self._busy:
            return
        self.side = "a" if i == 0 else "b"
        self._sync_controls()
        self._force()

    def _events_changed(self, text):
        if self.pa is not None:
            self.pa.set_mode(text)

    def _binning_changed(self):
        if self._busy or self.pa is None:
            return
        self.pa.set_binning(self.side, nr=self.nr.value(), ntheta=self.nth.value(), r_max=self.rmax.value())

    def _fit_rmax(self):
        if self.pa is None:
            return
        try:
            self.pa.fit_r_max(self.side)
        except RuntimeError:
            return                                          # no data yet
        self._sync_controls()

    def _auto_toggled(self, on):
        if self._busy or self.pa is None:
            return
        if on:
            self.pa.set_center_mode(self.side, "auto")
        else:
            self.pa.lock_center(self.side)                  # freeze the centre in use
            self.pa.set_center_mode(self.side, "manual")
        self._sync_controls()

    def _estimator_changed(self, text):
        if self.pa is not None:
            self.pa.set_estimator(self.side, text)

    def _center_typed(self):
        if self._busy or self.pa is None or self.auto.isChecked():
            return
        self.pa.set_center(self.side, self.cx.value(), self.cy.value())

    def _push_abel(self, side):
        ab = self.session.abel
        ab.set_orders(side, n_max=self.n_max.value(), even_only=self.even_only.isChecked())
        ab.set_axis_angle(side, self.axis.value())
        ab.set_regularisation(side, self.tik.value(), self.tik_diff.value())
        ab.set_loss(side, weighting=self.weighting.isChecked(), smooth=self.smooth.value(),
                    smooth_order=2 if self.smooth_kind.currentIndex() == 0 else 1,
                    smooth_all=self.smooth_all.isChecked(),
                    positivity=("off" if self.entropy.value() > 0
                                else ("off", "intensity", "distribution")[max(self.positivity.currentIndex(), 0)]),
                    entropy=self.entropy.value(), entropy_floor=self.entropy_floor.value() / 100.0)
        ab.set_min_intensity(side, self.beta_cut.value() / 100.0)

    def _abel_changed(self):
        self._reg_sync()
        if self._busy or self.session is None or self.session.abel is None:
            return
        self._push_abel(self.side)

    def _reg_sync(self):
        """Hard positivity and maximum entropy exclude each other; say in plain words what is applied."""
        ent, pos = self.entropy.value() > 0, self.positivity.currentIndex() > 0
        self.positivity.setEnabled(not ent)
        self.entropy.setEnabled(not pos)
        self.entropy_floor.setEnabled(not pos and ent)
        parts = []
        if self.tik.value() > 0:
            parts.append(f"Tikhonov damping {self.tik.value():g}"
                         + ("" if self.tik_diff.value() == 0 else f" (of the {self.tik_diff.value()}. difference)"))
        if self.weighting.isChecked():
            parts.append("statistical weights")
        if self.smooth.value() > 0:
            parts.append(f"{self.smooth_kind.currentText()} smoothness {self.smooth.value():g}"
                         + (" (all orders)" if self.smooth_all.isChecked() else " (I(rho) only)"))
        if pos:
            parts.append("hard positivity: " + self.positivity.currentText())
        if ent:
            parts.append(f"maximum entropy {self.entropy.value():g} (default level {self.entropy_floor.value():g} %)")
        self.reg_label.setText("Applied: " + (" + ".join(parts) if parts else
                                              "nothing - exact inverse (noise is amplified; ringing possible)"))

    def _reg_clear(self):
        for w in (self.tik, self.smooth, self.entropy):
            w.blockSignals(True); w.setValue(0.0); w.blockSignals(False)
        self.tik_diff.blockSignals(True); self.tik_diff.setValue(0); self.tik_diff.blockSignals(False)
        for w in (self.weighting, self.smooth_all):
            w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
        self.positivity.blockSignals(True); self.positivity.setCurrentIndex(0); self.positivity.blockSignals(False)
        self._abel_changed()

    def _push_boot(self, side):
        self.session.abel.set_bootstrap(side, n_boot=self.boot_n.value(),
                                        unit=self.boot_unit.currentText(),
                                        n_sigma=self.boot_sigma.value(), auto=self.boot_auto.isChecked())

    def _boot_changed(self):
        if self._busy or self.session is None or self.session.abel is None:
            return
        self._push_boot(self.side)

    def _boot_start(self):
        if self.session is None or self.session.abel is None:
            return
        self._push_boot(self.side)
        try:
            run = self.session.abel.start_bootstrap(self.side)
        except (ValueError, RuntimeError) as exc:
            self.boot_label.setText(str(exc))
            return
        if run is None:
            self.boot_label.setText("no inversion to resample yet")
        self._force()

    def _boot_halt(self):
        if self.session is not None and self.session.abel is not None:
            self.session.abel.stop_bootstrap(self.side)

    def _apply_toggled(self, on):
        if not self._busy and self.pa is not None:
            self.pa.enable_circ(self.side, on)

    def _params(self):
        rmax = getattr(self.pa.settings, self.side).r_max
        return CircParams(theta_ref_deg=self.ref.value(),
                          direction=1 if self.direction.currentIndex() == 0 else -1,
                          r_lo=self.fit_lo.value() / 100.0 * rmax, r_hi=self.fit_hi.value() / 100.0 * rmax,
                          s_min=self.s_min.value(), s_max=self.s_max.value(),
                          max_step=None if self.step.value() <= 0 else self.step.value() / 100.0,
                          harmonics=int(self.harm.value()), smooth_bins=self.smooth.value(),
                          free_amplitude=self.free_amp.isChecked())

    def _compute(self):
        if self.pa is None:
            return
        try:
            self.pa.compute_circularisation(self.side, self._params())
        except RuntimeError as exc:
            self.circ_label.setText(str(exc))

    # file dialogs are separate methods so that they can be replaced in tests
    def _ask_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "save circularisation",
                                                        "circularisation.json", "JSON (*.json)")
        return path or None

    def _ask_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "load circularisation", "", "JSON (*.json)")
        return path or None

    def _save(self):
        path = self._ask_save() if self.pa is not None else None
        if path:
            try:
                self.pa.save_circ(self.side, path)
                self.circ_label.setText(f"saved to {path}")
            except Exception as exc:
                self.circ_label.setText(f"save failed: {exc}")

    def _load(self):
        path = self._ask_open() if self.pa is not None else None
        if not path:
            return
        try:
            warns = self.pa.load_circ(self.side, path)
        except Exception as exc:
            self.circ_label.setText(f"load failed: {exc}")
            return
        self._sync_controls()
        self._force()
        self.circ_label.setText(self.pa.status(self.side) + ("\n" + "\n".join(warns) if warns else ""))

    def _clear(self):
        if self.pa is not None:
            self.pa.clear_circ(self.side)
            self._sync_controls()

    # ------------------------------------------------------------ refresh
    def refresh(self, force=False):
        pa = self.pa
        if pa is None:
            return
        res = pa.result()
        if res is None:
            return
        ares = self.session.abel_result()
        on_abel = self.pages.currentWidget() is self.page_abel
        view = "cartesian" if self.view.currentIndex() == 1 else "polar"
        boot = self.session.abel.bootstrap_result(self.side) if self.session.abel else None
        key = (res.version, ares.version if ares else None, self.side, view, pa.status(self.side),
               pa.settings.version, on_abel, None if boot is None else (id(boot), boot.n_done, boot.running),
               self.boot_show.isChecked())
        if key == self._last and not force:
            return
        self._last = key
        p = polar_panel(res, self.side, view)
        if p is None:
            return
        ap = (abel_panel(ares, self.side, res[self.side].binning, view, boot=boot)
              if ares is not None else None)
        if ap is not None and not self._abel_ready:
            self._abel_ready = True
            for i in (4, 5, 6):                             # the Abel controls are live from now on
                self.sections[i].set_content_enabled(True)
        if on_abel:
            self._draw_abel_page(p, ap)
        else:
            self._draw_center_page(p, ap)
        self._update_labels(p, ap)

    def _draw_center_page(self, p, ap):
        self.p_img.set_image(p.image, p.extent, p.levels, title=f"detector {p.det}: {p.title}",
                             aspect=p.aspect_locked, xlabel=p.xlabel, ylabel=p.ylabel)
        self.p_prof.set_curve("raw", p.prof_raw)
        self.p_prof.set_curve("corr", p.prof_corr)
        for name, h in (("x", p.cx_proj), ("xm", p.cx_mirror), ("y", p.cy_proj), ("ym", p.cy_mirror)):
            self.p_center.set_curve(name, h)
        self.p_circ.curves["s"].setData(p.circ_theta_deg, p.circ_s)
        for i, line in enumerate(self.planes):
            line.setVisible(i < len(p.mirror_planes_deg))
            if i < len(p.mirror_planes_deg):
                line.setValue(p.mirror_planes_deg[i])
        if ap is not None:
            set_order_curves(self.p_bins, ap.bins_after)
            before = ap.bins_before if ap.bins_before is not None else ap.bins_after
            set_order_curves(self.p_bins_before, before)
            self.p_bins_before.getPlotItem().setTitle(
                "radial bins d_n(r), centred only" if ap.bins_before is not None else
                "radial bins d_n(r), centred only (= above: not circularised)", size="9pt")

    def _draw_abel_page(self, p, ap):
        self.p_img2.set_image(p.image, p.extent, p.levels, title=f"measured: detector {p.det}, {p.title}",
                              aspect=p.aspect_locked, xlabel=p.xlabel, ylabel=p.ylabel)
        if ap is None:
            return
        set_order_curves(self.p_bins2, ap.bins_after)
        if ap.slice_image is not None:
            self.p_slice.set_image(ap.slice_image, ap.slice_extent, ap.slice_levels,
                                   title="reconstructed: distribution in the plane of the laser axis",
                                   aspect=ap.slice_aspect, xlabel=ap.slice_xlabel, ylabel=ap.slice_ylabel)
            self.p_intensity.set_curve("I", ap.intensity)
            show = self.boot_show.isChecked()
            self.p_intensity.set_band("I", ap.intensity_band if show else None)
            set_order_curves(self.p_betas, ap.betas, key="b", label="beta_{n}")
            set_order_bands(self.p_betas, ap.beta_bands if show else (), key="b")
            if not self._beta_range_set:                    # a sensible window once; the user may zoom
                self.p_betas.getPlotItem().setYRange(-2.0, 3.0)
                self._beta_range_set = True
        self.abel_text.setPlainText(ap.text)

    def _update_labels(self, p, ap):
        pa = self.pa
        est = f"({p.center_auto[0]:.4f}, {p.center_auto[1]:.4f})" if p.center_auto else "-"
        self.center_label.setText(f"in use ({p.center[0]:.4f}, {p.center[1]:.4f}) [{p.center_mode}]\n"
                                  f"automatic estimate {est}")
        if p.center_mode == "auto":
            self._busy = True
            set_quiet(self.cx, p.center[0])
            set_quiet(self.cy, p.center[1])
            self._busy = False
        self.circ_label.setText(pa.status(self.side) + (("\n" + "\n".join(p.warnings)) if p.warnings else ""))
        on = getattr(pa.settings, self.side).circ_enabled          # a finished computation switches it on
        if self.circ_apply.isChecked() != on:
            self.circ_apply.blockSignals(True)
            self.circ_apply.setChecked(on)
            self.circ_apply.blockSignals(False)
        if ap is not None:
            self.abel_label.setText(ap.status)
            self.boot_label.setText(ap.boot_text)


PolarTab = AnalysisTab          # the name used before the tab was renamed
