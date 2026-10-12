"""Tab 2: coincidence analysis.

Left: controls. Right: two pages to switch between.
 Page "timing": six panels
  dt overview | dt zoom (drag the window here) | window vs control windows
  pair rate vs time | clock offset vs time | text summary
 Page "Position Correlation": one plot per detector, x (left axis) and y (right axis) of the hit
  position against dt, as binned means with standard errors plus a subsample of the raw pairs.

dt = t_B - t_A [us]. The main window is blue; control windows (the same window
moved by whole periods of the beam structure) are green. Their counts estimate the
accidental coincidences inside the main window.
"""
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtGui, QtWidgets

from analysis.imaging import CartesianBinning
from analysis.poscorr import (apply_walk, correction_images, cut_pairs, dt_profile, fit_correction, fit_walk,
                                 position_correlation)
from gui.plotdata import EMPTY, Hist, coincidence_panel
from gui.widgets import COLOR, CollapsibleSection, DualAxisPlot, HistPlot, ImagePlot, fill, new_plot, set_quiet, spin


class CoincidenceTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self._last = None
        self._busy = False
        self._pos_t, self._pos_dur = 0.0, 0.0

        # ---------------- controls (left) ----------------
        # search: need a full re-search
        self.search = spin(100, 1e6, 2500.0, step=100, decimals=0, suffix=" us")
        self.shift_auto = QtWidgets.QCheckBox("auto")
        self.shift_auto.setChecked(True)
        self.shift = QtWidgets.QSpinBox()
        self.shift.setRange(-1000, 1000)
        self.shift.setEnabled(False)
        self.apply_btn = QtWidgets.QPushButton("apply and re-search")
        self.shift_auto.toggled.connect(lambda on: self.shift.setEnabled(not on))
        self.apply_btn.clicked.connect(lambda *_: self._restart())
        g1 = CollapsibleSection("search (re-searches all data)", False)
        f1 = g1.form
        f1.addRow("search window +-", self.search)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.shift_auto)
        row.addWidget(self.shift)
        f1.addRow("cycle shift B-A", row)
        f1.addRow(self.apply_btn)

        # beam structure / accidentals: live
        self.period = spin(1.0, 1e7, 1000.0, step=1.0, decimals=2, suffix=" us")
        self.shots = QtWidgets.QComboBox()
        self.shots.addItems(["+-1, +-2", "+-1"])
        self.period.valueChanged.connect(self._period_changed)
        self.shots.currentIndexChanged.connect(self._shots_changed)
        g2 = CollapsibleSection("accidentals (live)", True)
        f2 = g2.form
        f2.addRow("nominal period", self.period)
        f2.addRow("control windows", self.shots)
        self.period_label = QtWidgets.QLabel("measured period: -")
        f2.addRow(self.period_label)

        # window
        self.auto = QtWidgets.QCheckBox("automatic window")
        self.auto.setChecked(True)
        self.sigma = spin(0.5, 50.0, 1.5, step=0.5, decimals=1, suffix=" sigma")
        self.tmin = spin(-1e6, 1e6, -10.0, step=0.5, decimals=2, suffix=" us")
        self.tmax = spin(-1e6, 1e6, 10.0, step=0.5, decimals=2, suffix=" us")
        self.tmin.setEnabled(False)
        self.tmax.setEnabled(False)
        self.auto.toggled.connect(self._auto_toggled)
        self.sigma.valueChanged.connect(self._sigma_changed)
        self.tmin.valueChanged.connect(lambda v: self._typed(lo=v))
        self.tmax.valueChanged.connect(lambda v: self._typed(hi=v))
        g3 = CollapsibleSection("coincidence window  (dt = t_B - t_A)", True)
        f3 = g3.form
        f3.addRow(self.auto)
        f3.addRow("auto half-width", self.sigma)
        f3.addRow("min", self.tmin)
        f3.addRow("max", self.tmax)

        self.log = QtWidgets.QCheckBox("log scale (overview)")
        self.log.toggled.connect(self._log_toggled)

        # position correlation (live)
        self.pc_half = spin(1.0, 2500.0, 20.0, step=1.0, decimals=1, suffix=" us")
        self.pc_bin = spin(0.05, 100.0, 0.5, step=0.1, decimals=2, suffix=" us")
        self.pc_bkg = QtWidgets.QCheckBox("subtract accidentals in the means")
        self.pc_bkg.setChecked(True)
        self.pc_bkg.toggled.connect(lambda *_: self._pc_force())
        self.pc_enable = QtWidgets.QCheckBox("enable position correlation")
        self.pc_enable.setChecked(False)
        self.pc_enable.setToolTip("Master switch of the Position Correlation page and of passing the correction on to the "
                                  "Analysis tab. Off: nothing of it is computed.")
        self.pc_enable.toggled.connect(lambda *_: self._pc_enabled())
        self.pc_cuts = QtWidgets.QCheckBox("apply Sum / time cuts of the Detectors tab")
        self.pc_cuts.setChecked(True)
        self.pc_cuts.setToolTip("Only pairs whose two events pass the charge-sum (and time) limits are used "
                                "for the correlation, the correction, the images and the walk fit.")
        self.pc_cuts.toggled.connect(lambda *_: self._pc_force())
        self.pc_raw = QtWidgets.QCheckBox("show individual pairs")
        self.pc_raw.setChecked(False)
        self.pc_window = QtWidgets.QCheckBox("mark coincidence window")
        self.pc_window.setChecked(True)
        for w in (self.pc_half, self.pc_bin):
            w.valueChanged.connect(lambda *_: self._pc_force())
        self.pc_raw.toggled.connect(lambda *_: self._pc_style())
        self.pc_window.toggled.connect(lambda *_: self._pc_style())
        # correction of the electron position by the neutral position and dt (fitted inside the window)
        self.pc_electron = QtWidgets.QComboBox()
        self.pc_electron.addItems(["A", "B"])
        self.pc_use_n = QtWidgets.QCheckBox("neutral position (M)")
        self.pc_use_n.setChecked(True)
        self.pc_order = QtWidgets.QSpinBox()                  # polynomial order of the dt term (0 = none)
        self.pc_order.setRange(0, 6)
        self.pc_order.setValue(2)
        self.pc_order.valueChanged.connect(lambda *_: self._pc_force())
        self.pc_pass = QtWidgets.QCheckBox("pass the correction on to the Analysis tab")
        self.pc_pass.setToolTip("Runs the electron correction (and the walk correction and Sum/time cuts set here) in the "
                                "pipeline: the electron positions are displaced before centring, circularisation and the Abel "
                                "inversion. Rows without a partner in the window or a control window are not displaced.")
        self.pc_pass.toggled.connect(lambda *_: self._pc_force())
        self.pc_corrected = QtWidgets.QCheckBox("Show corrected position (dashed)")
        self.pc_corrected.setChecked(True)
        self.pc_weighted = QtWidgets.QCheckBox("weighted fit (subtract accidentals)")
        self.pc_weighted.setChecked(True)
        self.pc_slices = QtWidgets.QSpinBox()
        self.pc_slices.setRange(1, 6)
        self.pc_slices.setValue(1)
        self.pc_slices.valueChanged.connect(lambda *_: self._pc_force())
        self.pc_view = QtWidgets.QSpinBox()                   # which dt selection the two images show
        self.pc_view.setRange(0, 3)
        self.pc_view.setSpecialValueText("full window")
        self.pc_view.valueChanged.connect(lambda *_: self._render_images())
        self._ci = self._cor = None
        self.pc_electron.currentIndexChanged.connect(lambda *_: self._pc_force())
        for w in (self.pc_use_n, self.pc_corrected, self.pc_weighted):
            w.toggled.connect(lambda *_: self._pc_force())
        g4 = CollapsibleSection("position correlation (live)", False)
        f4 = g4.form
        f4.addRow("dt range: peak +-", self.pc_half)
        f4.addRow("bin width", self.pc_bin)
        f4.addRow(self.pc_cuts)
        f4.addRow(self.pc_bkg)
        f4.addRow(self.pc_raw)
        f4.addRow(self.pc_window)
        # timing walk: dt' = dt - linear function of the hit positions (fitted inside the peak)
        self.pc_walk = QtWidgets.QCheckBox("correct dt for the hit positions")
        self.pc_walk.setChecked(True)
        self.pc_walk_sigma = spin(0.5, 10.0, 3.0, step=0.5, decimals=1, suffix=" sigma")
        self.pc_walk_n = QtWidgets.QCheckBox("include the neutral detector's position")
        self.pc_walk_n.setChecked(True)
        for w in (self.pc_walk, self.pc_walk_n):
            w.toggled.connect(lambda *_: self._pc_force())
        self.pc_walk_sigma.valueChanged.connect(lambda *_: self._pc_force())
        g6 = CollapsibleSection("timing walk (dt' = dt - k.position)", False)
        f6 = g6.form
        f6.addRow(self.pc_walk)
        f6.addRow("fit inside peak +-", self.pc_walk_sigma)
        f6.addRow(self.pc_walk_n)
        g5 = CollapsibleSection("electron correction (fit inside the window)", False)
        f5 = g5.form
        f5.addRow("electron detector", self.pc_electron)
        f5.addRow("correct for", self.pc_use_n)
        f5.addRow("dt polynomial order", self.pc_order)
        f5.addRow(self.pc_weighted)
        f5.addRow(self.pc_corrected)
        f5.addRow(self.pc_pass)
        f5.addRow("dt slices of the window", self.pc_slices)
        f5.addRow("show images for slice", self.pc_view)

        self._pc_sections = (g4, g6, g5)
        self._pc_enabled()                                   # off by default: sections greyed, nothing computed
        left = QtWidgets.QVBoxLayout()
        for g in (g1, g2, g3, self.pc_enable, g4, g6, g5):
            left.addWidget(g)
        left.addWidget(self.log)
        left.addStretch(1)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(300)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(lw)
        left_scroll.setMaximumWidth(330)

        # ---------------- plots (right) ----------------
        self.p_full = HistPlot("dt, full search window", "dt = t_B - t_A (us)")
        self.p_full.add_curve("pairs", COLOR["sel"], filled=False)
        self.full_main = self._region(self.p_full, COLOR["main"])
        self.full_ctrl = [self._region(self.p_full, COLOR["acc"]) for _ in range(4)]

        self.p_zoom = HistPlot("dt, zoom: drag the window", "dt (us)")
        self.p_zoom.add_curve("pairs", COLOR["sel"])
        self.zoom_region = pg.LinearRegionItem(values=(-1, 1), brush=fill(COLOR["main"], 50))
        self.p_zoom.getPlotItem().addItem(self.zoom_region)
        self.zoom_region.sigRegionChanged.connect(self._zoom_dragged)

        self.p_bars = new_plot("window vs control windows", "", "pairs")
        self.bars = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush=COLOR["main"])
        self.acc_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(COLOR["acc"], width=2))
        self.p_bars.getPlotItem().addItem(self.bars)
        self.p_bars.getPlotItem().addItem(self.acc_line)

        self.p_rate = HistPlot("pair rate (stationarity check)", "time (s)", "pairs / s", legend=True)
        self.p_rate.add_curve("main", COLOR["main"], filled=False, label="window", width=2)
        self.p_rate.add_curve("acc", COLOR["acc"], filled=False, label="control windows (mean)", width=2)

        self.p_clock = HistPlot("clock offset B-A, residual to linear drift", "time (s)", "us")
        self.p_clock.add_curve("res", COLOR["true"], filled=False)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

        self.page_timing = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.page_timing)
        for i, w in enumerate((self.p_full, self.p_zoom, self.p_bars, self.p_rate, self.p_clock, self.text)):
            grid.addWidget(w, i // 3, i % 3)

        # ---------------- position correlation page ----------------
        self.pc_plots, self.pc_regions = {}, {}
        self.page_pos = QtWidgets.QWidget()
        pos_lay = QtWidgets.QHBoxLayout(self.page_pos)
        dual = QtWidgets.QVBoxLayout()
        for side in ("a", "b"):
            p = DualAxisPlot(f"detector {side.upper()}", "dt = t_B - t_A (us)", "x", "y",
                             color_left=COLOR["main"], color_right=COLOR["true"])
            p.add_series("raw_x", "left", points=True)
            p.add_series("raw_y", "right", points=True)
            p.add_series("x", "left")
            p.add_series("y", "right")
            p.add_series("xc", "left", dashed=True)           # the corrected means, next to the others
            p.add_series("yc", "right", dashed=True)
            p.add_series("xs", "left", line=True)             # the fitted dt shape (dt polynomial order >= 1)
            p.add_series("ys", "right", line=True)
            r = pg.LinearRegionItem(values=(0, 0), movable=False, brush=fill(COLOR["acc"], 35))
            r.setVisible(False)
            p.getPlotItem().addItem(r)
            self.pc_plots[side], self.pc_regions[side] = p, r
            dual.addWidget(p)
        self.pc_plots["b"].getPlotItem().setXLink(self.pc_plots["a"].getPlotItem())   # same dt axis
        # right column: electron images (fixed detector grid, coordinates x1000) before / after the correction
        # for the whole window, the corrected image for each slice of dt, and the fit
        self.pc_img_before = ImagePlot("electron image, uncorrected", "x (x1000)", "y (x1000)")
        self.pc_img_after = ImagePlot("electron image, corrected", "x (x1000)", "y (x1000)")
        self.pc_text = QtWidgets.QPlainTextEdit()
        self.pc_text.setReadOnly(True)
        self.pc_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.pc_text.setMaximumHeight(190)
        corr = QtWidgets.QVBoxLayout()
        corr.addWidget(self.pc_img_before, 1)
        corr.addWidget(self.pc_img_after, 1)
        corr.addWidget(self.pc_text)
        pos_lay.addLayout(dual, 3)
        pos_lay.addLayout(corr, 2)

        self.pages = QtWidgets.QTabWidget()
        self.pages.addTab(self.page_timing, "Timing")
        self.pages.addTab(self.page_pos, "Position Correlation")
        self.pages.currentChanged.connect(lambda *_: self.refresh(force=True))
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(left_scroll)
        lay.addWidget(self.pages, 1)

    @staticmethod
    def _region(plot, color):
        r = pg.LinearRegionItem(values=(0, 0), movable=False, brush=fill(color, 45))
        r.setVisible(False)
        plot.getPlotItem().addItem(r)
        return r

    # -- session -------------------------------------------------------------
    def attach(self, session):
        self.session, self._last = session, None
        self._push_pos()
        self._period_changed(self.period.value())
        self._shots_changed(self.shots.currentIndex())
        self._sigma_changed(self.sigma.value())
        if self.auto.isChecked():
            session.flt.clear_window()
        else:
            session.flt.set_window(self.tmin.value(), self.tmax.value())

    def detach(self):
        self.session = None

    @property
    def flt(self):
        return self.session.flt if self.session is not None else None

    # -- controls ------------------------------------------------------------
    def _restart(self):
        if self.session is None:
            return
        shift = None if self.shift_auto.isChecked() else int(self.shift.value())
        self.session.restart_coincidence(search_window_us=self.search.value(), cycle_shift=shift)
        self._last = None

    def _period_changed(self, v):
        if self.flt is not None:
            self.flt.set_shot_period(v)

    def _shots_changed(self, i):
        if self.flt is not None:
            self.flt.set_bkg_shots((-2, -1, 1, 2) if i == 0 else (-1, 1))

    def _sigma_changed(self, v):
        if self.flt is not None:
            self.flt.set_auto_sigma(v)

    def _log_toggled(self, on):
        self.p_full.set_log(on)

    def _auto_toggled(self, on):
        for s in (self.tmin, self.tmax):
            s.setEnabled(not on)
        if self.flt is None:
            return
        if on:
            self.flt.clear_window()
        else:                                   # freeze what is currently shown
            self.flt.set_window(self.tmin.value(), self.tmax.value())

    def _typed(self, lo=None, hi=None):
        if self._busy or self.flt is None or self.auto.isChecked():
            return
        a = self.tmin.value() if lo is None else lo
        b = self.tmax.value() if hi is None else hi
        self.flt.set_window(min(a, b), max(a, b))

    def _zoom_dragged(self, *_):
        if self._busy or self.flt is None:
            return
        lo, hi = self.zoom_region.getRegion()
        self._busy = True
        if self.auto.isChecked():
            self.auto.blockSignals(True)
            self.auto.setChecked(False)         # dragging means: take manual control
            self.auto.blockSignals(False)
            self.tmin.setEnabled(True)
            self.tmax.setEnabled(True)
        set_quiet(self.tmin, lo)
        set_quiet(self.tmax, hi)
        self._busy = False
        self.flt.set_window(lo, hi)

    # -- position correlation --------------------------------------------------
    def _pc_enabled(self):
        """Master switch: the sections and the pass-on checkbox only matter while it is on."""
        on = self.pc_enable.isChecked()
        self.pc_pass.setEnabled(on)
        for g in self._pc_sections:
            g.set_content_enabled(on)
        self._pc_force()

    def _pos_off(self):
        """Position correlation switched off: leave the page empty and compute nothing."""
        for p in self.pc_plots.values():
            for n in ("raw_x", "raw_y", "x", "y", "xc", "yc", "xs", "ys"):
                p.set_series(n, [], [])
            p.getPlotItem().setTitle("position correlation is off (tick 'enable position correlation')", size="9pt")
        self._ci = self._cor = None
        self.pc_text.setPlainText("position correlation is off: nothing is computed on this page and nothing is "
                                  "passed on to the Analysis tab.")

    def _push_pos(self):
        """Hand the correction settings to the pipeline stage (PositionCorrector)."""
        s = self.session
        if s is None or getattr(s, "pos", None) is None:
            return
        s.pos.configure(enabled=self.pc_enable.isChecked() and self.pc_pass.isChecked(), side_e="a" if self.pc_electron.currentIndex() == 0 else "b",
                        use_neutral=self.pc_use_n.isChecked(), tau_order=self.pc_order.value(),
                        background=self.pc_weighted.isChecked(), walk=self.pc_walk.isChecked(),
                        walk_sigma=self.pc_walk_sigma.value(), walk_neutral=self.pc_walk_n.isChecked(),
                        cuts=self.pc_cuts.isChecked())

    def _pc_force(self):
        self._push_pos()
        self._last = None
        self.refresh()

    def _pc_style(self):
        for p in self.pc_plots.values():
            for n in ("raw_x", "raw_y"):
                p.set_visible(n, self.pc_raw.isChecked())
        for r in self.pc_regions.values():
            lo, hi = r.getRegion()
            r.setVisible(bool(self.pc_window.isChecked() and hi > lo))

    def _refresh_pos(self, res):
        s = self.session
        snap = s.ds.snapshot() if s.ds is not None else None
        da, db = s.flt.det_a, s.flt.det_b
        side_e = "a" if self.pc_electron.currentIndex() == 0 else "b"
        self._cut_note = ""
        if self.pc_cuts.isChecked():
            res, n0, n1 = cut_pairs(res, s.selection())
            if n1 != n0:
                self._cut_note = f"Sum/time cuts: {n1} of {n0} pairs kept\n\n"
        walk = None
        if self.pc_walk.isChecked():                       # everything below then lives on the dt' axis
            walk = fit_walk(res, snap, da, db, n_sigma=self.pc_walk_sigma.value(),
                            use_neutral=self.pc_walk_n.isChecked(), side_e=side_e)
            if walk is not None:
                res = apply_walk(res, snap, walk)
        label = "dt' = dt - walk(position) (us)" if walk is not None else "dt = t_B - t_A (us)"
        for p in self.pc_plots.values():
            p.getPlotItem().setLabel("bottom", label)
        cor = fit_correction(res, snap, da, db, side_e=side_e, use_neutral=self.pc_use_n.isChecked(),
                             tau_order=self.pc_order.value(), background=self.pc_weighted.isChecked())
        self._refresh_correction(res, snap, cor, walk)
        show_c = cor is not None and self.pc_corrected.isChecked()
        pc = position_correlation(res, snap, da, db, half_us=self.pc_half.value(), bin_us=self.pc_bin.value(),
                                  correction=cor if show_c else None)
        tag = ", dashed = corrected" if show_c else ""
        bkg = self.pc_bkg.isChecked()
        prof = dt_profile(res, snap, da, db, half_us=self.pc_half.value(), bin_us=self.pc_bin.value(),
                          correction=cor if show_c else None) if bkg else None
        if bkg:
            tag += ", accidentals subtracted"
        for side, p in self.pc_plots.items():
            r = pc[side] if pc else None
            if r is None:
                for n in ("raw_x", "raw_y", "x", "y", "xc", "yc", "xs", "ys"):
                    p.set_series(n, [], [])
                p.getPlotItem().setTitle(f"detector {side.upper()}: no data", size="9pt")
                continue
            p.set_series("raw_x", r.raw_dt, r.raw_x)
            p.set_series("raw_y", r.raw_dt, r.raw_y)
            q = prof[side] if prof else None                    # accidental-subtracted means, same bins
            if q is not None:
                p.set_series("x", q.centers, q.mean_x, err=q.err_x)
                p.set_series("y", q.centers, q.mean_y, err=q.err_y)
            else:
                p.set_series("x", r.centers, r.mean_x, err=r.sem_x)
                p.set_series("y", r.centers, r.mean_y, err=r.sem_y)
            if q is not None and q.corr_mean_x is not None:
                p.set_series("xc", q.centers, q.corr_mean_x, err=q.corr_err_x)
                p.set_series("yc", q.centers, q.corr_mean_y, err=q.corr_err_y)
            elif q is None and r.corr_mean_x is not None:
                p.set_series("xc", r.centers, r.corr_mean_x, err=r.corr_sem_x)
                p.set_series("yc", r.centers, r.corr_mean_y, err=r.corr_sem_y)
            else:
                p.set_series("xc", [], [])
                p.set_series("yc", [], [])
            if show_c and cor.side_e == side and cor.tau_order >= 1:
                tt = np.linspace(cor.window[0], cor.window[1], 120)
                xs, ys = cor.shape(tt)
                p.set_series("xs", tt, xs)
                p.set_series("ys", tt, ys)
            else:
                p.set_series("xs", [], [])
                p.set_series("ys", [], [])
            e = " electron" if cor is not None and cor.side_e == side else (" neutral" if cor is not None else "")
            p.getPlotItem().setTitle(
                f"detector {r.det} ({side.upper()}{e}{tag}): x (left) and y (right) vs dt;  "
                f"{r.n_valid} pairs in {r.lo:.1f}..{r.hi:.1f} us", size="9pt")
            if res.window_us is not None:
                self.pc_regions[side].setRegion(res.window_us)
        self._pc_style()

    def _image_binning(self, det):
        """The binning of the Detectors-tab image of this detector (so both tabs show the same grid)."""
        try:
            img = self.session.imaging()
            b = img[det].binning
            if hasattr(b, "x_range") and hasattr(b, "nx"):
                return b
        except Exception:
            pass
        return CartesianBinning(256, 256)

    @staticmethod
    def _show(plot, image, extent, title):
        """Fixed grid, coordinates x1000, colour scale 0 .. 99.8th percentile: grows in place as data come in."""
        pos = image[image > 0]
        top = float(np.percentile(pos, 99.8)) if pos.size else 1.0
        plot.set_image(image, tuple(1000.0 * v for v in extent), (0.0, max(top, 1e-9)), title=title)

    def _render_images(self):
        """The two images for the selected dt selection (0 = the whole window, i = slice i)."""
        ci, cor = self._ci, self._cor
        if ci is None or cor is None:
            return
        i = min(self.pc_view.value(), len(ci.slices))
        if i == 0:
            raw, corrected, what = ci.raw, ci.corrected, "full window"
            std_b, std_a = cor.std_before, cor.std_after
            tb = f"detector {cor.det_e}, uncorrected, {what}: std x {1000 * std_b[0]:.1f} y {1000 * std_b[1]:.1f}"
            ta = f"corrected: std x {1000 * std_a[0]:.1f} y {1000 * std_a[1]:.1f}"
        else:
            sl = ci.slices[i - 1]
            raw, corrected = sl.image_raw, sl.image
            what = f"slice {i}: dt {sl.lo:.1f}..{sl.hi:.1f} us, {sl.n_true:.0f} pairs"
            tb = f"detector {cor.det_e}, uncorrected, {what}"
            ta = (f"corrected: mean x {1000 * sl.mean_corr[0]:.1f}, y {1000 * sl.mean_corr[1]:.1f}"
                  f" (uncorrected {1000 * sl.mean_raw[0]:.1f}, {1000 * sl.mean_raw[1]:.1f})")
        self._show(self.pc_img_before, raw, ci.extent, tb)
        self._show(self.pc_img_after, corrected, ci.extent, ta)

    def _refresh_correction(self, res, snap, cor, walk=None):
        """Right column: electron images (before / after) of the selected dt selection, and the fit."""
        n_sl = self.pc_slices.value()
        self.pc_view.setMaximum(n_sl)
        if cor is None:
            self._ci = self._cor = None
            self.pc_text.setPlainText("no correction: too few pairs with a position inside the window, "
                                      "or no window yet" + ("\n\n" + walk.text() if walk else ""))
            return
        s = self.session
        ci = correction_images(res, snap, s.flt.det_a, s.flt.det_b, cor, self._image_binning(cor.det_e),
                               n_slices=n_sl)
        if ci is None:
            return
        self._ci, self._cor = ci, cor
        self._render_images()
        st = s.position_correction()
        passed = (st.text() + "\n\n") if st is not None else ""
        self.pc_text.setPlainText(passed + getattr(self, "_cut_note", "") + (walk.text() + "\n\n" if walk else "")
                                  + cor.text() + ci.text())

    # -- refresh -------------------------------------------------------------
    def refresh(self, force=False):
        s = self.session
        if s is None:
            return
        res = s.coincidence()
        if res is None:
            return
        on_pos = self.pages.currentWidget() is self.page_pos
        sel = s.selection()
        key = (res.version, res.window_us, res.bkg_shots, res.n_sigma, on_pos, self.pc_enable.isChecked(),
               (self.pc_half.value(), self.pc_bin.value(), self.pc_electron.currentIndex(),
                self.pc_use_n.isChecked(), self.pc_order.value(), self.pc_corrected.isChecked(),
                self.pc_weighted.isChecked(), self.pc_slices.value(), self.pc_bkg.isChecked(),
                self.pc_walk.isChecked(), self.pc_walk_sigma.value(), self.pc_walk_n.isChecked(),
                self.pc_cuts.isChecked(), sel.version if sel is not None else None)
               if on_pos else None)
        if key == self._last and not force:
            return
        if on_pos and not force and self._last is not None and self._last[1:] == key[1:] \
                and time.monotonic() - self._pos_t < max(1.0, 4.0 * self._pos_dur):
            return                              # only new data: refit at most ~20% of the time
        self._last = key
        if on_pos and not self.pc_enable.isChecked():
            self._pos_off()
            return
        if on_pos:                              # only the visible page is computed
            t0 = time.monotonic()
            self._refresh_pos(res)
            self._pos_t = time.monotonic()
            self._pos_dur = self._pos_t - t0
            return
        c = coincidence_panel(res)
        self._busy = True
        self.p_full.set_curve("pairs", c.full)
        self.p_zoom.set_curve("pairs", c.zoom)
        if c.window is not None:
            self.zoom_region.setRegion(c.window)
            self.full_main.setRegion(c.window)
            self.full_main.setVisible(True)
            if self.auto.isChecked():
                set_quiet(self.tmin, c.window[0])
                set_quiet(self.tmax, c.window[1])
        for i, r in enumerate(self.full_ctrl):
            if i < len(c.controls):
                r.setRegion(c.controls[i][:2])
                r.setVisible(True)
            else:
                r.setVisible(False)
        names, counts = c.bars
        n = len(counts)
        self.bars.setOpts(x=np.arange(n), height=np.asarray(counts, float), width=0.6)
        self.p_bars.getPlotItem().getAxis("bottom").setTicks([list(enumerate(names))])
        self.acc_line.setValue(c.acc if c.acc is not None else 0.0)
        self.p_rate.set_curve("main", c.rate_main)
        self.p_rate.set_curve("acc", c.rate_acc)
        self.p_clock.set_curve("res", Hist(c.clock_t, c.clock_residual_us) if len(c.clock_t) else EMPTY)
        drift = f"  (drift {c.drift_ppm:+.2f} ppm)" if c.drift_ppm is not None else ""
        self.p_clock.getPlotItem().setTitle("clock offset B-A, residual to linear drift" + drift, size="9pt")
        self.period_label.setText(f"measured period: {c.period_us:.3f} us" if c.period_us
                                  else "measured period: not found (flat background)")
        self.text.setPlainText(c.text)
        self._busy = False