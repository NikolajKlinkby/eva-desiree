"""Tab 1: one row per detector -> image | event time | charge sum.

The event-time and charge-sum plots carry two draggable vertical lines that set
the time and Sum cuts live (checkbox to enable, numeric boxes for exact values).
Grey = all events, orange = true coincidence pairs (background-subtracted,
independent of the current cuts: pick limits here), blue = events kept.
"""
from PyQt6 import QtWidgets

from gui.plotdata import apply_cut, detector_panel
from gui.widgets import COLOR, CutPlot, ImagePlot


class DetectorRow(QtWidgets.QGroupBox):
    def __init__(self, tab, side):
        super().__init__("")
        self.tab, self.side, self.det = tab, side, None
        self.image = ImagePlot("image")
        self.time = CutPlot("event time", "time since first event (s)",
                            on_change=lambda e, lo, hi: self._cut("time", e, lo, hi), decimals=1)
        self.sum = CutPlot("charge sum", "Sum",
                           on_change=lambda e, lo, hi: self._cut("sum", e, lo, hi), decimals=0)
        self.time.plot.add_curve("all", COLOR["all"], label="all events")
        self.time.plot.add_curve("sel", COLOR["sel"], label="kept")
        self.sum.plot.add_curve("all", COLOR["all"], label="all events")
        self.sum.plot.add_curve("true", COLOR["true"], label="true pairs", filled=False, width=2)
        self.sum.plot.add_curve("sel", COLOR["sel"], label="kept")
        lay = QtWidgets.QHBoxLayout(self)
        for w in (self.image, self.time, self.sum):
            lay.addWidget(w, 1)

    def set_detector(self, det, label):
        self.det = det
        self.setTitle(f"Detector {det} ({label})")

    def _cut(self, kind, enabled, lo, hi):
        s = self.tab.session
        if s is not None and s.sel is not None and self.det is not None:
            apply_cut(s.sel, self.det, kind, enabled, lo, hi)

    def push(self):
        """Send both cuts to the pipeline (used after a (re)start)."""
        self.time.push()
        self.sum.push()

    def update_panel(self, p):
        if p is None:
            return
        self.image.set_image(p.image, p.extent, p.levels,
                             title=f"image  ({p.n_selected} kept)" + (
                                 f"  centroid ({p.centroid[0]:.3f}, {p.centroid[1]:.3f})" if p.centroid else ""))
        self.time.set_axis_max(p.t_max)
        self.sum.set_axis_max(p.sum_max)
        self.time.plot.set_curve("all", p.time_all)
        self.time.plot.set_curve("sel", p.time_sel)
        self.sum.plot.set_curve("all", p.sum_all)
        self.sum.plot.set_curve("true", p.sum_true)
        self.sum.plot.set_curve("sel", p.sum_sel)


class DetectorsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.session = None
        self._last = None

        # controls that apply to both detectors
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["subtracted", "raw", "all"])
        self.use_coin = QtWidgets.QCheckBox("use coincidence")
        self.use_coin.setChecked(True)
        self.stage = QtWidgets.QComboBox()
        self.stage.addItems(["pre", "post"])
        self.fit_a = QtWidgets.QPushButton("fit image range A")
        self.fit_b = QtWidgets.QPushButton("fit image range B")
        self.mode.currentTextChanged.connect(self._mode_changed)
        self.use_coin.toggled.connect(self._coin_toggled)
        self.stage.currentTextChanged.connect(self._stage_changed)
        self.fit_a.clicked.connect(lambda *_: self._fit("a"))
        self.fit_b.clicked.connect(lambda *_: self._fit("b"))

        top = QtWidgets.QHBoxLayout()
        for text, w in (("image:", self.mode), ("Sum cut applied", self.stage)):
            top.addWidget(QtWidgets.QLabel(text))
            top.addWidget(w)
        top.addWidget(self.use_coin)
        top.addStretch(1)
        top.addWidget(self.fit_a)
        top.addWidget(self.fit_b)

        self.rows = {"a": DetectorRow(self, "a"), "b": DetectorRow(self, "b")}
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.rows["a"], 1)
        lay.addWidget(self.rows["b"], 1)

    # -- session -------------------------------------------------------------
    def attach(self, session):
        """Called when a session starts: label the rows and push all UI state."""
        self.session, self._last = session, None
        self.rows["a"].set_detector(session.config.det_a, "A")
        self.rows["b"].set_detector(session.config.det_b, "B")
        self._mode_changed(self.mode.currentText())
        self._stage_changed(self.stage.currentText())
        self._coin_toggled(self.use_coin.isChecked())
        for r in self.rows.values():
            r.push()

    def detach(self):
        self.session = None

    # -- controls ------------------------------------------------------------
    def _mode_changed(self, text):
        if self.session is not None and self.session.img is not None:
            self.session.img.set_mode(text)

    def _stage_changed(self, text):
        if self.session is not None and self.session.sel is not None:
            self.session.sel.set_stage(text)

    def _coin_toggled(self, on):
        if self.session is not None and self.session.sel is not None:
            self.session.sel.use_coincidence(bool(on))

    def _fit(self, side):
        s = self.session
        if s is None or s.img is None:
            return
        try:
            s.img.fit_range(self.rows[side].det)
        except RuntimeError:
            pass                                    # no data yet

    # -- refresh (called by the main window's timer) ---------------------------
    def refresh(self, force=False):
        s = self.session
        if s is None:
            return
        sel, img = s.selection(), s.imaging()
        key = (sel.version if sel else None, img.version if img else None)
        if key == self._last and not force:
            return
        self._last = key
        if sel is None:
            return
        for side, row in self.rows.items():
            row.update_panel(detector_panel(sel, img, row.det))