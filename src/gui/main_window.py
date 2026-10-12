"""Main window: data-source strip, file strip, tabs, status bar, and the refresh timer.

File strip: "save analysis to file" writes the raw data, all settings and everything computed from them to one
HDF5 file (storage/h5file.py), by hand ("save now"), every N seconds while data come in, and when the session
stops. "open file" loads such a file and re-runs the whole analysis on its raw data (offline analysis): all
settings come back, and everything can be changed and saved again.

The pipeline runs in worker threads and publishes immutable results. The GUI
thread only polls them from one QTimer (no Qt signals cross threads) and redraws
the visible tab when something changed.
"""
import os
import time
from dataclasses import fields
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from gui import gui_state
from gui.tab_coincidence import CoincidenceTab
from gui.tab_detectors import DetectorsTab
from gui.tab_polar import AnalysisTab
from paths import EXAMPLE_DATA, scan_detectors
from session import Session, SessionConfig
from storage import capture, h5file
from storage.saver import Saver

REFRESH_MS = 200
TITLE = "EVA – Electron–neutral VMI Analysis"
FILE_FILTER = "EVA analysis file (*.h5 *.hdf5)"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, directory=".", det_a=3, det_b=1, bins=256, demo_speed=2.0):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1500, 900)
        self.session = None
        self._bins = bins
        self.saver = Saver()
        self._offline = None            # path of the analysis file being worked on (None: live / demo)
        self._loaded = None             # its metadata (settings, widget state, config)
        self._live_dir = None
        self._t_start = time.time()

        # ---- data source strip ----
        self.dir_edit = QtWidgets.QLineEdit(str(directory))
        self.browse = QtWidgets.QPushButton("browse...")
        self.det_a = QtWidgets.QSpinBox(); self.det_a.setRange(1, 999); self.det_a.setValue(det_a)
        self.det_b = QtWidgets.QSpinBox(); self.det_b.setRange(1, 999); self.det_b.setValue(det_b)
        self.demo = QtWidgets.QCheckBox("demo")
        self.speed = QtWidgets.QDoubleSpinBox(); self.speed.setRange(0.1, 500.0)
        self.speed.setValue(demo_speed); self.speed.setSuffix(" x")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.browse.clicked.connect(lambda *_: self._browse())
        self.demo.toggled.connect(self._demo_toggled)
        self.start_btn.clicked.connect(lambda *_: self._toggle())

        strip = QtWidgets.QHBoxLayout()
        strip.addWidget(QtWidgets.QLabel("data folder"))
        strip.addWidget(self.dir_edit, 1)
        strip.addWidget(self.browse)
        strip.addWidget(QtWidgets.QLabel("A (filtered) det"))
        strip.addWidget(self.det_a)
        strip.addWidget(QtWidgets.QLabel("B (partner) det"))
        strip.addWidget(self.det_b)
        strip.addWidget(self.demo)
        strip.addWidget(self.speed)
        strip.addWidget(self.start_btn)

        # ---- file strip: saving the analysis, opening a saved one ----
        self.save_on = QtWidgets.QCheckBox("save analysis to file")
        self.save_on.setToolTip("Write raw data, all settings and the results (filtered images, Abel inversion, bootstrap)\n"
                                "to one HDF5 file: when you press 'save now', every N seconds while data come in\n"
                                "(only if something changed) and when the session stops.")
        self.save_path = QtWidgets.QLineEdit()
        self.save_path.setPlaceholderText("file name (.h5); empty = eva_analysis_<date>.h5 in the data folder")
        self.save_browse = QtWidgets.QPushButton("browse...")
        self.save_every = QtWidgets.QSpinBox(); self.save_every.setRange(5, 86400); self.save_every.setValue(60)
        self.save_every.setSuffix(" s")
        self.save_every.setToolTip("Automatic save interval while data are coming in.")
        self.save_now = QtWidgets.QPushButton("save now")
        self.open_btn = QtWidgets.QPushButton("open saved file (offline analysis)...")
        self.save_label = QtWidgets.QLabel("")
        self.save_browse.clicked.connect(lambda *_: self._save_browse())
        self.save_now.clicked.connect(lambda *_: self._save_now(background=True, force=True))
        self.open_btn.clicked.connect(lambda *_: self._open_file())
        self.save_on.toggled.connect(lambda *_: self._save_ui())

        strip2 = QtWidgets.QHBoxLayout()
        strip2.addWidget(self.save_on)
        strip2.addWidget(self.save_path, 1)
        strip2.addWidget(self.save_browse)
        strip2.addWidget(QtWidgets.QLabel("every"))
        strip2.addWidget(self.save_every)
        strip2.addWidget(self.save_now)
        strip2.addWidget(self.save_label)
        strip2.addStretch(1)
        strip2.addWidget(self.open_btn)
        self._save_ui()

        # ---- tabs ----
        self.tab_det, self.tab_coin, self.tab_polar = DetectorsTab(), CoincidenceTab(), AnalysisTab()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_det, "Detectors")
        self.tabs.addTab(self.tab_coin, "Coincidence")
        self.tabs.addTab(self.tab_polar, "Analysis")
        self.tabs.currentChanged.connect(lambda *_: self._refresh(force=True))

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(strip)
        lay.addLayout(strip2)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.status = QtWidgets.QLabel("stopped")
        self.statusBar().addWidget(self.status, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.setInterval(1000)
        self.save_timer.timeout.connect(self._save_tick)
        self.save_timer.start()

    # ---- source strip ----
    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "data folder", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _demo_toggled(self, on):
        for w in (self.dir_edit, self.browse):
            w.setEnabled(not on)
        if on:
            dets = scan_detectors(EXAMPLE_DATA)
            if len(dets) >= 2:
                self.det_a.setValue(dets[1])
                self.det_b.setValue(dets[0])

    def _toggle(self):
        self.stop() if self.session is not None else self.start()

    # ---- file strip ----
    def _save_ui(self):
        on = self.save_on.isChecked()
        for w in (self.save_path, self.save_browse, self.save_every):
            w.setEnabled(on)

    def _default_save_path(self):
        if self._offline:
            return str(Path(self._offline).with_name(Path(self._offline).stem + "_reanalysed.h5"))
        d = Path(self.dir_edit.text()) if (self.dir_edit.text() and not self.demo.isChecked()) else Path.cwd()
        if not d.is_dir():
            d = Path.cwd()
        return str(d / f"eva_analysis_{time.strftime('%Y%m%d_%H%M%S')}.h5")

    def _save_path(self):
        p = self.save_path.text().strip() or self._default_save_path()
        if not p.lower().endswith((".h5", ".hdf5")):
            p += ".h5"
        p = os.path.expanduser(p)
        self.save_path.setText(p)
        return p

    def _save_browse(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "analysis file", self.save_path.text().strip()
                                                     or self._default_save_path(), FILE_FILTER)
        if p:
            if not p.lower().endswith((".h5", ".hdf5")):
                p += ".h5"
            self.save_path.setText(p)

    def _same_as_source(self, path):
        return bool(self._offline) and os.path.abspath(path) == os.path.abspath(self._offline)

    def _save_now(self, background=True, force=True):
        """Write the file now (the widget state is read here, in the GUI thread; the rest in a worker thread)."""
        if self.session is None:
            self.save_label.setText("nothing to save: no session running")
            return False
        path = self._save_path()
        if self._same_as_source(path) and force and background:
            ans = QtWidgets.QMessageBox.question(self, "overwrite?", f"{path}\nis the file you opened. Overwrite it with the "
                                                 "current analysis?")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        return self.saver.save(self.session, path, gui=gui_state.capture(self), source=self._offline, force=force,
                               wait=not background)

    def _save_tick(self):
        s = self.session
        self.save_label.setText(self.saver.status() if (self.save_on.isChecked() or self.saver.last_time) else "")
        if s is None or self._offline or not self.save_on.isChecked() or self.saver.busy:
            return
        since = time.time() - (self.saver.last_time or self._t_start)
        if since >= self.save_every.value() and self.saver.needs_save(s, self._save_path()):
            self._save_now(background=True, force=False)

    def _open_file(self):
        start = os.path.dirname(self.save_path.text().strip()) or self.dir_edit.text() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "open saved analysis", start, FILE_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path):
        """Offline analysis: raw data and settings from the file; the whole pipeline runs on them again."""
        try:
            info = h5file.read_meta(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "cannot open", f"{path}\n\n{type(exc).__name__}: {exc}")
            return
        if self.session is not None:
            self.stop()
        cfg = info.get("config", {})
        self._live_dir = self.dir_edit.text()
        self._offline, self._loaded = path, info
        self.demo.setChecked(False)
        self.det_a.setValue(int(cfg.get("det_a", self.det_a.value())))
        self.det_b.setValue(int(cfg.get("det_b", self.det_b.value())))
        self._bins = int(cfg.get("bins", self._bins))
        self.dir_edit.setText(path)
        self.save_path.setText(self._default_save_path())
        gui_state.restore(self, info.get("gui"))
        self.start()
        self.setWindowTitle(f"EVA - offline analysis: {os.path.basename(path)}")

    # ---- session control ----
    def start(self):
        if self.session is not None:
            return
        demo = self.demo.isChecked() and not self._offline
        if self._offline:
            known = {f.name for f in fields(SessionConfig)}
            c = {k: v for k, v in self._loaded.get("config", {}).items() if k in known}
            c.update(directory="", demo_source=None, offline_file=self._offline, det_a=self.det_a.value(),
                     det_b=self.det_b.value(), bins=self._bins)
            cfg = SessionConfig(**c)
        else:
            cfg = SessionConfig(directory=self.dir_edit.text(), det_a=self.det_a.value(),
                                det_b=self.det_b.value(), bins=self._bins,
                                demo_source=str(EXAMPLE_DATA) if demo else None,
                                demo_speed=self.speed.value())
        self.session = Session(cfg)
        if self._offline:
            self.session.preset = self._loaded.get("settings")
        try:
            self.session.start()
        except Exception as exc:
            self.session = None
            self._offline = self._loaded = None
            QtWidgets.QMessageBox.critical(self, "cannot start", f"{type(exc).__name__}: {exc}")
            return
        self.saver.reset()
        self._t_start = time.time()
        self.tab_det.attach(self.session)
        self.tab_coin.attach(self.session)
        self.tab_polar.attach(self.session)
        if self._offline:                                  # the settings of the file win over what attach pushed
            capture.apply_settings(self.session, self._loaded.get("settings") or {}, running=True)
            try:
                capture.restore_bootstrap(self.session, h5file.read_results(self._offline, ("bootstrap",)).get("bootstrap"))
            except Exception:
                pass
        self.start_btn.setText("Stop")
        for w in (self.det_a, self.det_b, self.demo, self.speed, self.dir_edit, self.browse):
            w.setEnabled(False)

    def stop(self):
        if self.session is None:
            return
        if self.save_on.isChecked():                        # last save, before the data are gone
            path = self._save_path()
            if self._same_as_source(path):
                self.save_label.setText("not saved on stop: the target is the file that was opened")
            else:
                self.status.setText("saving the analysis ...")
                QtWidgets.QApplication.processEvents()
                self.saver.join(120)
                self._save_now(background=False, force=True)
        self.tab_det.detach()
        self.tab_coin.detach()
        self.tab_polar.detach()
        self.session.stop()
        self.session = None
        if self._offline:
            self.dir_edit.setText(self._live_dir or "")
            self.setWindowTitle(TITLE)
        self._offline = self._loaded = None
        self.start_btn.setText("Start")
        for w in (self.det_a, self.det_b, self.demo, self.speed):
            w.setEnabled(True)
        self._demo_toggled(self.demo.isChecked())
        self.status.setText("stopped")

    # ---- refresh ----
    def _tick(self):
        if self.session is None:
            return
        self.status.setText(self.session.status_line())
        self._refresh()

    def _refresh(self, force=False):
        tab = self.tabs.currentWidget()
        if tab is self.tab_det:
            self.tab_det.refresh(force)
        elif tab is self.tab_coin:
            self.tab_coin.refresh(force)
        elif tab is self.tab_polar:
            self.tab_polar.refresh(force)

    def closeEvent(self, event):
        self.stop()                     # joins worker threads, ends the replay process
        super().closeEvent(event)