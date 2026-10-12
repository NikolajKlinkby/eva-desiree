"""Reusable Qt/pyqtgraph building blocks. They draw what plotdata.py prepares and
report user input through plain Python callbacks (no custom Qt signals), so they
stay small and easy to swap.

Conventions: image arrays are indexed [iy, ix] (row-major), x to the right, y up.
"""
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

COLOR = dict(all=(150, 150, 150), sel=(70, 130, 255), true=(255, 150, 40),
             acc=(120, 200, 120), main=(70, 130, 255))


# one colour per angular order n (index n modulo the length): 0 blue, 1 orange, 2 green, ...
ORDER_COLORS = [(70, 130, 255), (255, 150, 40), (90, 200, 120), (230, 90, 160), (170, 120, 255),
                (240, 220, 80), (90, 210, 220), (200, 200, 200), (255, 110, 110)]


def fill(color, alpha=60):
    return (*color, alpha)


def spin(lo, hi, value, step=1.0, decimals=1, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    return w


def set_quiet(widget, value):
    """Set a widget's value without triggering its change signal."""
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)


class CollapsibleSection(QtWidgets.QWidget):
    """A titled section whose body can be folded away by clicking the header.

    Put widgets in `.form` (a QFormLayout) or add your own layout to `.body`.
    Folded widgets keep their values and keep working, they are only hidden."""
    def __init__(self, title, expanded=True):
        super().__init__()
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; padding: 3px; }")
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._arrow(expanded)
        self.body = QtWidgets.QFrame()
        self.body.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.form = QtWidgets.QFormLayout(self.body)
        self.body.setVisible(expanded)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.body)
        self.toggle.toggled.connect(self._toggled)

    def _arrow(self, expanded):
        arrow = QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        self.toggle.setArrowType(arrow)

    def _toggled(self, on):
        self._arrow(bool(on))
        self.body.setVisible(bool(on))

    def set_expanded(self, on):
        self.toggle.setChecked(bool(on))

    def is_expanded(self):
        return self.toggle.isChecked()

    def set_content_enabled(self, on):
        """Disable the body (e.g. a placeholder) but keep the header clickable."""
        self.body.setEnabled(bool(on))


def new_plot(title="", xlabel="", ylabel="", grid=True):
    pw = pg.PlotWidget()
    pi = pw.getPlotItem()
    if title:
        pi.setTitle(title, size="9pt")
    pi.setLabel("bottom", xlabel)
    pi.setLabel("left", ylabel)
    if grid:
        pi.showGrid(x=True, y=True, alpha=0.25)
    return pw


# ------------------------------------------------------------------- image
class ImagePlot(pg.PlotWidget):
    """2D image; keeps the user's zoom between updates. The aspect ratio can be
    locked (Cartesian images) or free (polar map)."""
    def __init__(self, title="", xlabel="x", ylabel="y"):
        super().__init__()
        pi = self.getPlotItem()
        pi.setTitle(title, size="9pt")
        pi.setLabel("bottom", xlabel)
        pi.setLabel("left", ylabel)
        pi.setAspectLocked(True)
        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))
        pi.addItem(self.image_item)
        self._key = None                                # (extent, aspect) of what is shown

    def set_image(self, image, extent, levels, title=None, aspect=True, xlabel=None, ylabel=None):
        if image is None:
            return
        pi = self.getPlotItem()
        self.image_item.setImage(image, autoLevels=False)
        self.image_item.setLevels(levels)
        if (extent, aspect) != self._key:               # extent = (x0, x1, y0, y1)
            x0, x1, y0, y1 = extent
            pi.setAspectLocked(bool(aspect))
            self.image_item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            pi.getViewBox().autoRange()
            self._key = (extent, aspect)
        if title is not None:
            pi.setTitle(title, size="9pt")
        if xlabel is not None:
            pi.setLabel("bottom", xlabel)
        if ylabel is not None:
            pi.setLabel("left", ylabel)


# --------------------------------------------------------------- histograms
class HistPlot(pg.PlotWidget):
    """A plot holding named histogram curves."""
    def __init__(self, title="", xlabel="", ylabel="counts", legend=False):
        super().__init__()
        pi = self.getPlotItem()
        pi.setTitle(title, size="9pt")
        pi.setLabel("bottom", xlabel)
        pi.setLabel("left", ylabel)
        pi.showGrid(x=True, y=True, alpha=0.25)
        self.curves = {}
        self.bands = {}
        self._legend = pi.addLegend(offset=(5, 5)) if legend else None

    def add_band(self, name, color, alpha=55):
        """A shaded band between two curves (an error band), drawn behind the curves."""
        lo = pg.PlotDataItem(pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine))
        hi = pg.PlotDataItem(pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine))
        area = pg.FillBetweenItem(lo, hi, brush=fill(color, alpha))
        for item in (area, lo, hi):
            self.getPlotItem().addItem(item)
        self.bands[name] = (lo, hi, area)
        return self.bands[name]

    def set_band(self, name, band):
        """band: (x, lower, upper) or None (hides it)."""
        lo, hi, area = self.bands[name]
        on = band is not None
        for item in (lo, hi, area):
            item.setVisible(on)
        if on:
            lo.setData(band[0], band[1])
            hi.setData(band[0], band[2])

    def add_curve(self, name, color, filled=True, label=None, width=1, dash=False, symbol=None):
        style = QtCore.Qt.PenStyle.DashLine if dash else QtCore.Qt.PenStyle.SolidLine
        kw = dict(pen=pg.mkPen(color, width=width, style=style))
        if filled:
            kw.update(fillLevel=0, brush=fill(color))
        if symbol:
            kw.update(symbol=symbol, symbolSize=5, symbolBrush=color, symbolPen=None)
        curve = pg.PlotDataItem(**kw)
        self.getPlotItem().addItem(curve)
        if self._legend is not None and label:
            self._legend.addItem(curve, label)
        self.curves[name] = curve
        return curve

    def set_curve(self, name, hist):
        self.curves[name].setData(hist.x, hist.y)

    def set_log(self, on):
        self.getPlotItem().setLogMode(x=False, y=bool(on))


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


class DualAxisPlot(pg.PlotWidget):
    """One plot, one shared x axis, two y axes: left quantity (colour L) and right quantity (colour R).
    Each side holds named data curves (the binned means, with error bars) and raw-point layers."""
    def __init__(self, title="", xlabel="", left="", right="", color_left=(70, 130, 255),
                 color_right=(255, 150, 40)):
        super().__init__()
        pi = self.getPlotItem()
        pi.setTitle(title, size="9pt")
        pi.setLabel("bottom", xlabel)
        pi.setLabel("left", left, color=_hex(color_left))
        pi.showAxis("right")
        pi.getAxis("right").setLabel(right, color=_hex(color_right))
        pi.showGrid(x=True, y=True, alpha=0.25)
        self.color = {"left": color_left, "right": color_right}
        self.vb2 = pg.ViewBox()                       # the right axis has its own view box
        pi.scene().addItem(self.vb2)
        pi.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(pi)
        pi.getViewBox().sigResized.connect(self._sync)
        self.curves, self.errors = {}, {}
        self._sync()

    def _sync(self, *_):
        vb = self.getPlotItem().getViewBox()
        self.vb2.setGeometry(vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(vb, self.vb2.XAxis)

    def _add(self, side, item):
        (self.getPlotItem() if side == "left" else self.vb2).addItem(item)

    def add_series(self, name, side, points=False, dashed=False, line=False):
        """A line with markers and error bars (the binned means), or just faint markers (points=True).
        dashed=True draws the line dashed with triangles (the corrected means next to the uncorrected)."""
        c = self.color[side]
        if points:
            item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=2, symbolBrush=fill(c, 50), symbolPen=None)
        elif line:                                               # a plain thin curve (a model), no markers
            item = pg.PlotDataItem(pen=pg.mkPen(c, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        elif dashed:
            item = pg.PlotDataItem(pen=pg.mkPen(c, width=2, style=QtCore.Qt.PenStyle.DashLine), symbol="t",
                                   symbolSize=7, symbolBrush=None, symbolPen=pg.mkPen(c, width=1.5))
            err = pg.ErrorBarItem(pen=pg.mkPen(c, width=1, style=QtCore.Qt.PenStyle.DashLine))
            self._add(side, err)
            self.errors[name] = err
        else:
            item = pg.PlotDataItem(pen=pg.mkPen(c, width=2), symbol="o", symbolSize=5, symbolBrush=c, symbolPen=None)
            err = pg.ErrorBarItem(pen=pg.mkPen(c, width=1))
            self._add(side, err)
            self.errors[name] = err
        self._add(side, item)
        self.curves[name] = item
        return item

    def set_series(self, name, x, y, err=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.curves[name].setData(x, y)
        if name in self.errors:
            if err is None:
                self.errors[name].setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
            else:
                good = np.isfinite(y) & np.isfinite(err)
                self.errors[name].setData(x=x[good], y=y[good], height=2 * np.asarray(err, float)[good])

    def set_visible(self, name, on):
        self.curves[name].setVisible(bool(on))
        if name in self.errors:
            self.errors[name].setVisible(bool(on))


def set_order_curves(plot, items, key="n", label="n = {n}", width=2):
    """Draw one curve per angular order on a HistPlot: ((n, Hist), ...). Curves are created the first
    time an order appears and hidden when it is no longer in use (e.g. after lowering n_max)."""
    used = set()
    for n, hist in items:
        name = f"{key}{n}"
        if name not in plot.curves:
            plot.add_curve(name, ORDER_COLORS[n % len(ORDER_COLORS)], filled=False,
                           label=label.format(n=n), width=width)
        plot.curves[name].setVisible(True)
        plot.set_curve(name, hist)
        used.add(name)
    for name, curve in plot.curves.items():
        if name.startswith(key) and name not in used:
            curve.setVisible(False)


def set_order_bands(plot, items, key="b"):
    """One error band per angular order: ((n, (x, lower, upper)), ...); bands of orders no longer in
    use are hidden. Same colours as set_order_curves."""
    used = set()
    for n, band in items:
        name = f"{key}{n}"
        if name not in plot.bands:
            plot.add_band(name, ORDER_COLORS[n % len(ORDER_COLORS)])
        plot.set_band(name, band)
        used.add(name)
    for name in plot.bands:
        if name.startswith(key) and name not in used:
            plot.set_band(name, None)


class CutPlot(QtWidgets.QWidget):
    """A histogram plot with two draggable vertical lines (a LinearRegionItem)
    that define a [min, max] cut, plus a checkbox and numeric boxes.

    on_change(enabled, lo, hi) is called (throttled to ~60 ms) whenever the cut
    changes, whether by dragging a line, typing a number or ticking the box.
    """
    def __init__(self, title, xlabel, on_change=None, decimals=1):
        super().__init__()
        self._on_change = on_change
        self._enabled, self._lo, self._hi = False, 0.0, 1.0
        self._placed = False              # has the region been positioned yet?
        self._xmax = None
        self._busy = False

        self.plot = HistPlot(title, xlabel, legend=True)
        self.region = pg.LinearRegionItem(values=(0.0, 1.0), brush=fill((255, 255, 255), 25))
        self.region.setVisible(False)
        self.plot.getPlotItem().addItem(self.region)
        self.region.sigRegionChanged.connect(self._region_moved)

        self.check = QtWidgets.QCheckBox("cut")
        self.lo_spin = spin(-1e12, 1e12, 0.0, decimals=decimals)
        self.hi_spin = spin(-1e12, 1e12, 1.0, decimals=decimals)
        for s in (self.lo_spin, self.hi_spin):
            s.setEnabled(False)
        self.check.toggled.connect(self._toggled)
        self.lo_spin.valueChanged.connect(lambda v: self._typed(lo=v))
        self.hi_spin.valueChanged.connect(lambda v: self._typed(hi=v))

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.check)
        bar.addWidget(QtWidgets.QLabel("min"))
        bar.addWidget(self.lo_spin)
        bar.addWidget(QtWidgets.QLabel("max"))
        bar.addWidget(self.hi_spin)
        bar.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot, 1)
        lay.addLayout(bar)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.push)

    # -- data ------------------------------------------------------------------
    def set_axis_max(self, xmax):
        """Largest x of the data: used to place the lines the first time."""
        self._xmax = float(xmax)

    # -- state -----------------------------------------------------------------
    def state(self):
        return self._enabled, self._lo, self._hi

    def set_state(self, enabled, lo, hi):
        """Programmatic setting (does not call on_change; call push() for that)."""
        self._busy = True
        self._enabled, self._lo, self._hi, self._placed = bool(enabled), float(lo), float(hi), True
        self.check.blockSignals(True)
        self.check.setChecked(self._enabled)
        self.check.blockSignals(False)
        self._sync_widgets()
        self._busy = False

    def push(self):
        """Report the current state to the owner now."""
        if self._on_change is not None:
            self._on_change(self._enabled, self._lo, self._hi)

    # -- user input ------------------------------------------------------------
    def _toggled(self, on):
        self._enabled = bool(on)
        if self._enabled and not self._placed and self._xmax:
            self._lo, self._hi, self._placed = 0.1 * self._xmax, 0.9 * self._xmax, True
        self._busy = True
        self._sync_widgets()
        self._busy = False
        self._timer.start()

    def _region_moved(self, *_):
        if self._busy:
            return
        lo, hi = self.region.getRegion()
        self._lo, self._hi = float(lo), float(hi)
        self._busy = True
        set_quiet(self.lo_spin, self._lo)
        set_quiet(self.hi_spin, self._hi)
        self._busy = False
        self._timer.start()

    def _typed(self, lo=None, hi=None):
        if self._busy:
            return
        self._lo = self._lo if lo is None else float(lo)
        self._hi = self._hi if hi is None else float(hi)
        self._busy = True
        self.region.setRegion((self._lo, self._hi))
        self._busy = False
        self._timer.start()

    def _sync_widgets(self):
        self.region.setVisible(self._enabled)
        self.region.setRegion((self._lo, self._hi))
        for s, v in ((self.lo_spin, self._lo), (self.hi_spin, self._hi)):
            s.setEnabled(self._enabled)
            set_quiet(s, v)
