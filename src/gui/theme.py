"""Application theme: one place for the dark / light palette (Qt widgets and pyqtgraph).

Call apply_theme(app, 'dark') before any window is created: pyqtgraph reads its
background / foreground options when a plot is constructed. The data colours used in
the plots live in widgets.COLOR (chosen to read well on both).
"""
import pyqtgraph as pg
from PyQt6 import QtGui

DARK = dict(Window=(45, 45, 48), WindowText=(220, 220, 220), Base=(30, 30, 32),
            AlternateBase=(45, 45, 48), ToolTipBase=(60, 60, 63), ToolTipText=(230, 230, 230),
            Text=(220, 220, 220), Button=(60, 60, 64), ButtonText=(230, 230, 230),
            BrightText=(255, 90, 90), Link=(90, 160, 255), Highlight=(60, 120, 230),
            HighlightedText=(255, 255, 255))
DISABLED = (125, 125, 125)
PLOT_DARK = dict(background=(22, 22, 24), foreground=(190, 190, 190))
PLOT_LIGHT = dict(background="w", foreground="k")


def apply_theme(app, name="dark"):
    app.setStyle("Fusion")
    if name == "dark":
        pal = QtGui.QPalette()
        roles, groups = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
        for role, rgb in DARK.items():
            pal.setColor(getattr(roles, role), QtGui.QColor(*rgb))
        for role in ("Text", "ButtonText", "WindowText"):
            pal.setColor(groups.Disabled, getattr(roles, role), QtGui.QColor(*DISABLED))
        app.setPalette(pal)
        opts = PLOT_DARK
    else:
        app.setPalette(app.style().standardPalette())
        opts = PLOT_LIGHT
    for key, value in opts.items():
        pg.setConfigOption(key, value)