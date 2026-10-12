"""Value of every input widget of the tabs, as a plain dict (saved in the analysis file) and back.

The tabs push the values of their widgets into the pipeline when a session starts (attach). Restoring the
widgets before that start therefore restores the settings the pipeline gets, and the window shows what was
used. Keys are attribute paths, e.g. "tab_polar.tik" or "tab_det.rows.a.sum.lo".
Walks attributes of the tabs and of gui objects below them (no Qt children, no other packages).
"""
from PyQt6 import QtWidgets

TABS = ("tab_det", "tab_coin", "tab_polar")
_INPUT = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QLineEdit, QtWidgets.QAbstractButton)
_MAX_DEPTH = 4


def _widgets(obj, path, out, seen, depth):
    if depth > _MAX_DEPTH or id(obj) in seen:
        return
    seen.add(id(obj))
    for name, val in list(getattr(obj, "__dict__", {}).items()):
        if name.startswith("_"):
            continue
        p = f"{path}.{name}"
        if isinstance(val, _INPUT):
            out[p] = val
        elif isinstance(val, dict):
            for k, v in val.items():
                _item(v, f"{p}.{k}", out, seen, depth)
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                _item(v, f"{p}.{i}", out, seen, depth)
        else:
            _item(val, p, out, seen, depth)


def _item(val, p, out, seen, depth):
    if isinstance(val, _INPUT):
        out[p] = val
    elif type(val).__module__.startswith("gui."):
        _widgets(val, p, out, seen, depth + 1)


def find_widgets(window):
    out, seen = {}, set()
    for t in TABS:
        tab = getattr(window, t, None)
        if tab is not None:
            _widgets(tab, t, out, seen, 0)
    return out


def _get(w):
    if isinstance(w, QtWidgets.QAbstractSpinBox):
        return {"kind": "spin", "value": w.value()}
    if isinstance(w, QtWidgets.QComboBox):
        return {"kind": "combo", "text": w.currentText(), "index": w.currentIndex()}
    if isinstance(w, QtWidgets.QLineEdit):
        return None if w.isReadOnly() else {"kind": "text", "value": w.text()}
    if isinstance(w, QtWidgets.QAbstractButton) and w.isCheckable():
        return {"kind": "check", "value": bool(w.isChecked())}
    return None


def capture(window):
    """{path: {"kind": ..., ...}} of the tabs of a MainWindow."""
    out = {}
    for path, w in find_widgets(window).items():
        v = _get(w)
        if v is not None:
            out[path] = v
    return out


def restore(window, state):
    """Set the widgets from a captured dict (unknown keys are ignored). Returns the number set."""
    n = 0
    for path, w in find_widgets(window).items():
        v = (state or {}).get(path)
        if not v:
            continue
        try:
            k = v.get("kind")
            if k == "spin" and isinstance(w, QtWidgets.QAbstractSpinBox):
                w.setValue(v["value"])
            elif k == "combo" and isinstance(w, QtWidgets.QComboBox):
                i = w.findText(v["text"])
                if i >= 0:
                    w.setCurrentIndex(i)
            elif k == "text" and isinstance(w, QtWidgets.QLineEdit):
                w.setText(v["value"])
            elif k == "check" and isinstance(w, QtWidgets.QAbstractButton):
                w.setChecked(bool(v["value"]))
            else:
                continue
            n += 1
        except Exception:                                 # a widget that changed since the file was written
            pass
    return n
