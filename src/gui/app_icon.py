"""Application icon on every platform.

  * Windows and Linux/X11: QApplication.setWindowIcon is enough, plus (Windows) an explicit
    AppUserModelID, otherwise the taskbar groups EVA under python.exe and shows Python's icon.
  * macOS: setWindowIcon also sets the Dock icon while EVA runs.
  * Linux/Wayland (GNOME, KDE): the desktop takes the icon from a .desktop file whose name matches the
    app id (QGuiApplication.desktopFileName), not from the window. install_linux_desktop_entry() writes one,
    plus the icon, to ~/.local/share the first time EVA starts (and whenever they are out of date; a new
    icon also refreshes the icon cache, though GNOME may still show the old one until the next login).
    EVA then also appears in the application menu.

The icons live in gui/resources: eva.svg is the source, the PNGs and eva.ico are rendered from it.
"""
import sys
from pathlib import Path

from PyQt6 import QtGui

APP_ID = "eva-desiree"                       # Wayland app id = .desktop file name = icon name
APP_NAME = "EVA"
RESOURCES = Path(__file__).resolve().parent / "resources"
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)


def app_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for s in PNG_SIZES:
        p = RESOURCES / f"eva_{s}.png"
        if p.exists():
            icon.addFile(str(p))
    return icon


def set_windows_app_id():
    """Call before the QApplication is created."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"desiree.{APP_ID}")
        except Exception:
            pass


def _desktop_quote(arg) -> str:
    """Quote one Exec= argument as the Desktop Entry spec wants it."""
    a = str(arg).replace("%", "%%")
    if not any(c in a for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return a
    for c in '\\"`$':
        a = a.replace(c, "\\" + c)
    return f'"{a}"'


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def _refresh_icon_cache(icons: Path):
    """Tell the desktop that the icons changed: bump the theme folder's time stamp (GTK rescans on that)
    and rebuild the icon cache if the tool is installed."""
    import os
    import shutil
    import subprocess
    try:
        os.utime(icons)
    except OSError:
        pass
    tool = shutil.which("gtk-update-icon-cache") or shutil.which("gtk4-update-icon-cache")
    if tool:
        try:
            subprocess.run([tool, "-q", "-f", "-t", str(icons)], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def install_linux_desktop_entry(script: Path):
    """Write ~/.local/share/applications/<APP_ID>.desktop and the icon; never raises."""
    if not sys.platform.startswith("linux"):
        return
    try:
        share = Path.home() / ".local" / "share"
        icons = share / "icons" / "hicolor"
        changed = _write_if_changed(icons / "scalable" / "apps" / f"{APP_ID}.svg", (RESOURCES / "eva.svg").read_bytes())
        for s in PNG_SIZES:
            png = RESOURCES / f"eva_{s}.png"
            if png.exists():
                changed |= _write_if_changed(icons / f"{s}x{s}" / "apps" / f"{APP_ID}.png", png.read_bytes())
        if changed:
            _refresh_icon_cache(icons)
        exec_line = " ".join(_desktop_quote(a) for a in (sys.executable, Path(script).resolve()))
        entry = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            "GenericName=Electron–neutral VMI Analysis\n"
            "Comment=Coincidence velocity-map imaging analysis (DESIREE)\n"
            f"Exec={exec_line}\n"
            f"Icon={APP_ID}\n"
            "Terminal=false\n"
            "Categories=Science;Physics;Education;\n"
        )
        _write_if_changed(share / "applications" / f"{APP_ID}.desktop", entry.encode())
    except Exception:
        pass


def setup_app_icon(app):
    """Call right after the QApplication is created."""
    app.setDesktopFileName(APP_ID)
    app.setWindowIcon(app_icon())
