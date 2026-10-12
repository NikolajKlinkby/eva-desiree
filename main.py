"""EVA: Electron–neutral VMI Analysis: start the GUI.

  python main.py                              # then pick a folder and press Start
  python main.py --dir /path/to/daq/output --a 3 --b 2 --start
  python main.py --demo                       # replay examples/test_data, no hardware needed

Requires:  pip install PyQt6 pyqtgraph numpy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="folder the DAQ writes RA<det>_<date>.<time>.csv into")
    ap.add_argument("--a", type=int, default=3, help="detector to filter/image first (e.g. VMI)")
    ap.add_argument("--b", type=int, default=1, help="partner detector (e.g. RAEA)")
    ap.add_argument("--bins", type=int, default=256, help="image size in pixels")
    ap.add_argument("--demo", action="store_true", help="replay examples/test_data and start immediately")
    ap.add_argument("--speed", type=float, default=2.0, help="demo replay speed factor")
    ap.add_argument("--start", action="store_true", help="start the pipeline immediately")
    ap.add_argument("--theme", choices=["dark", "light"], default="dark", help="colour theme")
    args = ap.parse_args()

    try:
        from PyQt6 import QtWidgets
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        sys.exit(f"{exc}\nInstall the GUI dependencies with:  pip install PyQt6 pyqtgraph numpy")
    from gui.main_window import MainWindow
    from gui.theme import apply_theme

    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app, args.theme)                 # before any window: pyqtgraph reads it at creation
    win = MainWindow(directory=args.dir, det_a=args.a, det_b=args.b, bins=args.bins,
                     demo_speed=args.speed)
    win.show()
    if args.demo:
        win.demo.setChecked(True)
    if args.demo or args.start:
        win.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()