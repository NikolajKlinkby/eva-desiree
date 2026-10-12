"""Run the analysis on raw files and write everything to one HDF5 file.

The file holds the raw data, all settings, the results and the data of every plot (see README, "Saving the
analysis"). This is what the "save analysis to file" toggle does in the GUI.

  python examples/04_save_h5.py                                   # example data -> examples/output/example_run.h5
  python examples/04_save_h5.py --dir /data/run42 --a 3 --b 1 --out run42.h5 --comment "10 K source"
"""
import argparse

import common
from paths import EXAMPLE_DATA
from storage import capture, h5file, plots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(EXAMPLE_DATA))
    ap.add_argument("--a", type=int)
    ap.add_argument("--b", type=int)
    ap.add_argument("--out", default=str(common.OUTPUT / "example_run.h5"))
    ap.add_argument("--comment", default="", help="free text stored in /meta")
    ap.add_argument("--bootstrap", type=int, default=0, help="also run a bootstrap with this many samples first")
    args = ap.parse_args()

    s = common.start_session(args.dir, args.a, args.b)
    try:
        common.wait_settled(s)
        if args.bootstrap:
            s.abel.set_bootstrap("a", n_boot=args.bootstrap)
            s.abel.start_bootstrap("a")
            common.wait_until(lambda: not s.abel.bootstrap_running("a"), what="bootstrap")
        snap = capture.capture(s, comment=args.comment)      # raw + settings + results + plots; any thread
        h5file.write_file(args.out, snap)
        print(f"written {args.out}")
        print("sections:", ", ".join(k for k in ("meta", "config", "gui", "settings", "raw", "results", "plots")))
        print("plots stored:", len(plots.index(snap["plots"])))
    finally:
        s.stop()


if __name__ == "__main__":
    main()
