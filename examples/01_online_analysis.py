"""Online analysis: follow a DAQ folder while the files grow.

The same pipeline as the GUI, without a window:  DetectorSet (tails the CSV files) -> coincidence -> cuts ->
position correction -> images -> centring -> Abel inversion. Every stage works in its own thread and publishes
immutable results; this script only reads them.

  python examples/01_online_analysis.py --demo                      # replay examples/test_data as if it were a live run
  python examples/01_online_analysis.py --dir /data/live --a 3 --b 1  # your DAQ folder, runs until Ctrl-C
  python examples/01_online_analysis.py --demo --save examples/output/live.h5 --autosave 10

--save writes the analysis file (raw data, settings, results, plots) every --autosave seconds while data come in,
and once more at the end - the non-GUI version of the "save analysis to file" toggle.
"""
import argparse
import time

import common
from common import example_detectors
from session import Session, SessionConfig
from paths import EXAMPLE_DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".", help="folder with RA<det>_<date>.<time>.csv files")
    ap.add_argument("--demo", action="store_true", help="replay examples/test_data into a temporary folder instead")
    ap.add_argument("--speed", type=float, default=20.0, help="replay speed of --demo")
    ap.add_argument("--a", type=int, help="detector filtered and imaged first (default: demo detectors)")
    ap.add_argument("--b", type=int, help="partner detector")
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run (0 = until Ctrl-C)")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between status lines")
    ap.add_argument("--save", help="analysis file (.h5) to write while running")
    ap.add_argument("--autosave", type=float, default=30.0, help="seconds between saves")
    args = ap.parse_args()

    a, b = args.a, args.b
    if args.demo and not (a and b):
        a, b = example_detectors(EXAMPLE_DATA)
    s = Session(SessionConfig(directory=args.dir, det_a=a, det_b=b, bins=128,
                              demo_source=str(EXAMPLE_DATA) if args.demo else None, demo_speed=args.speed))
    s.start()
    saver = None
    if args.save:
        from storage.saver import Saver
        saver = Saver()
    t0 = last_save = time.time()
    try:
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            time.sleep(args.every)
            print(f"[{time.time() - t0:5.0f} s] {s.status_line()}")
            ab = s.abel_result()
            if ab is not None and ab.a.abel is not None and ab.a.n_events:
                i = common.peak_index(ab.a.abel)
                print(f"          Abel (det {ab.a.det}): {ab.a.n_events} events, I(rho) peak at rho = {ab.a.rho[i]:.4g}")
            if saver and time.time() - last_save >= args.autosave:
                if saver.save(s, args.save):                 # only writes if something changed
                    print("          saving to", args.save)
                last_save = time.time()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if saver:
            saver.join()
            saver.save(s, args.save, force=True, wait=True)  # the final state
            print("saved", args.save, "-", saver.status())
        common.report(s, "final state")
        s.stop()


if __name__ == "__main__":
    main()
