"""Offline analysis of finished DAQ files (raw CSV), without a window.

The files are complete, so the pipeline runs through them once and settles. Then everything the GUI shows is
available as arrays: coincidence timing, cuts, images, centre, Abel inversion.

  python examples/02_offline_raw_analysis.py                       # the recorded example runs
  python examples/02_offline_raw_analysis.py --dir /data/run42 --a 3 --b 1
"""
import argparse

import common
from paths import EXAMPLE_DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(EXAMPLE_DATA))
    ap.add_argument("--a", type=int)
    ap.add_argument("--b", type=int)
    args = ap.parse_args()

    s = common.start_session(args.dir, args.a, args.b)
    try:
        common.wait_settled(s)
        common.report(s, "raw files, default settings")

        # --- results are immutable snapshots; these are the arrays behind the plots --------------------
        coin = s.coincidence()
        print("\ncoincidence window (us):", coin.window_us, "| pairs found:", len(coin.dt_us))
        sel = s.selection()
        print("events kept on A / B:", int(sel.mask_a.sum()), "/", int(sel.mask_b.sum()),
              "| weight after background subtraction:", round(float(sel.weight_a.sum()), 1))
        img = s.imaging()[s.config.det_a]
        print("image of detector", img.det, img.image.shape, "total", round(float(img.total), 1), "centroid", img.centroid())
        pol = s.polar_result()["a"]
        print("centre (x, y):", tuple(round(c, 4) for c in pol.center), "| polar image", pol.corr.shape,
              "= [angle, radius]")
        ab = s.abel_result()["a"]
        print("Abel: orders", ab.orders, "| I_n(rho) array", ab.abel.I_n.shape, "| beta", ab.abel.beta.shape)

        # --- optional picture -------------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            ax[0].hist(coin.dt_us, bins=400); ax[0].set(title="dt = t_B - t_A", xlabel="us")
            ax[1].imshow(img.image, origin="lower", extent=img.binning.extent); ax[1].set(title=f"detector {img.det}")
            ax[2].plot(ab.rho, ab.abel.intensity); ax[2].set(title="I(rho)", xlabel="rho")
            fig.tight_layout()
            out = common.OUTPUT / "02_offline_raw_analysis.png"
            fig.savefig(out, dpi=110)
            print("picture:", out)
        except ImportError:
            print("(matplotlib not installed: no picture)")
    finally:
        s.stop()


if __name__ == "__main__":
    main()
