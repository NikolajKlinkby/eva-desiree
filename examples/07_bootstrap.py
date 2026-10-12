"""Statistical error bands of I(rho) and beta_n(rho) by bootstrap, and saving them.

The bootstrap resamples the events (Poisson weights; or whole 1 s cycles with unit="cycle"), repeats the whole
inversion for each resample - with the same corrections as the measurement - and gives bias-corrected intervals.
It runs in a background thread; this script waits for it.

  python examples/07_bootstrap.py                  # 300 samples
  python examples/07_bootstrap.py --samples 2000 --entropy 0.3 --save examples/output/bootstrap.h5
"""
import argparse

import common
from storage import capture, h5file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--unit", choices=("event", "cycle"), default="event")
    ap.add_argument("--entropy", type=float, default=0.0, help="maximum-entropy strength (0 = exact inverse)")
    ap.add_argument("--save")
    args = ap.parse_args()

    s = common.start_session()
    try:
        if args.entropy:
            s.abel.set_loss("a", entropy=args.entropy)
        common.wait_settled(s)
        s.abel.set_bootstrap("a", n_boot=args.samples, unit=args.unit, n_sigma=1.0)
        s.abel.start_bootstrap("a")
        common.wait_until(lambda: not s.abel.bootstrap_running("a"), timeout=3600, what="bootstrap")
        b = s.abel.bootstrap_result("a")
        i = common.peak_index(s.abel_result().a.abel)
        print(b.summary(i))
        print(f"I(rho) at the peak (rho = {b.rho[i]:.4g}): {b.intensity.estimate[i]:.4g}  "
              f"[{b.intensity.lower[i]:.4g}, {b.intensity.upper[i]:.4g}]  (std {b.intensity.std[i]:.3g})")
        if b.beta is not None:
            for j, n in enumerate(b.beta_orders):
                print(f"beta_{n} at the peak: {b.beta.estimate[j, i]:+.3f}  "
                      f"[{b.beta.lower[j, i]:+.3f}, {b.beta.upper[j, i]:+.3f}]")
        print("all samples are kept: intensity_samples", b.intensity_samples.shape, "beta_samples", b.beta_samples.shape)
        if args.save:
            h5file.write_file(args.save, capture.capture(s, comment=f"bootstrap, {args.samples} samples"))
            print("written", args.save, "(bands under /plots/analysis/.../abel, samples under /results/bootstrap)")
    finally:
        s.stop()


if __name__ == "__main__":
    main()
