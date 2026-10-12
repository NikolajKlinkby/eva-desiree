"""Change settings of a running analysis and compare: cuts, coincidence window, Abel corrections.

Every stage has set_* methods (the GUI calls exactly these). A change makes the stage recompute from the same
raw data; wait_settled() waits for it. Here: one run, several variants, printed side by side.

  python examples/03_change_settings.py
"""
import numpy as np

import common


def peak_line(s, label):
    a = s.abel_result().a
    i = common.peak_index(a.abel)
    b = ", ".join(f"{x:+.3f}" for x in a.abel.beta[:, i])
    neg = float(np.minimum(a.abel.intensity, 0).sum() / max(a.abel.intensity.max(), 1e-12))
    print(f"{label:<44s} events {a.n_events:>7}  peak rho {a.rho[i]:.4g}  beta[{', '.join(map(str, a.abel.beta_orders))}] = {b}"
          f"   negative part {neg:+.3f}")


def main():
    s = common.start_session()
    try:
        common.wait_settled(s)
        det_a = s.config.det_a
        peak_line(s, "default")

        # --- charge-sum cut on the electron detector (None = open side), then back ---------------------
        s.sel.set_sum(det_a, 2000, None)
        common.wait_settled(s)
        peak_line(s, f"Sum cut det {det_a}: Sum >= 2000")
        s.sel.clear_sum(det_a)

        # --- a manual coincidence window instead of the automatic one (us, dt = t_B - t_A) -------------
        window = s.coincidence().window_us
        s.flt.set_window(window[0] + 0.5, window[1] - 0.5)
        common.wait_settled(s)
        peak_line(s, "coincidence window narrowed by 0.5 us each side")
        s.flt.clear_window()                                  # back to automatic
        common.wait_settled(s)

        # --- Abel inversion: the corrections of the Regularisation section -----------------------------
        for label, kw in (("exact inverse", {}),
                          ("statistical weights + curvature smoothness", dict(weighting=True, smooth=0.3)),
                          ("hard positivity: I(rho) >= 0", dict(positivity="intensity")),
                          ("maximum entropy 0.3", dict(entropy=0.3))):
            s.abel.set_regularisation("a", 0.0, 0)            # Tikhonov damping off
            s.abel.set_loss("a", weighting=False, smooth=0.0, positivity="off", entropy=0.0)
            s.abel.set_loss("a", **kw)
            common.wait_settled(s)
            peak_line(s, label)
        s.abel.set_loss("a", weighting=False, smooth=0.0, positivity="off", entropy=0.0)
        s.abel.set_regularisation("a", 0.5, 0)                # Tikhonov damping of strength 0.5
        common.wait_settled(s)
        peak_line(s, "Tikhonov damping 0.5")

        # --- binning and angular orders ----------------------------------------------------------------
        s.polar.set_binning("a", nr=100, ntheta=48, r_max=s.polar.settings.a.r_max)
        s.abel.set_orders("a", n_max=4, even_only=True)
        common.wait_settled(s)
        peak_line(s, "100 radial bins, even orders up to 4")
        print("\nstage settings are plain objects:", s.abel.settings.a)
    finally:
        s.stop()


if __name__ == "__main__":
    main()
