"""The Abel inversion alone, on synthetic data: why the corrections exist.

No DAQ data, no pipeline: a known 3D distribution (a ring with beta_2 = 1.2) is projected, Poisson noise is added,
and the plain inverse is compared with the corrections of the Regularisation section. The plain inverse is exact
and therefore amplifies the noise (ringing, negative dips); the others trade a little bias for a stable result.

  python examples/08_abel_inversion_synthetic.py
"""
import numpy as np

import common  # noqa: F401  (makes src/ importable)
from analysis.abel import AbelBasis
from analysis.abel_loss import LossTerms

R = 60                                   # radial bins
orders = (0, 2)
basis0 = AbelBasis(R, n_max=2, even_only=True)
rho = np.arange(R)
ring = np.exp(-0.5 * ((rho - 30) / 2.5) ** 2)
beta2 = 1.2
# the 3D distribution f(rho, theta) = L_0 (1 + beta2 P_2(cos theta)) in the basis cos^n: I_n coefficients
P2 = (-0.5, 1.5)                         # P_2 = -1/2 + 3/2 cos^2
I_true = np.zeros((2, R))
I_true[0] = ring * (1 + beta2 * P2[0])
I_true[1] = ring * beta2 * P2[1]
P = np.einsum("nrj,nj->nr", basis0.pT, I_true)           # its projection: P_{nr}
rng = np.random.default_rng(1)
scale = 4000.0                                            # counts: more = less noise
P_noisy = rng.poisson(np.clip(P * scale, 0, None)) / scale
d = np.array([np.linalg.inv(basis0.H[r]) @ P_noisy[:, r] for r in range(R)]).T     # the radially binned data d_{nr}


def result(loss, tikhonov=0.0):
    b = AbelBasis(R, n_max=2, even_only=True, tikhonov=tikhonov, loss=loss)
    return b.invert(d, 1.0)


print(f"{'correction':<34s} {'rms error of L0':>16s} {'most negative I':>16s} {'beta2 at the ring':>18s}")
true_I = None
for label, kw in (("none (exact inverse)", dict(loss=None)),
                  ("Tikhonov damping 0.5", dict(loss=None, tikhonov=0.5)),
                  ("curvature smoothness 0.3", dict(loss=LossTerms(smooth=0.3))),
                  ("hard positivity I(rho) >= 0", dict(loss=LossTerms(positivity="intensity"))),
                  ("maximum entropy 0.3", dict(loss=LossTerms(entropy=0.3)))):
    res = result(**kw)
    L0_true = np.einsum("ni,ir->nr", basis0.C_inv, I_true)[0]
    err = float(np.sqrt(np.mean((res.legendre_coefficients[0] - L0_true) ** 2)) / L0_true.max())
    print(f"{label:<34s} {err:>16.3f} {res.legendre_coefficients[0].min() / L0_true.max():>16.3f} "
          f"{res.beta[0][30]:>18.2f}   (true {beta2})")
