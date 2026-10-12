"""Additional loss terms for the Abel inversion: statistical weighting, smoothness of I(rho),
non-negativity (no Qt; needs scipy for the Cholesky factorisations).

Why
---
The plain rBASEX inverse is exact: it reproduces the measured moments d_{nr} to rounding and so also
every statistical fluctuation of them, which the inverse Abel transform amplifies. The typical
symptom is ringing, with a negative dip of I(rho) in front of a peak. Tikhonov (abel._inverse_operator)
damps it order by order. The terms here work on the whole inversion at once:

  minimise  chi2(c)  +  sum_n lambda_n^2 |Gamma c_n|^2  +  lambda_s^2 sum_l w_l |D (rho_hat^2 L_l)|^2
  subject to  (optionally)  I(rho) >= 0,  or  f(rho, theta') >= 0 for all theta'

  c[n, R]    the unknowns I_{n rho}: coefficients of cos^n(theta') on the radial hat basis
  chi2       sum_r (P'_r - A_r c)^T W_r (P'_r - A_r c): the data term, in the row-equilibrated units
             of abel._inverse_operator (P' = P / rs, A = pT / rs). W_r = 1 (unweighted), or the
             inverse covariance of P'_r estimated from the events (statistical weighting)
  Gamma      identity or a finite-difference operator: the existing Tikhonov term (per order)
  smoothness D = finite difference of order 2 (curvature) or 1 along rho; acts on rho_hat^2 L_l(rho),
             i.e. on I(rho) / (4 pi rho_ref^2), rho_hat = rho / rho_max. l = 0 (the intensity) only, or
             every Legendre order l. lambda_s is a relative strength in the same units as lambda
  positivity "intensity": L_0(rho) >= 0 at every bin;  "distribution": the full angular
             distribution sum_n c_n cos^n(theta') >= 0 for 24 angles at every bin

Maximum entropy (Poisson entropy)
---------------------------------
  minimise  chi2(c) + ... + alpha^2 F^2 sum_{r,k} w_k s(f_{rk}/F),   s(x) = x ln(x/m) - x + m
with f_{rk} = sum_n c_{nr} cos^n(theta_k) the angular distribution at radius r and angle theta_k (16 angles),
m the default level (a fraction of the peak F of the unconstrained solution) and alpha a relative strength.
s is the Skilling entropy of a positive additive distribution, i.e. the entropy that follows from Poisson
counting statistics: its curvature 1/f makes the tails stiff and the peaks soft, as the noise of counts
is. It keeps f positive without a hard constraint (the solution is pulled to the default level where the
data say nothing). The data term stays the chi2 with the event covariance: the Abel moments are weighted sums
with negative accidental-subtraction weights, they are not counts, so a Poisson likelihood of the data is
not defined; the entropy is where the Poisson statistics enter. One smooth convex problem per sample,
solved by damped Newton steps (one dense Cholesky per step).

Statistical weighting
---------------------
An event with weight w, radius fraction f and cos(theta) = x adds w x^i (1 - f) to d_{i,b} and
w x^i f to d_{i,b+1}. Hence Cov(d_{ir}, d_{jr}) = sum_e w_e^2 x_e^(i+j) g_e(r)^2, summed over the events
that touch bin r (RadialBinner.covariance; this includes the negative weights of the accidental
subtraction, whose noise adds). The covariance of P_r = H_r d_r follows, and of P'_r after
the equilibration. It is smoothed over 5 neighbouring radii and given a floor of 2 % of its mean (a bin
with one or two events would otherwise get an enormous weight: weights from a noisy variance estimate
are biased towards bins that happened to fluctuate low). W_r is normalised to unit mean, so lambda keeps
its meaning. Without a penalty or constraint the weights do nothing (the system is square and the
fit exact); they decide where the penalties bite: the noisy peak is smoothed more than the quiet flank.

Linear and nonlinear
--------------------
Weighting, Tikhonov and smoothness keep the inversion a linear map P -> c with a fixed matrix. It is
built once per measurement (a dense (N R)^2 Cholesky factorisation, milliseconds to a second) and
applied to a whole bootstrap batch with one matrix product. The covariance used for the weights is that
of the measurement and stays the same for all bootstrap samples (the weights are part of the
estimator's definition, not of the data that is resampled).

Positivity makes it a quadratic programme per sample. It is solved with ADMM (Boyd et al. 2011, 3.1):
the matrix (Q + rho A^T A) is factorised once; each iteration is one matrix product for the whole
batch, so a bootstrap of 2000 samples is minutes, not hours. Samples whose unconstrained solution
already satisfies the constraint are done at once and leave the iteration.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

POSITIVITY = ("off", "intensity", "distribution")


@dataclass(frozen=True)
class LossTerms:
    weighting: bool = False           # statistical weights W_r from the event covariance
    smooth: float = 0.0               # lambda_s; 0 = off
    smooth_order: int = 2             # 1: gradient, 2: curvature
    smooth_all: bool = False          # also on the higher Legendre orders (default: the intensity only)
    positivity: str = "off"           # "off" | "intensity" | "distribution"
    entropy: float = 0.0              # alpha of the maximum-entropy term; 0 = off
    entropy_floor: float = 0.01       # default level m, as a fraction of the peak of the distribution

    def __post_init__(self):
        if self.positivity not in POSITIVITY:
            raise ValueError(f"positivity must be one of {POSITIVITY}")
        if self.smooth_order not in (1, 2, 3):
            raise ValueError("smooth_order must be 1, 2 or 3")
        if self.entropy < 0 or not (0.0 < self.entropy_floor < 1.0):
            raise ValueError("entropy must be >= 0 and entropy_floor in (0, 1)")
        if self.entropy > 0 and self.positivity != "off":
            raise ValueError("maximum entropy already keeps the distribution positive: switch the hard positivity constraint off")

    @property
    def active(self):
        return self.weighting or self.smooth > 0.0 or self.positivity != "off" or self.entropy > 0.0

    @property
    def nonlinear(self):
        return self.positivity != "off" or self.entropy > 0.0

    def text(self):
        parts = []
        if self.weighting:
            parts.append("statistical weights")
        if self.smooth > 0:
            kind = {1: "gradient", 2: "curvature", 3: "3rd difference"}[self.smooth_order]
            parts.append(f"{kind} smoothness {self.smooth:g}" + (" (all Legendre orders)" if self.smooth_all else ""))
        if self.positivity == "intensity":
            parts.append("I(rho) >= 0")
        elif self.positivity == "distribution":
            parts.append("f(rho, theta) >= 0")
        if self.entropy > 0:
            parts.append(f"maximum entropy {self.entropy:g} (default level {self.entropy_floor:.3g} of peak)")
        return ", ".join(parts) if parts else "none"


def diff_matrix(n, order):
    """(n - order, n) finite-difference operator of the given order."""
    return np.diff(np.eye(n), n=order, axis=0)


def _smooth_radially(E, width):
    """Moving average over axis 0 (radius) with edge renormalisation."""
    if width <= 1:
        return E
    k = np.ones(width)
    norm = np.convolve(np.ones(E.shape[0]), k, mode="same")
    flat = E.reshape(E.shape[0], -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(flat[:, j], k, mode="same") / norm
    return out.reshape(E.shape)


def statistical_weights(cov, H, S, floor=2e-3, width=5):
    """W[r] (N, N): inverse covariance of P'_r = S_r * (H_r d_r), normalised to unit geometric mean.

    cov[r, i, j] = Cov(d_{ir}, d_{jr}) (RadialBinner.covariance); H[r, n, i] and S[n, r] from the basis.
    The covariance of the angular moments is very anisotropic (the 4th-order coefficient is far noisier than
    the 0th), so the weights are too: that is the point, they tell the penalties which combinations of
    the data to trust. floor (relative to the mean variance) keeps empty bins finite.
    Returns (W, text); (identity, text) if the covariance carries no information."""
    R, N, _ = cov.shape
    E = _smooth_radially(cov, width)
    SigP = np.einsum("rni,rij,rmj->rnm", H, E, H)
    St = S.T                                                   # (R, N)
    Sig = SigP * St[:, :, None] * St[:, None, :]
    tr = np.trace(Sig, axis1=1, axis2=2) / N
    ok = tr > 0
    if not ok.any():
        return np.broadcast_to(np.eye(N), (R, N, N)).copy(), "no events: unweighted"
    tau = tr[ok].mean()
    Sig = Sig + floor * tau * np.eye(N)[None]
    W = np.linalg.inv(Sig)
    W = 0.5 * (W + np.swapaxes(W, 1, 2))
    logdet = np.linalg.slogdet(W)[1]
    W = W / np.exp(logdet[ok].mean() / N)                       # unit geometric mean of the eigenvalues
    ev = np.linalg.eigvalsh(W[ok])
    return W, f"weights {ev.min():.2g} .. {ev.max():.2g} (geometric mean 1)"


class _Spd:
    """Cholesky factorisation of a symmetric positive-definite matrix with Jacobi scaling and a tiny
    ridge (retried larger if the matrix is numerically indefinite)."""
    def __init__(self, K):
        n = K.shape[0]
        d = np.sqrt(np.maximum(np.diag(K), 1e-300))
        Ks = K / np.outer(d, d)
        ridge = 1e-12
        for _ in range(10):
            try:
                self.cf = cho_factor(Ks + ridge * np.eye(n), lower=True)
                break
            except LinAlgError:
                ridge *= 100.0
        else:
            raise LinAlgError("the Abel loss matrix is not positive definite")
        self.d, self.ridge, self.n = d, ridge, n

    def solve(self, B):
        """K^-1 B for B of shape (n, m)."""
        return cho_solve(self.cf, B / self.d[:, None]) / self.d[:, None]

    def inverse(self):
        return self.solve(np.eye(self.n))


class JointInverter:
    """The joint inversion  P_{nr} -> I_{n rho}  with the loss terms above (see the module docstring).

    pT (N, R, R), H (R, N, N), C_inv (N, N): the matrices of AbelBasis. tikhonov: lambda per order;
    cov: RadialBinner.covariance of the measurement (needed for weighting)."""
    def __init__(self, pT, H, C_inv, orders, tikhonov, diff_order, loss, cov=None,
                 admm_tol=1e-5, admm_maxit=3000, positivity_angles=16, admm_rho=None):
        N, R, _ = pT.shape
        self.N, self.R, self.loss, self.orders = N, R, loss, tuple(orders)
        rs = np.abs(pT).max(axis=2)
        rs[rs == 0] = 1.0
        self.S = 1.0 / rs                                                      # (N, R)
        A = pT * self.S[:, :, None]                                            # A[n, r, R]
        self.weights_info = ""
        if loss.weighting and cov is not None:
            W, self.weights_info = statistical_weights(cov, H, self.S)
        else:
            W = np.broadcast_to(np.eye(N), (R, N, N)).copy()
            if loss.weighting:
                self.weights_info = "no covariance: unweighted"
        self.W = W
        # --- the quadratic form Q (data + Tikhonov + smoothness)
        Q = np.zeros((N * R, N * R))
        for n in range(N):
            for m in range(n, N):
                blk = A[n].T @ (W[:, n, m, None] * A[m])
                Q[n * R:(n + 1) * R, m * R:(m + 1) * R] = blk
                if m != n:
                    Q[m * R:(m + 1) * R, n * R:(n + 1) * R] = blk.T
        lam = np.broadcast_to(np.asarray(tikhonov, float), (N,))
        for n in range(N):
            if lam[n] > 0:
                Gam = np.eye(R)
                for _ in range(diff_order):
                    Gam = np.diff(Gam, axis=0)
                Q[n * R:(n + 1) * R, n * R:(n + 1) * R] += lam[n] ** 2 * (Gam.T @ Gam)
        if loss.smooth > 0:
            D = diff_matrix(R, loss.smooth_order)
            rho2 = (np.arange(R) / max(R - 1, 1)) ** 2
            Bm = rho2[:, None] * (D.T @ D) * rho2[None, :]
            om = np.ones(N) if loss.smooth_all else (np.arange(N) == 0).astype(float)
            kappa = (C_inv.T * om) @ C_inv                                     # sum_l w_l Cinv[l,n] Cinv[l,m]
            Q += loss.smooth ** 2 * np.kron(kappa, Bm)
        self.Q = Q
        # data -> gradient term:  g = MtW @ P'
        self.MtW = np.einsum("nrR,rnm->nRmr", A, W).reshape(N * R, N * R)
        spd = _Spd(Q)
        self.ridge = spd.ridge
        self.G = spd.d[:, None] ** -1 * cho_solve(spd.cf, self.MtW / spd.d[:, None])      # Q^-1 MtW
        self.G_T, self.MtW_T = np.ascontiguousarray(self.G.T), np.ascontiguousarray(self.MtW.T)
        # --- positivity
        self.positivity = loss.positivity
        self.admm_tol, self.admm_maxit, self.admm_rho_factor = admm_tol, admm_maxit, admm_rho
        self.info = {}
        self.entropy = float(loss.entropy)
        self.entropy_floor = float(loss.entropy_floor)
        if self.positivity != "off":
            self._setup_constraint(C_inv, positivity_angles)
        if self.entropy > 0:
            self._setup_entropy(positivity_angles)

    # ------------------------------------------------------------------ constraint set-up
    def _setup_constraint(self, C_inv, n_angles):
        """ADMM works in the Jacobi-scaled variables y = D c (D^2 = diag Q), where Q has a unit
        diagonal, and each constraint row is normalised: the radii carry very different counts, and
        a single penalty for all of them converges 10-100 times slower."""
        N, R = self.N, self.R
        if self.positivity == "intensity":
            Am = C_inv[0:1, :].copy()                                         # L_0 = sum_i C^-1_{0i} c_i
        else:
            even = all(n % 2 == 0 for n in self.orders)
            t = np.linspace(0.0, np.pi / 2 if even else np.pi, n_angles)
            Am = np.cos(t)[:, None] ** np.asarray(self.orders, float)[None, :]
        self.D = np.sqrt(np.maximum(np.diag(self.Q), 1e-300))                 # (N R,), index n * R + r
        D = self.D.reshape(N, R)
        Ay = Am[:, None, :] / D.T[None, :, :]                                 # (K, R, N): rows in the y variables
        Ay = Ay / np.maximum(np.linalg.norm(Ay, axis=2, keepdims=True), 1e-300)
        self.Ay = Ay
        rho = self.admm_rho_factor if self.admm_rho_factor else (0.3 if self.positivity == "intensity" else 1.0)
        Ks = self.Q / np.outer(self.D, self.D)
        for r in range(R):
            ix = np.arange(N) * R + r
            Ks[np.ix_(ix, ix)] += rho * (Ay[:, r, :].T @ Ay[:, r, :])
        self.admm_rho = rho
        self.Kinv = np.ascontiguousarray(_Spd(Ks).inverse())
        self.G_y_T = np.ascontiguousarray((self.G * self.D[:, None]).T)         # P' -> y0 = D c0
        self.MtW_y_T = np.ascontiguousarray((self.MtW / self.D[:, None]).T)     # P' -> g' = g / D

    def _A(self, y):                                                           # (B, N, R) -> (B, K, R)
        return np.einsum("krn,bnr->bkr", self.Ay, y)

    def _At(self, z):                                                          # (B, K, R) -> (B, N, R)
        return np.einsum("krn,bkr->bnr", self.Ay, z)

    # ------------------------------------------------------------------ the map
    @property
    def nonlinear(self):
        return self.positivity != "off" or self.entropy > 0

    def solve(self, P):
        """P: (..., N, R) the angular-fit moments P_{nr} -> c = I_{n rho}: (..., N, R)."""
        P = np.asarray(P, float)
        shp = P.shape[:-2]
        B = int(np.prod(shp)) if shp else 1
        Pp = (P * self.S).reshape(B, -1)
        c = Pp @ self.G_T
        if self.positivity != "off":
            c = self._constrain(Pp, c, record=(B == 1))
        elif self.entropy > 0:
            c = self._maxent(Pp, c, record=(B == 1))
        return c.reshape(shp + (self.N, self.R))

    def _constrain(self, Pp, c0, record=False):
        """Quadratic programme  min 1/2 c^T Q c - g^T c  s.t.  A c >= 0  by ADMM, from the
        unconstrained solution c0, for every row of the batch that violates the constraint."""
        N, R = self.N, self.R
        B = c0.shape[0]
        y0 = (c0 * self.D).reshape(B, N, R)
        Ay0 = self._A(y0)
        scale = np.abs(Ay0).max(axis=(1, 2), keepdims=True)
        bad = np.flatnonzero((Ay0 < -1e-9 * np.maximum(scale, 1e-300)).any(axis=(1, 2)))
        out = c0.copy()
        iters, unconverged = 0, 0
        if len(bad):
            g = Pp[bad] @ self.MtW_y_T
            yb = y0.reshape(B, -1)[bad].copy()
            z = np.maximum(Ay0[bad], 0.0)
            u = np.zeros_like(z)
            rho, alpha, tol = self.admm_rho, 1.6, self.admm_tol
            idx = bad.copy()
            for it in range(self.admm_maxit):
                n_ = len(idx)
                rhs = g + rho * self._At(z - u).reshape(n_, -1)
                yn = rhs @ self.Kinv
                Ayn = self._A(yn.reshape(n_, N, R))
                Ah = alpha * Ayn + (1.0 - alpha) * z
                zn = np.maximum(Ah + u, 0.0)
                u = u + Ah - zn
                r_norm = np.linalg.norm((Ayn - zn).reshape(n_, -1), axis=1)
                s_norm = rho * np.linalg.norm(self._At(zn - z).reshape(n_, -1), axis=1)
                nA = np.maximum(np.linalg.norm(Ayn.reshape(n_, -1), axis=1),
                                np.linalg.norm(zn.reshape(n_, -1), axis=1))
                nU = rho * np.linalg.norm(self._At(u).reshape(n_, -1), axis=1)
                z, yb = zn, yn
                done = (r_norm <= tol * nA + 1e-14) & (s_norm <= tol * nU + 1e-14 * (1.0 + nU))
                iters = it + 1
                if done.any():
                    out[idx[done]] = yb[done] / self.D
                    keep = ~done
                    idx, yb, z, u, g = idx[keep], yb[keep], z[keep], u[keep], g[keep]
                    if not len(idx):
                        break
            if len(idx):
                out[idx] = yb / self.D
                unconverged = len(idx)
        if record:
            Aout = self._A((out[:1] * self.D).reshape(1, N, R))
            self.info = {"violating": int(len(bad)), "iterations": iters, "unconverged": unconverged,
                         "active_bins": int(np.sum(Aout[0] < 1e-6 * max(np.abs(Aout).max(), 1e-300))),
                         "min_relative": float(Aout.min() / max(np.abs(Aout).max(), 1e-300))}
        return out

    # ------------------------------------------------------------------ maximum entropy
    def _setup_entropy(self, n_angles):
        even = all(n % 2 == 0 for n in self.orders)
        t = np.linspace(0.0, np.pi / 2 if even else np.pi, n_angles)
        self.V = np.cos(t)[:, None] ** np.asarray(self.orders, float)[None, :]      # (K, N)
        self.wk = np.full(n_angles, 1.0 / n_angles)
        if 0 not in self.orders:
            raise ValueError("maximum entropy needs the order n = 0")
        self.i0 = self.orders.index(0)

    def _maxent(self, Pp, c0, record=False):
        """Rows of the batch one by one: damped Newton on the convex problem (see the module docstring)."""
        out = np.empty_like(c0)
        g_all = Pp @ self.MtW_T
        its, bad = [], 0
        for b in range(c0.shape[0]):
            out[b], it, ok = self._maxent_one(g_all[b], c0[b])
            its.append(it)
            bad += not ok
        if record:
            f = np.einsum("kn,nr->kr", self.V, out[0].reshape(self.N, self.R))
            top = max(f.max(), 1e-300)
            self.info = {"iterations": int(max(its)), "unconverged": int(bad),
                         "active_bins": int(np.sum(f < 2.0 * self.entropy_floor * top)), "min_relative": float(f.min() / top)}
        return out

    def _maxent_one(self, g, c0, tol=1e-5, maxit=100):
        N, R, V, wk, a2 = self.N, self.R, self.V, self.wk, self.entropy ** 2
        f0 = V @ c0.reshape(N, R)
        F = float(f0.max())
        if not np.isfinite(F) or F <= 0:
            return c0, 0, True
        m = self.entropy_floor
        # feasible start on the line from a flat positive distribution to the unconstrained solution
        cflat = np.zeros((N, R)); cflat[self.i0, :] = 2.0 * m * F
        fflat = V @ cflat
        d = f0 - fflat
        with np.errstate(divide="ignore", invalid="ignore"):
            tmax = np.where(d < 0, (fflat - 0.5 * m * F) / -d, np.inf).min()
        c = (cflat + min(1.0, 0.99 * tmax) * (c0.reshape(N, R) - cflat)).reshape(-1)
        scale = max(abs(0.5 * g @ c0), 1e-300)
        ar = np.arange(R)

        def J(cv, f):
            x = f / F
            return 0.5 * cv @ (self.Q @ cv) - g @ cv + a2 * F * F * np.sum(
                wk[:, None] * (x * np.log(x / m) - x + m))

        f = V @ c.reshape(N, R)
        Jc = J(c, f)
        for it in range(maxit):
            x = f / F
            grad = self.Q @ c - g
            grad = grad + (a2 * F * np.einsum("k,kr,kn->nr", wk, np.log(x / m), V)).reshape(-1)
            ent = a2 * np.einsum("k,kn,km,kr->rnm", wk, V, V, 1.0 / x)
            H = self.Q.copy()
            H4 = H.reshape(N, R, N, R)
            H4[:, ar, :, ar] += ent
            step = -_Spd(H).solve(grad[:, None])[:, 0]
            dec = -grad @ step
            if dec <= tol * scale:
                return c, it, True
            df = V @ step.reshape(N, R)
            with np.errstate(divide="ignore", invalid="ignore"):
                amax = np.where(df < 0, f / -df, np.inf).min()
            a = min(1.0, 0.995 * amax)
            for _ in range(40):
                cn = c + a * step
                fn = V @ cn.reshape(N, R)
                if fn.min() > 0:
                    Jn = J(cn, fn)
                    if Jn <= Jc - 1e-4 * a * dec:
                        break
                a *= 0.5
            else:                                   # no further decrease possible: round-off limit
                return c, it, dec <= 1e-3 * scale
            c, f, Jc = cn, fn, Jn
        return c, maxit, False

    def text(self):
        parts = []
        if self.weights_info:
            parts.append("weights: " + self.weights_info)
        if self.positivity != "off" and self.info:
            i = self.info
            parts.append(f"positivity: {'unconstrained solution already positive' if not i['violating'] else str(i['iterations']) + ' ADMM iterations'}"
                         f", {i['active_bins']} constraints active" + (f", {i['unconverged']} not converged" if i["unconverged"] else ""))
        if self.entropy > 0 and self.info:
            i = self.info
            parts.append(f"maximum entropy: {i['iterations']} Newton steps" + (f", {i['unconverged']} not converged" if i["unconverged"] else ""))
        return "; ".join(parts)
