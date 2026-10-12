"""Abel inversion of radially binned angular data (rBASEX type). No Qt, no pixel grid.

Notation follows docs/notes/main.tex, sections "Abel Transformation / rBASEX" and
"Velocity Map Imaging / Data analysis with rBASEX":

  3D (Newton sphere)                            code
  ------------------------------------------   ------------------------------------------
  rho, theta'      radius, polar angle           rho[R] = R * dr          radii of the bins
  R                index of the radial basis     b_R(rho): hat, 1 at rho = R, 0 at R +- 1
  n                angular order                 orders = (0, 2, 4, ...)  power of cos(theta')
  b_{R;n} = b_R(rho) cos^n(theta')                the basis set
  I(rho, theta') = sum_n I_n(rho) cos^n(theta')   I_n[n, R]   returned by get_angular_intensity
  I_n(rho) = sum_R c_{R;n} b_R(rho),  and sampled at integer radii  I_{n rho} = c_{rho;n}

  2D (detector)
  ------------------------------------------   ------------------------------------------
  r, theta         radius, angle                 r = j * dr; theta_i of event i
  p_{R;n}(r)       projection of b_{R;n}         P_n(r) = sum_R c_{R;n} p_{R;n}(r) = (p_n^T I_n)(r)
  p_n^T            matrix, P_n(r) = p_n^T I_n    pT[n, r, R]       (projected_basis)
  (p_n^T)^-1       inverse Abel transform        pT_inv[n, R, r]
  d_{nr}           radially binned data of       d[n, r]           (RadialBinner.d)
                   angular order n
  H_{rni}          inverse Hankel matrix per     H[r, n, i]        P_{nr} = sum_i H_{rni} d_{ir}
                   radius
  C                coefficients of the Legendre  C[n, l] (legendre_matrix);  C_inv[l, n]
                   polynomials in its columns
  I(rho)           radial intensity              intensity[R] = 4 pi rho^2 sum_k C^-1_{0k} I_{k rho}
  beta_n           anisotropy                    beta[k, R] = L_n / L_0 for n = beta_orders[k] = orders[k + 1]
                                                 (so beta[0] is beta_2 when only even orders are used), with
                                                 L_n = sum_i C^-1_{ni} I_{i rho}   (legendre_coefficients)

The chain, in the document's indices (all matrices depend only on the number of bins and
the angular orders, as stated there):

    P_{nr}      = sum_i H_{rni} d_{ir}                                   (angular fit)
    I_{n rho}   = sum_j (p^T)^-1_{rho j;n} P_{nj}                        (inverse Abel transform)
    L_{n rho}   = sum_i C^-1_{ni} I_{i rho}                              (Legendre coefficients)
    I(rho)      = 4 pi rho^2 L_{0 rho},    beta_n(rho) = L_{n rho} / L_{0 rho}


Live use and bootstrap
----------------------
The expensive part (pT) depends only on the number of bins and the orders, and is cached (in
memory, optionally on disk). The inversion is linear in d and accepts leading batch dimensions
(bootstrap samples), costing a few matrix products. d is linear in the event weights: RadialBinner
keeps the per-event binning, so any weight vector (background subtraction, bootstrap) gives d
quickly.

Regularisation: the inverse Abel operator per order is built by _inverse_operator(); tikhonov > 0
replaces the exact inverse by a Tikhonov-regularised one. The further loss terms (statistical weighting,
smoothness of I(rho), non-negativity) are in abel_loss.py and act on all orders and radii together;
AbelBasis(loss=LossTerms(...)) switches to them.
"""
import os
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg

_trapezoid = getattr(np, "trapezoid", None) or np.trapz       # numpy 2 renamed trapz


# ---------------------------------------------------------------- angular part: K, H, C
def cos_power_integrals(max_order):
    """K[p] = integral over [0, 2 pi) of cos^p(theta): zero for odd p."""
    K = np.zeros(max_order + 1)
    K[0] = 2 * np.pi
    for p in range(2, max_order + 1, 2):
        K[p] = K[p - 2] * (p - 1) / p
    return K


def hankel_matrix(orders):
    """K[n, m] = integral of cos^(n+m)(theta): a Hankel matrix (it depends on n + m only).
    The Hankel matrix of main.tex for radial slice r is d_{r Hankel} = (c_r / area) * K."""
    n = np.asarray(orders)
    return cos_power_integrals(2 * int(n.max()))[n[:, None] + n[None, :]]


def angular_orders(n_max, even_only=True):
    """Angular orders n used. even_only drops the odd orders, which vanish for the reflection
    symmetry about the plane perpendicular to the laser axis (main.tex, Velocity Map Imaging)."""
    if even_only and n_max % 2:
        raise ValueError("n_max must be even when even_only=True")
    return tuple(range(0, n_max + 1, 2 if even_only else 1))


def legendre_matrix(orders):
    """C[n, l]: coefficient of x^n in the Legendre polynomial P_l, for the given orders.
    (main.tex: "the matrix C consisting of the coefficients of the first n Legendre polynomials
    in its columns"; C_inv = inv(C) turns coefficients of cos^n into Legendre coefficients.)"""
    C = np.zeros((len(orders), len(orders)))
    for col, l in enumerate(orders):
        c = npleg.leg2poly([0] * l + [1])
        for row, n in enumerate(orders):
            C[row, col] = c[n] if n < len(c) else 0.0
    return C


def bin_area(n_bins):
    """c_r = integral of r b_r(r) dr, in units of dr^2: r for r >= 1, 1/6 for the one-sided bin 0.
    The area of the hat-weighted radial slice r, divided by 2 pi."""
    c = np.arange(n_bins, dtype=float)
    c[0] = 1.0 / 6.0
    return c


# ------------------------------------------ projected basis p_n^T (the expensive part)
def projected_basis(n_bins, orders, nq=12, nt=24, max_elements=4_000_000):
    """pT[n, r, R] = (p_n^T)_{rR}: the order-n projection of the basis function b_{R;n}, averaged
    over the radial bin r with the weight r b_r(r) / c_r (what the binned data d_{nr} measure).
    Units dr = 1; it is dimensionless.

    The projection of b_{R;n} onto the plane (Abel transform with cos^n(theta') -> cos^n(theta)) is
        p_{R;n}(r) = 2 r^n  Int_r^inf  b_R(rho) rho^(1-n) / sqrt(rho^2 - r^2)  d rho
                   = 2 r  Int  b_R(r cosh t) (cosh t)^(1-n)  dt         (rho = r cosh t)
    The substitution removes the 1/sqrt singularity; Gauss-Legendre (nt points) in t on each linear
    piece of the hat, then Gauss-Legendre (nq points) in r over every unit segment between bin
    centres, where all kinks of p_{R;n}(r) lie."""
    nb = int(n_bins)
    orders = list(orders)
    xq, wq = np.polynomial.legendre.leggauss(nq)
    xq, wq = 0.5 * (xq + 1.0), 0.5 * wq
    xt, wt = np.polynomial.legendre.leggauss(nt)
    R = np.arange(nb, dtype=float)                                   # 3D basis index R
    pT = np.zeros((len(orders), nb, nb))
    block = max(1, int(max_elements // (nq * nb * nt)))
    for s0 in range(0, nb, block):
        seg = np.arange(s0, min(s0 + block, nb), dtype=float)       # detector radii r in [seg, seg + 1]
        r = (seg[:, None] + xq[None, :])[:, :, None]                 # (S, nq, 1)
        p_point = np.zeros((len(orders), len(seg), nq, nb))          # p_{R;n}(r) at those radii
        for rising in (True, False):                                 # the two linear pieces of b_R
            lower = (R - 1.0) if rising else R                       # piece covers [lower, lower + 1]
            rho_lo = np.maximum(lower[None, None, :], r)
            rho_hi = (lower + 1.0)[None, None, :] + 0 * r
            ok = (rho_hi > rho_lo) & ((R >= 1)[None, None, :] if rising else True)
            t_lo = np.arccosh(np.maximum(rho_lo / r, 1.0))
            t_hi = np.arccosh(np.maximum(rho_hi / r, 1.0))
            half = 0.5 * (t_hi - t_lo)
            t = half[..., None] * xt + 0.5 * (t_hi + t_lo)[..., None]            # (S, nq, nb, nt)
            ch = np.cosh(t)
            rho = r[..., None] * ch
            b = rho - (R[None, None, :, None] - 1.0) if rising else (R[None, None, :, None] + 1.0) - rho
            for j, n in enumerate(orders):
                integral = (b * ch ** (1 - n) * wt).sum(-1) * half
                p_point[j] += np.where(ok, 2.0 * r * integral, 0.0)
        # average over the bin r with its hat weight r b_r(r): two hats touch every unit segment
        w_fall = wq * (1.0 - xq) * (seg[:, None] + xq[None, :])
        w_rise = wq * xq * (seg[:, None] + xq[None, :])
        fall = np.einsum("sq,jsqR->jsR", w_fall, p_point)
        rise = np.einsum("sq,jsqR->jsR", w_rise, p_point)
        for i, s in enumerate(seg.astype(int)):
            pT[:, s, :] += fall[:, i, :]
            if s + 1 < nb:
                pT[:, s + 1, :] += rise[:, i, :]
    return pT / bin_area(nb)[None, :, None]


_PT_CACHE = {}


def get_pT(n_bins, orders, nq=12, nt=24, cache_dir=None):
    """projected_basis with an in-memory cache and an optional on-disk cache."""
    key = (n_bins, tuple(orders), nq, nt)
    if key in _PT_CACHE:
        return _PT_CACHE[key]
    path = None
    if cache_dir is not None:
        path = os.path.join(cache_dir, f"abel_pT_bins{n_bins}_orders{'-'.join(map(str, orders))}_q{nq}_t{nt}.npy")
        if os.path.exists(path):
            _PT_CACHE[key] = np.load(path)
            return _PT_CACHE[key]
    pT = projected_basis(n_bins, orders, nq, nt)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, pT)
    _PT_CACHE[key] = pT
    return pT


def pT_cached(n_bins, orders, nq=12, nt=24):
    """True if pT for these bins and orders is already built (so AbelBasis(...) is instant)."""
    return (n_bins, tuple(orders), nq, nt) in _PT_CACHE


def clear_cache():
    _PT_CACHE.clear()


def _inverse_operator(pT_n, tikhonov=0.0, diff_order=0):
    """(p_n^T)^-1 for one order: the matrix with  I_n(rho) = (p_n^T)^-1 P_n(r).

    The rows of pT differ by many orders of magnitude (high orders are suppressed near the centre
    as r^n), so the system is equilibrated first. tikhonov = 0 gives the exact inverse.
    tikhonov > 0 minimises |pT' I - P'|^2 + tikhonov^2 |Gamma I|^2 on the row-normalised system,
    Gamma = identity (diff_order 0) or a finite-difference operator of that order (smoothness).
    tikhonov is therefore relative: ~1e-3 .. 1e-1."""
    rs = np.abs(pT_n).max(axis=1)
    rs[rs == 0] = 1.0
    A = pT_n / rs[:, None]
    if tikhonov <= 0.0:
        cs = np.abs(A).max(axis=0)
        cs[cs == 0] = 1.0
        inv = np.linalg.inv(A / cs[None, :])
        return (inv / cs[:, None]) / rs[None, :]
    Gamma = np.eye(pT_n.shape[1])
    for _ in range(diff_order):
        Gamma = np.diff(Gamma, axis=0)
    lhs = A.T @ A + tikhonov ** 2 * (Gamma.T @ Gamma)
    return np.linalg.solve(lhs, A.T) / rs[None, :]


# --------------------------------------------------------------------------- results
@dataclass(frozen=True)
class AbelResult:
    rho: np.ndarray                   # radii of the bins, rho = R * dr  (R = 0 .. n_bins - 1)
    orders: tuple                     # angular orders n
    I_n: np.ndarray                   # (..., N, R)  I_{n rho}: coefficients of cos^n(theta')
    legendre_coefficients: np.ndarray # (..., N, R)  L_{n rho} = sum_i C^-1_{ni} I_{i rho}
    intensity: np.ndarray             # (..., R)     I(rho) = 4 pi rho^2 L_{0 rho}: counts per unit radius
    beta: np.ndarray                  # (..., N-1, R) beta_n = L_n / L_0 for n = orders[1:] (NaN where L_0 ~ 0)
    beta_orders: tuple

    def total(self):
        """Integral of I(rho) over rho (trapezoid on the bins): the total event weight."""
        return _trapezoid(self.intensity, self.rho, axis=-1)


class AbelBasis:
    """All matrices of the inversion, built once for (number of bins, angular orders).

    d has the shape (..., N, n_bins) with N = len(orders): d[n, r] = d_{nr}; leading batch
    dimensions (bootstrap samples) are free. area: see the module docstring, convention 2."""
    def __init__(self, n_bins, n_max=4, even_only=True, tikhonov=0.0, diff_order=0, area=1.0,
                 nq=12, nt=24, cache_dir=None, loss=None, covariance=None):
        self.n_bins = int(n_bins)
        self.orders = angular_orders(n_max, even_only)
        self.area = float(area)
        self.bin_area = bin_area(self.n_bins)                        # c_r
        K_inv = np.linalg.inv(hankel_matrix(self.orders))
        # H[r, n, i]: the exact inverse of d_{r Hankel} = (c_r / area) K, for every radius r
        self.H = self.area * K_inv[None, :, :] / self.bin_area[:, None, None]
        self.C = legendre_matrix(self.orders)
        self.C_inv = np.linalg.inv(self.C)
        self.pT = get_pT(self.n_bins, self.orders, nq, nt, cache_dir)
        self.joint = None
        self.set_regularisation(tikhonov, diff_order, loss, covariance)

    def set_regularisation(self, tikhonov=0.0, diff_order=0, loss=None, covariance=None):
        """Rebuild (p_n^T)^-1 (cheap: one solve per order). tikhonov: scalar or one value per order.

        loss: abel_loss.LossTerms. With any of its terms active the inversion is the joint one of
        abel_loss.JointInverter (all orders and radii at once, optionally constrained); covariance
        (RadialBinner.covariance of the measurement) is needed for statistical weighting and is kept
        fixed for everything inverted with this basis, bootstrap samples included."""
        lam = np.broadcast_to(np.asarray(tikhonov, float), (len(self.orders),))
        self.tikhonov, self.diff_order = lam.copy(), diff_order
        self.pT_inv = np.array([_inverse_operator(p, l, diff_order) for p, l in zip(self.pT, lam)])
        self.loss, self.joint = loss, None
        self._build_joint(covariance)

    def _build_joint(self, covariance=None):
        loss = self.loss
        if loss is not None and loss.active:
            from analysis.abel_loss import JointInverter
            self.joint = JointInverter(self.pT, self.H, self.C_inv, self.orders, self.tikhonov, self.diff_order,
                                       loss, covariance)

    def with_loss(self, loss, covariance=None):
        """A new AbelBasis that shares the matrices of this one but has other loss terms / covariance.
        (A new object, never a change in place: bootstrap runs hold on to the basis of their measurement.)"""
        import copy
        b = copy.copy(self)
        b.loss, b.joint = loss, None
        b._build_joint(covariance)
        return b

    @property
    def nonlinear(self):
        """True if the inversion is a constrained one (a bootstrap sample then costs an optimisation)."""
        return self.joint is not None and self.joint.nonlinear

    def loss_text(self):
        return "none" if self.joint is None else self.loss.text() + (
            "" if not self.joint.text() else "  [" + self.joint.text() + "]")

    # -- the steps of the chain ------------------------------------------------------
    def get_angular_intensity(self, d, dr=1.0):
        """d_{ij} -> I_{n rho} = sum_j (p^T)^-1_{rho j;n} sum_i H_{jni} d_{ij}   (main.tex).
        Angular fit with H (giving P_{nr}), then the inverse Abel transform per angular order.
        Division by dr^3 makes I_n a density (events per volume)."""
        d = np.asarray(d, float)
        P = np.einsum("rni,...ir->...nr", self.H, d)                  # P_{nr} = sum_i H_{rni} d_{ir}
        if self.joint is not None:
            return self.joint.solve(P) / dr ** 3
        return np.einsum("nRj,...nj->...nR", self.pT_inv, P) / dr ** 3

    def get_legendre_coefficients(self, I_n):
        """I_{n rho} -> L_{n rho} = sum_i C^-1_{ni} I_{i rho}: the coefficients of an expansion in
        Legendre polynomials; the 0'th term is I(rho)/4 pi."""
        return np.einsum("ni,...iR->...nR", self.C_inv, np.asarray(I_n))

    def invert(self, d, dr=1.0, min_intensity=0.0):
        """The whole chain. beta is NaN where |L_0| < min_intensity * max|L_0| (default: only where 0)."""
        I_n = self.get_angular_intensity(d, dr)
        L = self.get_legendre_coefficients(I_n)
        rho = np.arange(self.n_bins) * dr
        L0 = L[..., 0, :]
        intensity = 4.0 * np.pi * rho ** 2 * L0                       # I(rho) = 4 pi rho^2 L_0
        ref = np.abs(L0).max(axis=-1, keepdims=True)
        bad = np.abs(L0) <= np.maximum(min_intensity * ref, 1e-9 * ref + 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(bad[..., None, :], np.nan, L[..., 1:, :] / L0[..., None, :])
        return AbelResult(rho=rho, orders=self.orders, I_n=I_n, legendre_coefficients=L,
                          intensity=intensity, beta=beta, beta_orders=self.orders[1:])


def distribution_slice(I_n, orders, dr, rho_eval, theta, axis_angle=np.pi / 2):
    """The 3D distribution f(rho, theta') = sum_n I_n(rho) cos^n(theta') in the plane that contains the
    symmetry axis, on a grid: the "reconstructed image" of the velocity map.

    I_n[n, R] are the node values (hat basis: linear in between, zero beyond the node after the last),
    rho_eval the radii to evaluate at, theta the angles of the grid points in the detector plane
    (rad, counter-clockwise from +x) and axis_angle the direction of the symmetry axis, so that
    theta' = theta - axis_angle. Returns an array of shape (len(theta), len(rho_eval))."""
    I_n = np.asarray(I_n, float)
    nodes = np.arange(I_n.shape[-1] + 1) * dr                       # one extra node where the last hat ends
    cos = np.cos(np.asarray(theta, float) - axis_angle)
    f = np.zeros((len(cos), len(rho_eval)))
    for j, n in enumerate(orders):
        on_grid = np.interp(rho_eval, nodes, np.append(I_n[j], 0.0), right=0.0)
        f += (cos ** n)[:, None] * on_grid[None, :]
    return f


# ------------------------------------------------------------------ events -> d_{nr}
def cos_theta_from_xy(x, y, axis_angle=np.pi / 2):
    """(r_i, cos theta_i) of events about the centre. x, y are x_i - x_c and y_i - y_c (apply Shift,
    and Circularise, first). theta is measured from the laser axis, pointing at axis_angle
    (rad, counter-clockwise from +x): for axis_angle = pi/2 this is main.tex's
    cos(theta_i) = (y_i - y_c) / ((x_i - x_c)^2 + (y_i - y_c)^2)^(1/2)."""
    r = np.hypot(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (x * np.cos(axis_angle) + y * np.sin(axis_angle)) / r
    return r, np.where(r > 0, c, 0.0)


class RadialBinner:
    """Per-event linear radial binning, done once; the weights can then change at will.

    d(w)[n, r] = d_{nr} = sum_i w_i cos^n(theta_i) * (weight of event i in bin r), where an event
    with r <= r_i < r + dr contributes
        cos^n(theta_i) * [1 - (r_i - r)/dr]   to bin r       and
        cos^n(theta_i) * (r_i - r)/dr         to bin r + 1.
    Events with r_i >= n_bins * dr are ignored; their weight entry is skipped and alignment is kept.
    w_i are event weights: 1 for counts, background-subtraction weights, bootstrap weights.
    append() adds events as they arrive."""
    def __init__(self, n_bins, dr, orders):
        self.n_bins, self.dr, self.orders = int(n_bins), float(dr), tuple(orders)
        self.bin = np.zeros(0, np.int64)          # lower bin r of each event (-1: outside the range)
        self.frac = np.zeros(0)                   # (r_i - r) / dr
        self.cos_n = np.zeros((0, len(self.orders)))   # cos^n(theta_i) for every order n
        self.cos = np.zeros(0)                    # cos(theta_i) itself (for the covariance of d)
        self._sorted = None                       # cache of _by_bin()

    @property
    def n_events(self):
        return len(self.bin)

    def subset(self, rows):
        """A new binner holding only these events (index array or boolean mask over all events), in
        that order. d_subset(w[rows]) == d(w) for any w that is zero outside rows: a bootstrap on
        the events that carry weight does not have to carry the others around."""
        b = RadialBinner(self.n_bins, self.dr, self.orders)
        b.bin, b.frac, b.cos_n, b.cos = self.bin[rows], self.frac[rows], self.cos_n[rows], self.cos[rows]
        return b

    def snapshot(self):
        """A frozen view of the events binned so far. append() never changes arrays in place
        (it builds new ones), so the snapshot stays valid while this binner keeps growing: a
        bootstrap can resample it from another thread."""
        b = RadialBinner(self.n_bins, self.dr, self.orders)
        b.bin, b.frac, b.cos_n, b.cos = self.bin, self.frac, self.cos_n, self.cos
        return b

    def append(self, r, cos_theta):
        r = np.asarray(r, float)
        pos = r / self.dr
        finite = np.isfinite(pos)                              # events without a position are skipped
        lower = np.floor(np.where(finite, pos, -1.0)).astype(np.int64)
        ok = finite & (lower >= 0) & (lower < self.n_bins)
        frac = np.where(ok, pos - lower, 0.0)
        cos_n = np.asarray(cos_theta, float)[:, None] ** np.asarray(self.orders)[None, :]
        self.bin = np.concatenate([self.bin, np.where(ok, lower, -1)])
        self.frac = np.concatenate([self.frac, frac])
        self.cos_n = np.concatenate([self.cos_n, np.where(ok[:, None], cos_n, 0.0)])
        self.cos = np.concatenate([self.cos, np.where(ok, np.asarray(cos_theta, float), 0.0)])
        self._sorted = None

    def d(self, weights):
        """weights: (E,) -> d of shape (N, n_bins);  (B, E) -> (B, N, n_bins)."""
        w = np.asarray(weights, float)
        if len(w) != self.n_events and w.ndim == 1 or (w.ndim == 2 and w.shape[1] != self.n_events):
            raise ValueError(f"weights {w.shape} for {self.n_events} events")
        if w.ndim == 2:
            return self._d_batch(w)
        ok = self.bin >= 0
        r, frac, wi = self.bin[ok], self.frac[ok], w[ok]
        d = np.zeros((len(self.orders), self.n_bins))
        for j in range(len(self.orders)):
            wc = wi * self.cos_n[ok, j]
            d[j] = (np.bincount(r, wc * (1.0 - frac), self.n_bins)
                    + np.bincount(r + 1, wc * frac, self.n_bins + 1)[:self.n_bins])
        return d

    def covariance(self, weights):
        """Cov(d_{ir}, d_{jr}) of the radial moments from the events, assuming independent events:
        E[r, i, j] = sum_e w_e^2 cos^(n_i + n_j)(theta_e) g_e(r)^2, with g = 1 - frac for events whose lower
        bin is r and g = frac for events whose lower bin is r - 1. Shape (n_bins, N, N). Neighbouring radii
        share events and are correlated; that is ignored. weights: (E,)"""
        w = np.asarray(weights, float)
        if len(w) != self.n_events:
            raise ValueError(f"weights {w.shape} for {self.n_events} events")
        ok = self.bin >= 0
        b, f, w2, c = self.bin[ok], self.frac[ok], w[ok] ** 2, self.cos[ok]
        N, R = len(self.orders), self.n_bins
        E = np.zeros((R, N, N))
        for i in range(N):
            for j in range(i, N):
                v = w2 * c ** (self.orders[i] + self.orders[j])
                e = (np.bincount(b, v * (1.0 - f) ** 2, R)
                     + np.bincount(b + 1, v * f ** 2, R + 1)[:R])
                E[:, i, j] = E[:, j, i] = e
        return E

    def _by_bin(self):
        """Events sorted by their lower bin, with their contributions to that bin and to the next
        one for every order, interleaved: columns (n=0 lower, n=0 upper, n=1 lower, ...). Built on
        first use of a batch (a bootstrap) and then reused."""
        if getattr(self, "_sorted", None) is None:
            good = np.flatnonzero(self.bin >= 0)
            order = good[np.argsort(self.bin[good], kind="stable")]
            starts = np.searchsorted(self.bin[order], np.arange(self.n_bins + 1))
            f, c = self.frac[order], self.cos_n[order]
            V = np.empty((len(order), 2 * len(self.orders)))
            V[:, 0::2] = c * (1.0 - f)[:, None]
            V[:, 1::2] = c * f[:, None]
            self._sorted = (order, starts, V)
        return self._sorted

    def _d_batch(self, w, chunk=32):
        """d for many weight vectors at once: per radial bin one matrix product (BLAS) of the
        weights of the events in that bin with their contributions. About 6x faster than looping
        over the single-vector path (measured: 1.1 ms per resample for 142k events), which is what a
        bootstrap needs; the inversion of a resample costs about 0.1 ms on top."""
        order, starts, V = self._by_bin()
        N, R = len(self.orders), self.n_bins
        out = np.zeros((w.shape[0], N, R))
        for b0 in range(0, w.shape[0], chunk):
            ws = w[b0:b0 + chunk][:, order]
            part = out[b0:b0 + chunk]
            for r in range(R):
                s, e = starts[r], starts[r + 1]
                if e > s:
                    t = ws[:, s:e] @ V[s:e]
                    part[:, :, r] += t[:, 0::2]
                    if r + 1 < R:
                        part[:, :, r + 1] += t[:, 1::2]
        return out