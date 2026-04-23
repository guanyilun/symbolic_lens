"""Unified search demo: compile BOTH estimator and A_L from the same
symbolic g, then rank candidates by both a Fisher-like FOM and a
data-driven cross-correlation with phi_true.

Key result: the unified pipeline works (compile_qe returns estimator,
A_L_fn, N_L_phi_fn from one symbolic g).  A_L is bit-for-bit vs
norm_lens.qtt.  The two scoring FOMs give different orderings,
reflecting different questions:

  Cross-correlation: which candidate best reconstructs phi_true from
                     THIS data?  (requires an oracle for phi.)
  Fisher N_L^phi   : which candidate has the tightest self-consistent
                     reconstruction noise?  (no data needed, but
                     gauge-dependent under g -> alpha·g.)

A production search tool should use cross-correlation when a reference
phi exists (simulations, or cross with other probes), and Fisher as a
regularizer / validity check.
"""
import numpy as np
import healpy as hp
from sympy import sympify, sqrt as sp_sqrt

from pixell import enmap, curvedsky as cs, lensing, utils as u

from sym_utils.l12_sum import l, l1, l2, wigner_3j, P
from sym_utils.namikawa import (
    hCT, CT, gamma_f, a, a_plus, a_minus, q_plus,
)
from sym_utils.compile_qe import compile_qe
from sym_utils.estimator_native import Pixelization, scalar_pair


# --- simulate lensed CMB ---
LMAX = 200
ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
clphi_theory = 1e-7 / (ell + 10) ** 2
clphi_theory[:2] = 0

T_unl = cs.rand_alm(cltt_theory, lmax=LMAX, seed=1)
phi_true = cs.rand_alm(clphi_theory, lmax=LMAX, seed=2)
shape, wcs = enmap.fullsky_geometry(res=15 * u.arcmin)
lensed_map = lensing.lens_map_curved(
    (1,) + tuple(shape), wcs, phi_true, T_unl[None, :], spin=[0]
)[0]
T_lensed = cs.map2alm(lensed_map[0], lmax=LMAX)

ocltt = cltt_theory + 1.0
spec = {"hCT": ocltt, "CT": cltt_theory}
px = Pixelization(nside=256)


# --- candidate grammar ---
def swap_l12(e): return e.subs({l1: l2, l2: l1}, simultaneous=True)

def f_TT(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_wrong_ladder(c=1):
    W = -2 * a_plus(l) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_wrong_m(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 0, 1, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_no_response(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W + swap_l12(W)

candidates = {
    "f_TT (Hu-Okamoto correct)"      : f_TT()            / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong ladder (a(l,2))"     : f_wrong_ladder()  / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong 3j m-row"            : f_wrong_m()       / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT no CT response"            : f_no_response()   / (sympify(2) * hCT(l1) * hCT(l2)),
    "garbage (trivial 3j)"           : gamma_f(l1, l, l2) * wigner_3j(l, l1, l2, 0, 0, 0)
                                       / (hCT(l1) * hCT(l2)),
}


# --- compile each: get (estimator, A_L, N_L^phi) ---
compiled = {}
for name, g in candidates.items():
    try:
        est, A_L, N_L_phi = compile_qe(
            g, LMAX, Delta=2, hCxx=hCT, hCyy=hCT, user_funcs=[hCT, CT], px=px,
        )
        compiled[name] = (est, A_L, N_L_phi, None)
    except Exception as e:
        compiled[name] = (None, None, None, f"{type(e).__name__}: {str(e)[:60]}")


# --- scoring: cross-correlate the A_L-NORMALIZED candidate output with phi_true.
# A_L · psi is the unbiased estimator for phi under the candidate's hypothesis;
# this normalization makes the comparison SCALE-INVARIANT under g -> αg.
def xcorr_score(est_fn, A_L_fn, L_lo=40, L_hi=180):
    out = est_fn(scalar_pair(T_lensed), scalar_pair(T_lensed), spec)
    psi = np.asarray(out[1] if 1 in out else next(iter(out.values())))
    # Apply A_L normalization: phi_hat = A_L · psi  (unbiased for phi)
    A_L = A_L_fn(ocltt, cltt_theory)
    A_L = np.where(np.isfinite(A_L) & (A_L > 0), A_L, 0.0)
    phi_hat = hp.almxfl(psi, A_L)
    cl_h = hp.alm2cl(phi_hat); cl_phi = hp.alm2cl(phi_true)
    cl_x = hp.alm2cl(phi_hat, phi_true)
    Ls = np.arange(L_lo, L_hi + 1)
    valid = (cl_h[Ls] > 0) & (cl_phi[Ls] > 0)
    r_L = cl_x[Ls][valid] / np.sqrt(cl_h[Ls][valid] * cl_phi[Ls][valid])
    w = 2 * Ls[valid] + 1
    return np.sum(w * r_L) / np.sum(w)


def fisher_score(N_L_phi_fn, L_lo=40, L_hi=180):
    """Integrated 1/N_L^phi."""
    N = N_L_phi_fn(ocltt, cltt_theory)
    Ls = np.arange(L_lo, L_hi + 1)
    good = np.isfinite(N[Ls]) & (N[Ls] > 0)
    return np.sum((2 * Ls[good] + 1) / N[Ls[good]])


print("=" * 90)
print("Unified compile: estimator + A_L + N_L^phi from one symbolic g")
print("=" * 90)
print()
print(f"{'candidate':<40s} {'<r(L)>':>10s} {'1/N_L^phi (F)':>16s} {'A_L(50)':>12s}")
print("-" * 90)

results = []
for name, (est, A_L_fn, N_L_phi_fn, err) in compiled.items():
    if err is not None:
        results.append((name, 0, 0, np.inf, err))
        continue
    r = xcorr_score(est, A_L_fn)
    F = fisher_score(N_L_phi_fn)
    A50 = float(A_L_fn(ocltt, cltt_theory)[50])
    results.append((name, r, F, A50, None))

for name, r, F, A50, err in results:
    if err:
        print(f"{name:<40s}  {'FAILED':>10s}")
        continue
    print(f"{name:<40s} {r:>+10.4f} {F:>16.3e} {A50:>12.3e}")

print()
print("Rank by cross-correlation (needs oracle):")
for i, t in enumerate(sorted(results, key=lambda t: -abs(t[1]))):
    print(f"  {i+1}. {t[0]}  (r={t[1]:+.3f})")

print()
print("Rank by Fisher (no oracle needed, but gauge-dependent under g->αg):")
for i, t in enumerate(sorted(results, key=lambda t: -t[2])):
    print(f"  {i+1}. {t[0]}  (F={t[2]:.2e})")
