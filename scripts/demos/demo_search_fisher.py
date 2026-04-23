"""Expression-space search for CMB lensing QE — Fisher-scored version.

Uses compile_qe to emit both the estimator and the symbolic A_L for
each candidate weight.  Ranks candidates by Fisher information
(lower A_L = better reconstruction noise = higher SNR on phi_true).

Compare to demo_search_lensing.py which ranks by one-realization
cross-correlation — that's noisier and the gap between correct and
close-wrong candidates is subtle.  Fisher scoring uses the full
mode-counting analytic normalization, so the gap is sharp.
"""
import numpy as np
import healpy as hp
from sympy import sympify, sqrt as sp_sqrt

from symqe.engine.l12_sum import l, l1, l2, wigner_3j, P
from symqe.engine.namikawa import (
    hCT, CT, gamma_f, a, a_plus, a_minus, q_plus,
)
from symqe.engine.compile_qe import compile_qe
from symqe.engine.estimator_native import Pixelization, scalar_pair

LMAX = 200   # keep modest so normalization compile is quick
ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt_theory + 1.0
px = Pixelization(nside=256)


# --- candidate grammar (same as cross-correlation demo) ---
def swap_l12(expr):
    return expr.subs({l1: l2, l2: l1}, simultaneous=True)

def f_TT_correct(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_TT_wrong_ladder(c=1):
    W = -2 * a_plus(l) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_TT_wrong_m(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 0, 1, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_TT_no_response(c=1):
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W + swap_l12(W)


candidates = {
    "f_TT (Hu-Okamoto correct)"      : f_TT_correct()       / (sympify(2) * hCT(l1) * hCT(l2)),
    "5·f_TT (scaled)"                : 5 * f_TT_correct()   / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong ladder (a(l,2))"     : f_TT_wrong_ladder()  / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong 3j m-row (0,1,-1)"   : f_TT_wrong_m()       / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT with no CT response"       : f_TT_no_response()   / (sympify(2) * hCT(l1) * hCT(l2)),
    "garbage (trivial 3j, no ladder)": gamma_f(l1, l, l2) * wigner_3j(l, l1, l2, 0, 0, 0)
                                       / (hCT(l1) * hCT(l2)),
}


# --- compile each candidate: get (estimator, normalization) pair ---
print("=" * 78)
print("Expression-space search with Fisher scoring (A_L-based)")
print("=" * 78)
print()

compiled = {}
for name, g in candidates.items():
    try:
        est, A_L_fn, N_L_phi_fn = compile_qe(
            g, LMAX, Delta=2, hCxx=hCT, hCyy=hCT,
            user_funcs=[hCT, CT], px=px,
        )
        compiled[name] = (est, A_L_fn, N_L_phi_fn, None)
    except Exception as e:
        compiled[name] = (None, None, None, f"{type(e).__name__}: {str(e)[:60]}")


# --- score each: integrated 1/N_L^phi (Fisher info) over L range ---
def fisher_score(N_L_phi_fn, L_lo=40, L_hi=180):
    """Total Fisher info: sum_L (2L+1) / N_L^phi.
    This IS scale-invariant under g -> α·g and reflects actual SNR."""
    N = N_L_phi_fn(ocltt, cltt_theory)
    Ls = np.arange(L_lo, L_hi + 1)
    good = np.isfinite(N[Ls]) & (N[Ls] > 0)
    return np.sum((2 * Ls[good] + 1) / N[Ls[good]])


print(f"{'rank':>4s}  {'Fisher':>12s}  {'N_L^φ(L=50)':>14s}  {'A_L(L=50)':>14s}  candidate")
print("-" * 90)
results = []
for name, (est, A_L_fn, N_L_phi_fn, err) in compiled.items():
    if err is not None:
        results.append((name, -np.inf, np.inf, np.inf, err))
        continue
    try:
        F = fisher_score(N_L_phi_fn)
        N50 = float(N_L_phi_fn(ocltt, cltt_theory)[50])
        A50 = float(A_L_fn(ocltt, cltt_theory)[50])
        results.append((name, F, N50, A50, None))
    except Exception as e:
        results.append((name, -np.inf, np.inf, np.inf, f"score failed: {e}"))

results.sort(key=lambda t: t[1], reverse=True)

for i, (name, F, N50, A50, err) in enumerate(results):
    if err:
        print(f"{i+1:>4d}  {'--':>12s}  {'--':>14s}  {'--':>14s}  {name}")
        print(f"{'':>4s}  {'':>12s}  {'':>14s}  {'--':>14s}    ({err})")
    else:
        print(f"{i+1:>4d}  {F:>12.4e}  {N50:>14.4e}  {A50:>14.4e}  {name}")

print()
print("Interpretation: Fisher = sum (2L+1)/N_L^phi, scale-invariant under g→αg.")
print("Correct f_TT and scaled f_TT should TIE (response and noise both scale).")
print("Wrong-ladder / wrong-m / no-response have larger N_L^phi → lower Fisher.")
print("Garbage with no ladder diverges.")
