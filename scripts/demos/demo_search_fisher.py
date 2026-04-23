"""Expression-space search for CMB lensing QE — Fisher-scored version,
with a side-by-side comparison of raw Fisher (gauge-pathological) vs
gauge-fixed Fisher.

The raw Fisher sum ``Σ (2L+1)/N_L^phi`` is gauge-dependent: under
``g → α·g`` the derived f scales as α too, N_L^phi ∝ 1/α², and the
raw sum ∝ α².  ``sq.fisher_fom`` multiplies by ``A_L(L_ref)`` to cancel
the α² — so the rank is invariant under a pure rescaling.
"""
import numpy as np
import healpy as hp
from sympy import sympify, sqrt as sp_sqrt

import symqe as sq
from symqe import l, l1, l2, wigner_3j, P, hCT, CT, gamma_f, a, a_plus, a_minus, q_plus
from symqe import Pixelization, scalar_pair, compile_qe, fisher_fom

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


L_LO, L_HI, L_REF = 40, 180, 80


def raw_fisher(N_L_phi):
    Ls = np.arange(L_LO, L_HI + 1)
    good = np.isfinite(N_L_phi[Ls]) & (N_L_phi[Ls] > 0)
    return float(np.sum((2 * Ls[good] + 1) / N_L_phi[Ls[good]]))


rows = []
for name, (est, A_L_fn, N_L_phi_fn, err) in compiled.items():
    if err is not None:
        rows.append((name, -np.inf, -np.inf, np.inf, np.inf, err))
        continue
    try:
        N = N_L_phi_fn(ocltt, cltt_theory)
        A = A_L_fn(ocltt, cltt_theory)
        F_raw   = raw_fisher(N)
        F_gauge = fisher_fom(N, A, L_range=range(L_LO, L_HI + 1), L_ref=L_REF)
        rows.append((name, F_raw, F_gauge, float(N[50]), float(A[50]), None))
    except Exception as e:
        rows.append((name, -np.inf, -np.inf, np.inf, np.inf, f"score failed: {e}"))


def print_ranked(rows, key_index, label):
    order = sorted(rows, key=lambda r: r[key_index], reverse=True)
    print(label)
    print(f"{'rank':>4s}  {'score':>12s}  {'N_L^φ(50)':>12s}  {'A_L(50)':>12s}  candidate")
    print("-" * 88)
    for i, (name, F_raw, F_gauge, N50, A50, err) in enumerate(order):
        if err:
            print(f"{i+1:>4d}  {'--':>12s}  {'--':>12s}  {'--':>12s}  {name}  ({err})")
        else:
            score = (F_raw if key_index == 1 else F_gauge)
            print(f"{i+1:>4d}  {score:>12.4e}  {N50:>12.4e}  {A50:>12.4e}  {name}")
    print()


print()
print_ranked(rows, 1, "=== Raw Fisher = Σ (2L+1)/N_L^phi  (gauge-DEPENDENT) ===")
print_ranked(rows, 2, f"=== Gauge-fixed Fisher = A_L({L_REF})·Σ (2L+1)/N_L^phi ===")

print("Interpretation:")
print("  Raw Fisher:   5·f_TT scores 25× f_TT — pure gauge artifact.")
print("  Gauge-fixed:  5·f_TT and f_TT TIE — the artifact is killed.")
print()
print("Residual caveat: candidates with structurally-different f (e.g. 'no CT")
print("response') can still rank differently from Hu-Okamoto-correct f_TT,")
print("because each candidate's N_L^phi is minimum-variance for its OWN")
print("implied target f.  Fisher alone cannot distinguish 'good reconstruction")
print("of the wrong thing' from 'good reconstruction of phi' — that requires")
print("either oracle cross-correlation or compile_qe_from_f with a fixed target.")
