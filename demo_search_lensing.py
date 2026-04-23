"""Proof-of-concept: expression-space search for a CMB lensing QE.

Given a lensed CMB map with known phi_true, enumerate candidate
symbolic W expressions, compile each through compile_native, and rank
by reconstruction quality vs phi_true.

Figure of merit: mode-count-weighted cross-correlation coefficient
  r_L = cl(psi, phi_true) / sqrt(cl(psi) · cl(phi_true))
  FOM  = sum_L (2L+1) r_L / sum_L (2L+1)
integrated over an L-range where the QE has support (40 ≤ L ≤ 280).

If the Hu-Okamoto f_TT wins the ranking, the methodology is validated.
"""
import numpy as np
import healpy as hp
from sympy import sympify, sqrt as sp_sqrt, Symbol

from pixell import enmap, curvedsky as cs, lensing, utils as u

from sym_utils.l12_sum import l, l1, l2, wigner_3j, P
from sym_utils.namikawa import (
    hCT, CT, f_TT, gamma_f, a, a_plus, a_minus, q_plus,
)
from sym_utils.estimator_native import (
    Pixelization, compile_native, scalar_pair,
)
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma


# --------------------------------------------------------------
# 1. Simulate a lensed CMB via pixell.lensing.lens_map_curved.
# --------------------------------------------------------------
LMAX = 300
ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
clphi_theory = 1e-7 / (ell + 10) ** 2   # Planck-scale-ish lensing phi
clphi_theory[:2] = 0

T_unlensed = cs.rand_alm(cltt_theory, lmax=LMAX, seed=1)
phi_true   = cs.rand_alm(clphi_theory, lmax=LMAX, seed=2)

shape, wcs = enmap.fullsky_geometry(res=15 * u.arcmin)
lensed_map = lensing.lens_map_curved(
    (1,) + tuple(shape), wcs, phi_true, T_unlensed[None, :], spin=[0]
)[0]
T_lensed = cs.map2alm(lensed_map[0], lmax=LMAX)

cl_obs = cltt_theory + 5.0  # IV filter includes white noise
spec = {"hCT": cl_obs, "CT": cltt_theory}
px = Pixelization(nside=256)


# --------------------------------------------------------------
# 2. Candidate grammar.  Each entry is a sympy expression for g(W).
# --------------------------------------------------------------
def swap_l12(expr):
    """Swap l1 <-> l2 simultaneously.  sympy's .subs sequentially
    applies the dict which collapses both vars to one; use simultaneous."""
    return expr.subs({l1: l2, l2: l1}, simultaneous=True)


def f_TT_correct(c=1):
    """The Hu-Okamoto lensing weight (same as namikawa.f_TT)."""
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


def f_TT_wrong_ladder(c=1):
    """Replace a(l_out, 0) = -sqrt(L(L+1)/2) with a_plus(l_out) =
    -sqrt((L-2)(L+3)/2) — structurally plausible but physically wrong
    (uses the polarization raising ladder on a temperature leg)."""
    W = -2 * a_plus(l) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


def f_TT_wrong_m(c=1):
    """Change 3j m-row from (1, 0, -1) to (0, 1, -1) — which leg
    gets the ladder transform."""
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 0, 1, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


def f_TT_no_response(c=1):
    """Drop the CT(l) response spectra — keep only the geometry."""
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l, l2) * \
        P * wigner_3j(l, l1, l2, 1, 0, -1)
    return W + swap_l12(W)


candidates = {
    "f_TT (Hu-Okamoto correct)"         : f_TT_correct() / (sympify(2) * hCT(l1) * hCT(l2)),
    "5·f_TT (scaled — same r)"          : 5 * f_TT_correct() / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong ladder (a(l,2))"        : f_TT_wrong_ladder() / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT wrong 3j m-row (0,1,-1)"      : f_TT_wrong_m() / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_TT with no CT response"          : f_TT_no_response() / (sympify(2) * hCT(l1) * hCT(l2)),
    "f_shear (Schaan-Ferraro — wrong obs)": sp_sqrt((l1 - 1) * l1 * (l1 + 1) * (l1 + 2)) *
                                            gamma_f(l1, l, l2) * wigner_3j(l, l1, l2, 2, -2, 0) / hCT(l1),
    "garbage (trivial 3j, no ladder)"   : gamma_f(l1, l, l2) * wigner_3j(l, l1, l2, 0, 0, 0) /
                                          (hCT(l1) * hCT(l2)),
}


# --------------------------------------------------------------
# 3. Figure of merit: mode-count-weighted <r(L)>.
# --------------------------------------------------------------
def score(psi_alm, phi_alm, L_lo=40, L_hi=280):
    """Weighted cross-correlation coefficient averaged over
    L in [L_lo, L_hi] with (2L+1) mode-count weighting."""
    cl_psi  = hp.alm2cl(psi_alm)
    cl_phi  = hp.alm2cl(phi_alm)
    cl_cross = hp.alm2cl(psi_alm, phi_alm)
    Ls = np.arange(L_lo, L_hi)
    valid = (cl_psi[Ls] > 0) & (cl_phi[Ls] > 0)
    if valid.sum() == 0:
        return 0.0, 0.0
    r_L = cl_cross[Ls][valid] / np.sqrt(cl_psi[Ls][valid] * cl_phi[Ls][valid])
    weights = (2 * Ls[valid] + 1)
    r_avg = np.sum(weights * r_L) / np.sum(weights)
    # Also return SNR-like statistic: sum_L (2L+1) r_L^2
    snr2 = np.sum(weights * r_L ** 2) / np.sum(weights)
    return r_avg, snr2


# --------------------------------------------------------------
# 4. Compile, evaluate, rank.
# --------------------------------------------------------------
print("=" * 74)
print("Expression-space search for CMB lensing estimator")
print("=" * 74)
print(f"Setup: lmax = {LMAX},  lensed T from pixell.lens_map_curved,")
print(f"       phi_true known.  FOM = (2L+1)-weighted <r(L)> over L in [40, 280].")
print()

results = []
for name, g in candidates.items():
    try:
        terms = strip_gamma(compile_estimator(g))
        if not terms:
            results.append((name, 0.0, 0.0, "empty after compile"))
            continue
        out = compile_native(terms, LMAX, px=px)(
            scalar_pair(T_lensed), scalar_pair(T_lensed), spec
        )
        # Pick output channel: for |sL|=1 lensing take +1 (phi);
        # for |sL|=0 source/mask take 0; for |sL|=2 shear take +1 too.
        if 1 in out:
            alm = np.asarray(out[1])
        elif 0 in out:
            alm = np.asarray(out[0])
        else:
            alm = np.asarray(next(iter(out.values())))
        r_avg, snr2 = score(alm, phi_true)
        results.append((name, r_avg, snr2, f"{len(terms)} term(s), keys={list(out.keys())}"))
    except Exception as e:
        results.append((name, 0.0, 0.0, f"FAILED: {type(e).__name__}: {str(e)[:40]}"))

# Rank by |r_avg|
results.sort(key=lambda t: abs(t[1]), reverse=True)

print(f"{'rank':>4s}  {'<r>':>8s}  {'<r²>':>8s}  candidate")
print("-" * 74)
for i, (name, r, s, note) in enumerate(results):
    print(f"{i+1:>4d}  {r:>+8.4f}  {s:>8.4f}  {name}")
    print(f"{'':>4s}  {'':>8s}  {'':>8s}    ({note})")
print()
winner = results[0][0]
print(f"Winner: {winner}")
print()
print("Expected: f_TT-correct and 5·f_TT should tie at the top (scalar")
print("multiples have identical r-coefficient).  Any wrong-ladder or")
print("wrong-m variant should score lower.  Garbage should score near zero.")
