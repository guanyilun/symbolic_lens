"""TE: check pol half and temp half individually to localize the bug."""
import numpy as np
import healpy as hp
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, f_TE
from sym_utils.estimator import compile_estimator, EstimatorTerm
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, compile_te_native, compile_native, scalar_pair, pol_E_pair,
    fuse_spin_pairs,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clte = 0.5 * np.sqrt(cltt * clee)
ocltt = cltt + 1; oclee = clee + 1.5
spec = {"hCT": ocltt, "hCE": oclee, "CTE": clte, "CT": cltt, "CE": clee}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
E = hp.synalm(oclee, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TE(px=+1) / (hCT(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)
for i, p in enumerate(plans):
    print(f"Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")

# Split terms into pol and temp groups based on which plan they come from
pol_terms = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
temp_terms = [t for t in terms if t.spin_X == 0 and abs(t.spin_Y) == 1]
print(f"\npol_terms: {len(pol_terms)}, temp_terms: {len(temp_terms)}")

ref = np.asarray(compile_te_native(terms, LMAX, px=px)(T, E, spec))
gen_full = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])
gen_pol  = np.asarray(compile_native(pol_terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])
gen_temp = np.asarray(compile_native(temp_terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])

def summary(label, gen, ref):
    mask = np.abs(ref) > 1e-6
    r = gen[mask] / ref[mask]
    print(f"  {label:20s}  median |r|={np.median(np.abs(r)):.4e}  "
          f"std|r|/mean|r|={np.std(np.abs(r))/np.mean(np.abs(r)):.4e}")

print("\nFull TE vs reference (includes both halves combined):")
summary("gen_full / ref", gen_full, ref)
print("\nPer-half breakdown — each half only vs full ref:")
summary("gen_pol_only / ref", gen_pol, ref)
summary("gen_temp_only / ref", gen_temp, ref)

# Cross-correlation diagnostics per-L
ref_ell = hp.alm2cl(ref); gen_ell = hp.alm2cl(gen_full)
cross_ell = hp.alm2cl(gen_full, ref)
print(f"\n{'L':>4s}  {'ref_cl':>12s}  {'gen_cl':>12s}  {'sqrt(gen/ref)':>14s}  {'xcorr':>10s}")
for L in [5, 10, 20, 50, 100, 150, 199]:
    if ref_ell[L] > 0 and gen_ell[L] > 0:
        r = np.sqrt(gen_ell[L] / ref_ell[L])
        x = cross_ell[L] / np.sqrt(ref_ell[L] * gen_ell[L])
        print(f"{L:>4d}  {ref_ell[L]:>12.4e}  {gen_ell[L]:>12.4e}  {r:>14.6e}  {x:>10.4f}")
