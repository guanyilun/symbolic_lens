"""TB detailed comparison — per-ell ratio, plan-by-plan."""
import numpy as np
import healpy as hp
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCB, f_TB
from sym_utils.estimator import compile_estimator, EstimatorTerm
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, compile_tb_native, compile_native, scalar_pair, pol_B_pair,
    fuse_spin_pairs,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clte = 0.5 * np.sqrt(cltt * clee)
ocltt = cltt + 1; oclbb = clbb + 1.5
spec = {"hCT": ocltt, "hCB": oclbb, "CTE": clte, "CT": cltt, "CB": clbb}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TB(px=+1) / (hCT(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)
print(f"TB plans: {len(plans)}")

ref = np.asarray(compile_tb_native(terms, LMAX, px=px)(T, B, spec))
gen = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), pol_B_pair(B), spec)[+1])

ref_ell = hp.alm2cl(ref); gen_ell = hp.alm2cl(gen)
cross_ell = hp.alm2cl(gen, ref)
print(f"\n{'L':>4s}  {'ref':>12s}  {'gen':>12s}  {'sqrt(gen/ref)':>14s}  {'xcorr':>10s}")
for L in [5, 10, 20, 50, 100, 150, 199]:
    if ref_ell[L] > 0:
        r = np.sqrt(gen_ell[L] / ref_ell[L])
        x = cross_ell[L] / np.sqrt(ref_ell[L] * gen_ell[L]) if gen_ell[L] > 0 else 0
        print(f"{L:>4d}  {ref_ell[L]:>12.4e}  {gen_ell[L]:>12.4e}  {r:>14.6e}  {x:>10.4f}")
