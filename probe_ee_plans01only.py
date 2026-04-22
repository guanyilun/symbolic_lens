"""Run compile_native EE but only using Plans 0 and 1.
If per-L ratio becomes FLAT (bit-for-bit up to constant), then Plan
2, 3 (ladder-on-Y) are the source of the 3% variation.
"""
import numpy as np
import healpy as hp
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCE, f_EE
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, compile_ee_native, compile_native, pol_E_pair,
    fuse_spin_pairs,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
clee = 40.0 / (ell + 10) ** 2 + 1e-3
oclee = clee + 1.5
spec = {"hCE": oclee, "CE": clee}

np.random.seed(42)
E = hp.synalm(oclee, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)
# Reconstruct terms for only the "ladder on X" plans (0, 1).
from sym_utils.estimator import EstimatorTerm
terms_01 = []
for p in [plans[0], plans[1]]:
    for sig, c in p.coeffs.items():
        sX, sY, sL = sig
        terms_01.append(EstimatorTerm(
            coeff=c, spin_X=sX * p.abs_spin_X, spin_Y=sY * p.abs_spin_Y,
            spin_L=sL * p.abs_spin_L,
            X_filter=p.X_filter, Y_filter=p.Y_filter, L_factor=p.L_factor))

ref = np.asarray(compile_ee_native(terms, LMAX, px=px)(E, spec))
gen01 = np.asarray(compile_native(terms_01, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])

ref_ell = hp.alm2cl(ref); gen_ell = hp.alm2cl(gen01)
print("Plans 0+1 only, ratios sqrt(gen/ref) per L:")
for L in [5, 10, 20, 50, 100, 150, 199]:
    if ref_ell[L] > 0:
        r = np.sqrt(gen_ell[L] / ref_ell[L])
        print(f"  L={L:3d}  ref={ref_ell[L]:.4e}  gen={gen_ell[L]:.4e}  ratio={r:.6e}")
