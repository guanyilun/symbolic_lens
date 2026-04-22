"""Same as probe_ee_plan_inspect but for EB — W_lens_m has ζ_-=i so
the plan coeffs are imaginary, which interacts with sign rules
differently."""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCE, hCB, f_EB
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, _eval_atom
from sym_utils.estimator_native import fuse_spin_pairs

LMAX = 200
ell = np.arange(LMAX + 1, dtype=float)
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
oclee = clee + 1.5; oclbb = clbb + 1.5
spec = {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": clbb}

g = f_EB(px=+1) / (hCE(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
print(f"Raw EstimatorTerms: {len(terms)}\n")
plans = fuse_spin_pairs(terms)
for i, p in enumerate(plans):
    print(f"=== Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L} ===")
    print(f"  X_filter (short): {str(p.X_filter)[:120]}...")
    print(f"  coeffs:")
    for sig, c in sorted(p.coeffs.items()):
        print(f"    sig={sig}: coeff={c}")
    print()
