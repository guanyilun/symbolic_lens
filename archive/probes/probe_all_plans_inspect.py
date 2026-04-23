"""Dump plan coeffs for all 6 lensing estimators — reveal how ±i
and ± real coeffs distribute across plans."""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import (
    hCT, hCE, hCB, f_TT, f_EE, f_BB, f_TB, f_EB, f_TE
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import fuse_spin_pairs

cases = [
    ("TT", f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))),
    ("EE", f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))),
    ("BB", f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))),
    ("TB", f_TB(px=+1) / (hCT(l1) * hCB(l2))),
    ("EB", f_EB(px=+1) / (hCE(l1) * hCB(l2))),
    ("TE", f_TE(px=+1) / (hCT(l1) * hCE(l2))),
]

for name, g in cases:
    terms = strip_gamma(compile_estimator(g))
    plans = fuse_spin_pairs(terms)
    print(f"=== {name}: {len(plans)} plans ===")
    for i, p in enumerate(plans):
        coeff_summary = ", ".join(
            f"{sig}→{c:+.4g}"
            for sig, c in sorted(p.coeffs.items())
        )
        print(f"  Plan{i} |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}  [{coeff_summary}]")
    print()
