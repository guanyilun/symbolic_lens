"""Inspect TB plans to understand its ladder structure."""
import numpy as np
from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCB, f_TB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import fuse_spin_pairs

g = f_TB() / (hCT(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)

print(f"TB: {len(plans)} plans, {len(terms)} terms")
for i, p in enumerate(plans):
    print(f"\nPlan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    print(f"  X_filter: {p.X_filter}")
    print(f"  Y_filter: {p.Y_filter}")
    print(f"  L_factor: {p.L_factor}")
    for k, v in p.coeffs.items():
        print(f"  coeff[{k}] = {complex(v):.6f}")
