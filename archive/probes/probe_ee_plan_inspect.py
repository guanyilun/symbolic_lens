"""Inspect FusedPlan structure for EE: what coeffs (with sig) come out,
what does X_filter evaluate to numerically, and how does that relate to
hand-coded fl_+2 / fl_-2?"""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, f_EE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, _eval_atom
from symqe.engine.estimator_native import fuse_spin_pairs

LMAX = 200
ell = np.arange(LMAX + 1, dtype=float)
clee = 40.0 / (ell + 10) ** 2 + 1e-3
oclee = clee + 1.5
spec = {"hCE": oclee, "CE": clee}

g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
print(f"Number of raw EstimatorTerms: {len(terms)}")
plans = fuse_spin_pairs(terms)
print(f"Number of FusedPlans: {len(plans)}\n")

for i, p in enumerate(plans):
    print(f"=== Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L} ===")
    print(f"  X_filter: {p.X_filter}")
    print(f"  Y_filter: {p.Y_filter}")
    print(f"  L_factor: {p.L_factor}")
    print(f"  coeffs:")
    for sig, c in sorted(p.coeffs.items()):
        print(f"    sig={sig}: coeff={c}")
    # Evaluate the filter at L=100 for sanity
    x_num = _eval_atom(p.X_filter, ell, spec).real
    y_num = _eval_atom(p.Y_filter, ell, spec).real
    L_num = _eval_atom(p.L_factor, ell, spec).real
    print(f"  X_filter @ l=50: {x_num[50]:.6e}   @ l=100: {x_num[100]:.6e}")
    print(f"  Y_filter @ l=50: {y_num[50]:.6e}   @ l=100: {y_num[100]:.6e}")
    print(f"  L_factor @ l=50: {L_num[50]:.6e}   @ l=100: {L_num[100]:.6e}")
    print()

# Now: what does falafel produce for the two a-factor filters?
print("\n=== Falafel's _gradient_spin internals ===")
print("For spin=+2 branch (M_+3 output):")
for l in [50, 100]:
    val = np.sqrt((l - 2) * (l + 3)) if l >= 2 else 0
    print(f"  fl_+2[{l}] = +sqrt((l-2)(l+3)) = {val:.6e}   sign=-1")
print("For spin=-2 branch (M_-1 output):")
for l in [50, 100]:
    val = np.sqrt((l - 1) * (l + 2)) if l >= 1 else 0
    print(f"  fl_-2[{l}] = +sqrt((l-1)(l+2)) = {val:.6e}   sign=+1")
