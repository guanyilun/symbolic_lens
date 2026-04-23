"""Print sig distribution for EE, BB, rot-EB plans and compare to the
hand-coded pair structure."""
from sympy import sympify
from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCE, hCB, f_EE, f_BB, f_TE, f_TB, f_EB, f_rot_EB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import fuse_spin_pairs

for name, g in [
    ("EE", f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))),
    ("BB", f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))),
    ("EB", f_EB(px=+1) / (hCE(l1) * hCB(l2))),
    ("TB", f_TB(px=+1) / (hCT(l1) * hCB(l2))),
    ("ROT-EB", f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))),
]:
    terms = strip_gamma(compile_estimator(g))
    plans = fuse_spin_pairs(terms)
    print(f"=== {name}: {len(terms)} terms → {len(plans)} plans ===")
    for i, p in enumerate(plans):
        print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
        for sig, c in sorted(p.coeffs.items()):
            print(f"    sig={sig}  c={c:+.4g}")
    print()
