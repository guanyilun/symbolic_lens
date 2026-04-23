"""Test EE plans individually to see which match hand-coded."""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, f_EE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs, compile_ee_native, compile_native,
    pol_E_pair,
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

ref = compile_ee_native(terms, LMAX, px=px)(E, spec)
mask = np.abs(ref) > 1e-6 * np.max(np.abs(ref))

# Test each plan's terms only
for i, plan in enumerate(plans):
    plan_terms = [t for t in terms
                  if (abs(t.spin_X) == plan.abs_spin_X
                      and abs(t.spin_Y) == plan.abs_spin_Y
                      and abs(t.spin_L) == plan.abs_spin_L)]
    emit = compile_native(plan_terms, LMAX, px=px)
    out = emit(pol_E_pair(E), pol_E_pair(E), spec)
    gen = out.get(+1, np.zeros_like(ref))
    r = np.abs(gen[mask] / ref[mask])
    print(f"Plan {i} (|sX|={plan.abs_spin_X} |sY|={plan.abs_spin_Y} |sL|={plan.abs_spin_L}): "
          f"median={np.median(r):.4f}  std/mean={np.std(r)/np.mean(r):.3e}")
