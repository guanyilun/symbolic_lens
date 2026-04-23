"""EE: compare Plan 0+1 (unswapped, ladder-on-X) vs Plan 2+3 (swapped,
ladder-on-Y), each against half of the hand-coded result, to isolate
whether the residual is structural (sign-flip missing on swapped half)
or per-L (ladder-factor mismatch).
"""
import numpy as np
import healpy as hp

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, f_EE, W_lens_p, CE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, compile_native, pol_E_pair, fuse_spin_pairs,
    compile_ee_native,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
clee = 40.0 / (ell + 10) ** 2 + 1e-3
oclee = clee + 1.5
spec = {"hCE": oclee, "CE": clee}

np.random.seed(42)
E = hp.synalm(oclee, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

# Full EE
g = f_EE() / (hCE(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)

print(f"EE: {len(plans)} plans, {len(terms)} terms")
for i, p in enumerate(plans):
    print(f"Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    for k, v in p.coeffs.items():
        print(f"    coeff[{k}] = {complex(v):.6f}")

# Build half1 (W_lens_p(l1, l, l2) · CE(l2)) and half2 (W_lens_p(l2, l, l1) · CE(l1)) separately
h1 = W_lens_p(l1, l2=l2, l_out=l1.__class__('l')) if False else None
# Use direct symbolic construction
from symqe.engine.l12_sum import l as l_
g1 = W_lens_p(l1, l_, l2) * CE(l2) / (hCE(l1) * hCE(l2))
g2 = W_lens_p(l2, l_, l1) * CE(l1) / (hCE(l1) * hCE(l2))
t1 = strip_gamma(compile_estimator(g1))
t2 = strip_gamma(compile_estimator(g2))

gen_h1 = np.asarray(compile_native(t1, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])
gen_h2 = np.asarray(compile_native(t2, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])
gen_full = np.asarray(compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])
ref = np.asarray(compile_ee_native(terms, LMAX, px=px)(E, spec))

p1 = fuse_spin_pairs(t1); p2 = fuse_spin_pairs(t2)
print(f"\nhalf1 plans: {len(p1)}")
for i, p in enumerate(p1):
    print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
print(f"half2 plans: {len(p2)}")
for i, p in enumerate(p2):
    print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")

def per_L(label, gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n-- {label} --")
    print(f"{'L':>4s} {'gen/ref':>12s} {'signed_ratio':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            r = np.sqrt(gc[L] / rc[L])
            s = xc[L] / rc[L]
            print(f"{L:>4d} {r:>12.5e} {s:>14.5e}")

per_L("gen_full vs ref", gen_full, ref)
per_L("gen_h1 vs ref", gen_h1, ref)
per_L("gen_h2 vs ref", gen_h2, ref)
per_L("gen_h1 vs gen_h2 (internal check)", gen_h1, gen_h2)
