"""EB: look per-half and inspect what's drifting."""
import numpy as np
import healpy as hp

from symqe.engine.l12_sum import l1, l2, l as l_
from symqe.engine.namikawa import hCE, hCB, f_EB, W_lens_m, CB, CE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, compile_native, pol_E_pair, pol_B_pair, fuse_spin_pairs,
    compile_eb_native,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
oclee = clee + 1.5; oclbb = clbb + 1.5
spec = {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": clbb}

np.random.seed(42)
E = hp.synalm(oclee, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_EB() / (hCE(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)

print(f"EB: {len(plans)} plans, {len(terms)} terms")
for i, p in enumerate(plans):
    print(f"Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    for k, v in p.coeffs.items():
        print(f"    coeff[{k}] = {complex(v):.6f}")

# f_EB = W_lens_m(l1,l,l2) * CB(l2) + W_lens_m(l2,l,l1) * CE(l1)
# (cross-spectrum, asymmetric: CB on l2 vs CE on l1)
g1 = W_lens_m(l1, l_, l2) * CB(l2) / (hCE(l1) * hCB(l2))
g2 = W_lens_m(l2, l_, l1) * CE(l1) / (hCE(l1) * hCB(l2))
t1 = strip_gamma(compile_estimator(g1))
t2 = strip_gamma(compile_estimator(g2))

gen_full = np.asarray(compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1])
gen_h1 = np.asarray(compile_native(t1, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1])
gen_h2 = np.asarray(compile_native(t2, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1])
ref = np.asarray(compile_eb_native(terms, LMAX, px=px)(E, B, spec))

p1 = fuse_spin_pairs(t1); p2 = fuse_spin_pairs(t2)
print(f"\nhalf1 (W_lens_m(l1,l,l2)*CB(l2), unswapped) plans: {len(p1)}")
for i, p in enumerate(p1):
    print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    for k, v in p.coeffs.items():
        print(f"    coeff[{k}] = {complex(v):.6f}")
print(f"\nhalf2 (W_lens_m(l2,l,l1)*CE(l1), swapped) plans: {len(p2)}")
for i, p in enumerate(p2):
    print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    for k, v in p.coeffs.items():
        print(f"    coeff[{k}] = {complex(v):.6f}")

def per_L(label, gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n-- {label} --")
    print(f"{'L':>4s} {'gen/ref':>12s} {'signed':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            r = np.sqrt(gc[L] / rc[L])
            s = xc[L] / rc[L]
            print(f"{L:>4d} {r:>12.5e} {s:>14.5e}")

per_L("gen_full vs ref", gen_full, ref)
per_L("gen_h1 vs ref", gen_h1, ref)
per_L("gen_h2 vs ref", gen_h2, ref)
