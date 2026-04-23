"""Full compile_native EE vs compile_ee_native, in alm space.
Check if the ratio is pure scalar (bit-for-bit up to constant) or has
structure — and where structure comes from."""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, f_EE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, compile_ee_native, compile_native, pol_E_pair
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

ref = np.asarray(compile_ee_native(terms, LMAX, px=px)(E, spec))
gen = np.asarray(compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])

# Per-ell ratio (avg over m)
ref_ell = hp.alm2cl(ref)
gen_ell = hp.alm2cl(gen)
cross_ell = hp.alm2cl(gen, ref)

print("per-ell ratio summary (cross/ref vs gen/ref):")
print(f"{'L':>4s}  {'ref':>12s}  {'gen':>12s}  {'gen/ref':>12s}  {'xcorr^2':>12s}")
for L in [5, 10, 20, 50, 100, 150, 199, 200]:
    if ref_ell[L] > 0:
        r = gen_ell[L] / ref_ell[L]
        x = cross_ell[L]**2 / (ref_ell[L] * gen_ell[L]) if gen_ell[L] > 0 else 0
        print(f"{L:>4d}  {ref_ell[L]:>12.4e}  {gen_ell[L]:>12.4e}  {np.sqrt(r):>12.6e}  {x:>12.6e}")

# Global ratio
mask = np.abs(ref) > 1e-6
r = gen[mask] / ref[mask]
print(f"\nglobal: median |r|={np.median(np.abs(r)):.6e}  "
      f"std|r|/mean|r|={np.std(np.abs(r))/np.mean(np.abs(r)):.6e}")
print(f"        median Re={np.median(r.real):.6e}  std Re={np.std(r.real):.6e}")
print(f"        median Im={np.median(r.imag):.6e}  std Im={np.std(r.imag):.6e}")
