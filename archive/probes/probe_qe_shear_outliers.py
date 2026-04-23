"""Find where gen/ref deviates from the constant pair_coeff."""
import numpy as np
import healpy as hp
from sympy import sqrt as sp_sqrt

from symqe.engine.l12_sum import l, l1, l2, wigner_3j
from symqe.engine.namikawa import gamma_f, hCT
from symqe.engine.estimator_native import (
    Pixelization, compile_native, scalar_pair,
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma

LMAX, NSIDE = 200, 256

ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1
spec = {"hCT": ocltt, "CT": cltt}
np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
fT = hp.almxfl(T, 1.0 / ocltt)
px = Pixelization(nside=NSIDE)

# Reference
from pixell import curvedsky as cs
ells = np.arange(LMAX)
filt = np.sqrt((ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0))
rmapT = px.alm2map(np.stack((T, T)), spin=0, ncomp=1, mlmax=LMAX)[0]
t_alm = cs.almxfl(fT.copy(), filt)
rmap = px.alm2map_spin(np.stack([t_alm, t_alm]), 0, 2, ncomp=2, mlmax=LMAX)
prod = rmap * rmapT
res1 = px.map2alm_spin(prod[0], LMAX, 2, 2)
ref = -2.0 * res1[0]

# Symbolic
g_shear = sp_sqrt((l1 - 1) * l1 * (l1 + 1) * (l1 + 2)) * \
          gamma_f(l1, l, l2) * wigner_3j(l, l1, l2, 2, -2, 0) / hCT(l1)
terms = strip_gamma(compile_estimator(g_shear))
gen = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)[1])

# Walk all (L, M) and find outliers
expected = -0.141047
for idx in range(len(gen)):
    L, M = hp.Alm.getlm(LMAX, idx)
    if abs(ref[idx]) < 1e-8:
        continue
    r = gen[idx] / ref[idx]
    # Report outliers
    if abs(r.real - expected) > 1e-4 or abs(r.imag) > 1e-4:
        print(f"  (L={L:>3d}, M={M:>3d}):  gen/ref = {r.real:+.6f}{r.imag:+.6f}j  "
              f"|ref|={abs(ref[idx]):.3e}  |gen|={abs(gen[idx]):.3e}")
        # Only print first 20 outliers
