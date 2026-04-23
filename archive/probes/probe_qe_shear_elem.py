"""Element-wise check: does compile_native's output for shear equal
a constant times qe_shear's shear_alm?  If yes, we're off by a constant
pair_coeff only.  If no, there's a structural mismatch."""
import numpy as np
import healpy as hp
from symqe.engine.l12_sum import l, l1, l2, wigner_3j
from symqe.engine.namikawa import gamma_f, hCT
from symqe.engine.estimator_native import Pixelization, compile_native, scalar_pair
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from sympy import sqrt as sp_sqrt

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1
spec = {"hCT": ocltt, "CT": cltt}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
fT = hp.almxfl(T, 1.0 / ocltt)
px = Pixelization(nside=NSIDE)


# --- Reference qe_shear ---
from pixell import curvedsky as cs
ells = np.arange(LMAX)
shear_filt = np.sqrt((ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0))
rmapT = px.alm2map(np.stack((T, T)), spin=0, ncomp=1, mlmax=LMAX)[0]
t_alm = cs.almxfl(fT.copy(), shear_filt)
rmap = px.alm2map_spin(np.stack([t_alm, t_alm]), 0, 2, ncomp=2, mlmax=LMAX)
prod_ref = rmap * rmapT
realsp2 = prod_ref[0]
res1 = px.map2alm_spin(realsp2, LMAX, 2, 2)
shear_alm = -2.0 * res1[0]


# --- Our compile_native ---
# qe_shear uses fT on X leg (IV-filtered) and plain T on Y leg (NOT filtered).
# To match its convention symbolically, divide by hCT(l1) only (not hCT(l2)).
g_shear = sp_sqrt((l1 - 1) * l1 * (l1 + 1) * (l1 + 2)) * gamma_f(l1, l, l2) * \
          wigner_3j(l, l1, l2, 2, -2, 0) / hCT(l1)
terms = strip_gamma(compile_estimator(g_shear))
out = compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)
gen = np.asarray(out[1])

# Element-wise ratio at individual (L, M) modes
# Since healpy alm ordering is l-major, let's extract specific modes and compare
def alm_at(alm, L, M, lmax):
    """Return the alm value at (L, M) (M >= 0) using healpy's flat indexing."""
    idx = hp.Alm.getidx(lmax, L, M)
    return alm[idx]

print(f"Element-wise comparison of gen vs shear_alm_ref at various (L, M):\n")
print(f"{'L':>4s} {'M':>4s} {'|gen|':>15s} {'|ref|':>15s} {'gen/ref':>20s}")
for L in [50, 100, 150, 199]:
    for M in [0, 30, 60, 90, min(L, 150)]:
        if M > L: continue
        g = alm_at(gen, L, M, LMAX)
        r = alm_at(shear_alm, L, M, LMAX)
        if abs(r) > 1e-15:
            ratio = g / r
            print(f"{L:>4d} {M:>4d} {abs(g):>15.4e} {abs(r):>15.4e} {ratio.real:>9.5f}{ratio.imag:>+10.5f}j")
        else:
            print(f"{L:>4d} {M:>4d} {abs(g):>15.4e} {abs(r):>15.4e} (ref is zero)")

# Is gen a pure multiple of ref?
ratios = []
for L in range(2, LMAX):
    for M in range(0, min(L + 1, 20)):
        g = alm_at(gen, L, M, LMAX)
        r = alm_at(shear_alm, L, M, LMAX)
        if abs(r) > 1e-8:
            ratios.append((L, g/r))

if ratios:
    real_parts = np.array([r.real for _, r in ratios])
    imag_parts = np.array([r.imag for _, r in ratios])
    Ls = np.array([L for L, _ in ratios])
    print(f"\nComplex ratio gen/ref statistics:")
    print(f"  real part: mean = {real_parts.mean():.5e}, std = {real_parts.std():.5e}")
    print(f"  imag part: mean = {imag_parts.mean():.5e}, std = {imag_parts.std():.5e}")
    print(f"\n  std(real) / mean(real) = {real_parts.std() / abs(real_parts.mean()):.3e}")
    # Binned by L
    print(f"\n  mean real ratio by L bin:")
    for Lo, Lhi in [(2,20), (20,50), (50,100), (100,150), (150,199)]:
        mask = (Ls >= Lo) & (Ls < Lhi)
        if mask.sum() > 0:
            print(f"    L in [{Lo:>3d}, {Lhi:>3d}): n={mask.sum()}, "
                  f"real mean = {real_parts[mask].mean():+.5e}")
