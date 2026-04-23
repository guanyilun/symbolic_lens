"""Diagnose why f_TT xcorr with phi_true is near zero in the demo."""
import numpy as np
import healpy as hp
from sympy import sympify

from pixell import enmap, curvedsky as cs, lensing, utils as u

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, f_TT
from symqe.engine.estimator_native import Pixelization, compile_native, scalar_pair
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma

LMAX = 300
ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
# More realistic phi spectrum — use l*(l+1) for a reasonable QE signal
# clphi ~ 1e-8 at peak L~50 (roughly matches Planck scale)
clphi_theory = 1e-7 / (ell + 10) ** 2
clphi_theory[:2] = 0

np.random.seed(0)
T_unlensed = cs.rand_alm(cltt_theory, lmax=LMAX, seed=1)
phi_true   = cs.rand_alm(clphi_theory, lmax=LMAX, seed=2)

# --- Check: signal magnitudes ---
print("=== Input magnitudes ===")
print(f"  T_unlensed rms = {np.sqrt(np.mean(np.abs(T_unlensed)**2)):.4e}")
print(f"  phi_true rms   = {np.sqrt(np.mean(np.abs(phi_true)**2)):.4e}")

# --- Lens ---
shape, wcs = enmap.fullsky_geometry(res=15 * u.arcmin)
lensed_map = lensing.lens_map_curved(
    (1,) + tuple(shape), wcs, phi_true, T_unlensed[None, :], spin=[0]
)[0]
T_lensed = cs.map2alm(lensed_map[0], lmax=LMAX)

# --- Check: did lensing change T? ---
print(f"\n=== Lensing sanity ===")
print(f"  T_lensed rms   = {np.sqrt(np.mean(np.abs(T_lensed)**2)):.4e}")
delta_T = T_lensed - T_unlensed
print(f"  delta T rms    = {np.sqrt(np.mean(np.abs(delta_T)**2)):.4e}")
print(f"  delta/T ratio  = {np.sqrt(np.mean(np.abs(delta_T)**2)) / np.sqrt(np.mean(np.abs(T_unlensed)**2)):.4e}")

# --- Run f_TT on lensed T ---
cl_obs = cltt_theory + 5.0
spec = {"hCT": cl_obs, "CT": cltt_theory}
px = Pixelization(nside=256)

g_TT = f_TT() / (sympify(2) * hCT(l1) * hCT(l2))
terms = strip_gamma(compile_estimator(g_TT))
out = compile_native(terms, LMAX, px=px)(scalar_pair(T_lensed), scalar_pair(T_lensed), spec)
psi = np.asarray(out[1])

print(f"\n=== f_TT QE output ===")
print(f"  psi (QE out) rms = {np.sqrt(np.mean(np.abs(psi)**2)):.4e}")

# --- Proper per-L cross-correlation ---
cl_psi = hp.alm2cl(psi)
cl_phi = hp.alm2cl(phi_true)
cl_cross = hp.alm2cl(psi, phi_true)

print(f"\n=== Per-L diagnostics ===")
print(f"{'L':>4s} {'cl_psi':>12s} {'cl_phi':>12s} {'cl_cross':>12s} {'r(L)':>10s}")
for L in [2, 5, 10, 20, 50, 100, 200, 299]:
    r = cl_cross[L] / np.sqrt(cl_psi[L] * cl_phi[L]) if cl_psi[L] > 0 and cl_phi[L] > 0 else 0
    print(f"{L:>4d} {cl_psi[L]:>12.3e} {cl_phi[L]:>12.3e} {cl_cross[L]:>12.3e} {r:>10.4f}")

# --- Summary xcorr (proper integration) ---
# Fisher-like: sum_L (cross)^2 / (psi·phi) weighted by (2L+1)
Ls = np.arange(2, LMAX)
valid = (cl_psi[Ls] > 0) & (cl_phi[Ls] > 0)
r_L = cl_cross[Ls][valid] / np.sqrt(cl_psi[Ls][valid] * cl_phi[Ls][valid])
print(f"\n=== Summary ===")
print(f"  mean r(L) (L=2..299) = {r_L.mean():.4f}")
print(f"  min / max r(L)       = {r_L.min():.4f} / {r_L.max():.4f}")
