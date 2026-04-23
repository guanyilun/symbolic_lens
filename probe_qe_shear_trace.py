"""Trace the shear output step by step to find where it diverges."""
import numpy as np
import healpy as hp

from sym_utils.l12_sum import l, l1, l2, wigner_3j
from sym_utils.namikawa import gamma_f, hCT
from sym_utils.estimator_native import Pixelization
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
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

# Manual symbolic f_shear computation, bypassing compile_native.
# Apply the SAME filters and SHT sequence as qe_shear, but using the
# coefficient / spin-signature that compile_estimator would produce.
from pixell import curvedsky as cs

ells = np.arange(LMAX)
shear_filt = np.sqrt((ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0))

# qe_shear reference
rmapT_ref = px.alm2map(np.stack((T, T)), spin=0, ncomp=1, mlmax=LMAX)[0]
t_alm_ref = cs.almxfl(fT.copy(), shear_filt)
rmap_ref = px.alm2map_spin(np.stack([t_alm_ref, t_alm_ref]), 0, 2, ncomp=2, mlmax=LMAX)
prod_ref = rmap_ref * rmapT_ref
realsp2 = prod_ref[0]  # M+2 · T
realsm2 = prod_ref[1]  # M-2 · T
res1 = px.map2alm_spin(realsp2, LMAX, 2, 2)
shear_alm_ref = -2.0 * res1[0]
print(f"Reference qe_shear: rms = {np.sqrt(np.mean(np.abs(shear_alm_ref)**2)):.4e}")

# Now mirror our compile_native step by step.
# Plan: |sX|=2, |sY|=0, |sL|=2, coeff[(-1, 0, 1)] = 0.2820948
# X input: T (spin_alm=0), Y input: T (spin_alm=0)
# X_filter = sqrt((l-1)·l·(l+1)·(l+2)) / hCT(l)
# Y_filter = 1 / hCT(l)

x_fl = shear_filt.copy() / ocltt[:LMAX]  # X_filter, truncated to length LMAX via filter
x_fl_full = np.zeros(LMAX + 1)
x_fl_full[:LMAX] = x_fl
x_fl_full[0] = 0
x_fl_full[1] = 0
x_fl_full[LMAX] = 0

y_fl_full = np.zeros(LMAX + 1)
y_fl_full[:] = 1.0 / ocltt

# hp.almxfl
Xf_pair = np.stack([hp.almxfl(T.copy(), x_fl_full), hp.almxfl(T.copy(), x_fl_full)])
Yf_pair = np.stack([hp.almxfl(T.copy(), y_fl_full), hp.almxfl(T.copy(), y_fl_full)])

# X: ladder, alm2map_spin(0 → 2)
X_maps = px.alm2map_spin(Xf_pair, spin_alm=0, spin_transform=2, ncomp=2, mlmax=LMAX)
# X_maps[0] = M+2 · (filtered_T), X_maps[1] = M-2 · (filtered_T)

# Y: no ladder, alm2map scalar
Y_map = px.alm2map(Yf_pair[0], spin=0, ncomp=1, mlmax=LMAX)[0]
Y_maps = (Y_map, Y_map)

# For sig sX=-1, sY=0, sL=+1 AND flip_X=True (|sL|>0 and |sX|>0):
#   sX_eff = +1, so Mx = X_maps[0] = M+2
#   My = Y_maps[0] = T_map
flip_X = True
sX, sY, sL = -1, 0, 1
sX_eff = -sX if flip_X else sX  # +1
Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]  # M+2
My = Y_maps[0]
coeff = 0.2820948
plan_factor = 1.0
prod = plan_factor * coeff * Mx * My

# Output: map2alm_spin(prod, spin_alm=0, spin_transform=2), pick sL=+1 → alm_pair[0]
alm_pair_gen = px.map2alm_spin(prod, LMAX, 0, 2)
pick_gen = alm_pair_gen[0]  # this is our compile_native output[+1]

print(f"\nOur compile_native mirror (pick_gen): rms = {np.sqrt(np.mean(np.abs(pick_gen)**2)):.4e}")

# Compare: ratio at each L
def per_L(label, gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n-- {label} --")
    print(f"{'L':>4s} {'signed_slope':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            print(f"{L:>4d} {xc[L] / rc[L]:>14.5e}")

per_L("pick_gen vs shear_alm_ref", pick_gen, shear_alm_ref)

# Now try WITHOUT the flip_X — what if we remove the flip?
sX_eff_nf = sX  # -1 (no flip)
Mx_nf = X_maps[0] if sX_eff_nf >= 0 else X_maps[1]  # M-2
prod_nf = plan_factor * coeff * Mx_nf * My
alm_pair_nf = px.map2alm_spin(prod_nf, LMAX, 0, 2)
pick_nf = alm_pair_nf[0]
per_L("pick_gen (NO flip_X) vs shear_alm_ref", pick_nf, shear_alm_ref)

# And with spin_alm=2 (qe_shear convention)
alm_pair_shear_spinalm = px.map2alm_spin(prod, LMAX, 2, 2)
pick_shear_spinalm = alm_pair_shear_spinalm[0]
per_L("pick_gen (spin_alm=2 for extract) vs shear_alm_ref", pick_shear_spinalm, shear_alm_ref)

# What if we DON'T filter with the shear ladder in X_filter and instead leave it in L_factor?
# Or: what if we pick alm_pair[1] (i.e., sL=-1 side)?
per_L("pick_gen[sL=-1]", alm_pair_gen[1], shear_alm_ref)

# Try coming from the OTHER ladder direction: alm2map_spin(0 → -2) via sign
# Actually hp doesn't support negative spin_transform directly.

# What if we multiply our output by sqrt((L-1)·L·(L+1)·(L+2)) / sqrt(L(L+1))?
# A candidate L_factor we might be missing.
L_fl_extra = np.sqrt((ell - 1.0) * ell * (ell + 1.0) * (ell + 2.0))
L_fl_extra = np.where(np.isfinite(L_fl_extra) & (ell >= 2), L_fl_extra, 0.0)
pick_gen_wL = hp.almxfl(pick_gen, 1.0 / L_fl_extra)  # maybe we're OVER-filtering?
per_L("pick_gen / shear_L_ladder (maybe compile_native over-filters?)", pick_gen_wL, shear_alm_ref)
