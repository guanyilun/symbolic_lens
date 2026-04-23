"""Compute Plan 0 output step-by-step alongside hand-coded, comparing
at every stage."""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, f_TT
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs, _alm_to_signed_pair, _eval_atom,
    _gradient_spin, _qe_spin_temp_defl, _deflection_to_phi_curl,
    compile_tt_native,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1.0
spectra = {"hCT": ocltt, "CT": cltt}
np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)
plan = plans[0]

# --- hand-coded reconstruction ---
x_fl_raw = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
L_fl_raw = _eval_atom(plan.L_factor, ell, spectra).real.astype(np.float64)
grad_l = np.sqrt(ell * (ell + 1))
x_response_fl = np.where(grad_l > 0, x_fl_raw / grad_l, 0.0)
L_residual_fl = np.where(grad_l > 0, L_fl_raw / grad_l, 0.0)

X_resp = hp.almxfl(T, x_response_fl)
Y_iv = hp.almxfl(T, y_fl)
grad_hc = _gradient_spin(px, np.stack((X_resp, X_resp)), LMAX, spin=0)
ymap_hc = px.alm2map(Y_iv, spin=0, ncomp=1, mlmax=LMAX)[0]
dmap_hc = -grad_hc * ymap_hc
phicurl_hc = _deflection_to_phi_curl(px, dmap_hc, LMAX)
phi_hc = phicurl_hc[0]
phi_hc_postL = hp.almxfl(phi_hc, L_residual_fl)  # pair_coeff=1 for TT

# --- generic Plan 0 with mlmax truncation ---
x_fl = x_fl_raw.copy()
x_fl[0] = 0; x_fl[1] = 0; x_fl[LMAX] = 0  # match hand-coded
L_fl = L_fl_raw.copy()
L_fl[LMAX] = 0  # match hand-coded

T_pair = np.stack([T, T])
Xf = np.stack([hp.almxfl(T_pair[0], x_fl), hp.almxfl(T_pair[1], x_fl)])
Yf = np.stack([hp.almxfl(T_pair[0], y_fl), hp.almxfl(T_pair[1], y_fl)])
X_maps = _alm_to_signed_pair(px, Xf, 0, plan.abs_spin_X, LMAX)
Y_maps = _alm_to_signed_pair(px, Yf, 0, plan.abs_spin_Y, LMAX)
# sig (-1,0,+1) with flip_X=True → Mx = X_maps[0] = M_+1
Mx = X_maps[0]
My = Y_maps[0]
c = [cc for sig, cc in plan.coeffs.items() if sig == (-1, 0, 1)][0]
prod_gen = c * Mx * My
alm_pair_gen = px.map2alm_spin(prod_gen, lmax=LMAX, spin_alm=0, spin_transform=plan.abs_spin_L)
phi_gen_preL = alm_pair_gen[0]
phi_gen = hp.almxfl(phi_gen_preL, L_fl)

# --- checks at every stage ---
print("=== stepwise comparison, Plan 0 (sig (-1,0,+1) under flip_X=True) only ===")
print(f"X maps: max|Mx - grad_hc|              = {np.max(np.abs(Mx - grad_hc)):.3e}")
print(f"Y maps: max|My - ymap_hc|              = {np.max(np.abs(My - ymap_hc)):.3e}")
# prod_gen = c·M_+1·Y,  dmap_hc = -M_+1·Y  →  prod_gen = -c·dmap_hc
diff_prod = prod_gen - (-c) * dmap_hc
print(f"prod: max|prod_gen - (-c)·dmap_hc|     = {np.max(np.abs(diff_prod)):.3e}")
# phi_gen_preL = [map2alm_spin(prod_gen)][0].  Hand-coded raw =
# [map2alm_spin(dmap_hc)][0], scaled by sqrt(L(L+1)) in _deflection_to_phi_curl.
raw_hc = px.map2alm_spin(dmap_hc, lmax=LMAX, spin_alm=0, spin_transform=1)[0]
diff_raw = phi_gen_preL - (-c) * raw_hc
print(f"raw map2alm_spin[0]: max|ratio diff|   = {np.max(np.abs(diff_raw)):.3e}")
# Now phi_hc = raw_hc · sqrt(L(L+1))_trunc   (cs.almxfl truncates L=LMAX)
# phi_gen = phi_gen_preL · L_fl  (L_fl = sqrt(L(L+1)) with LMAX zeroed)
# So (phi_gen/(-c))  should equal phi_hc.
diff_phi = phi_gen / (-c) - phi_hc
print(f"final phi: max|phi_gen/(-c) - phi_hc|  = {np.max(np.abs(diff_phi)):.3e}")
print(f"   max|phi_hc|                         = {np.max(np.abs(phi_hc)):.3e}")
print(f"   ratio                                = {np.max(np.abs(diff_phi))/max(np.max(np.abs(phi_hc)),1e-30):.3e}")
