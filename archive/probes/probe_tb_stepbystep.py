"""TB stepbystep: build Plan 0+1 products in generic vs hand-coded pipeline."""
import numpy as np
import healpy as hp
from sympy import sympify
import math

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCB, f_TB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, _eval_atom, _extract_pol_response_from
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs, _alm_to_signed_pair,
    _gradient_spin, _rot2d, scalar_pair, pol_B_pair,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clte = 0.5 * np.sqrt(cltt * clee)
ocltt = cltt + 1; oclbb = clbb + 1.5
spec = {"hCT": ocltt, "hCB": oclbb, "CTE": clte, "CT": cltt, "CB": clbb}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TB(px=+1) / (hCT(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)

# Build symbolic Plans 0+1 products for sL=+1
def plan_prod(plan, spectra, X_spin_alm=0, Y_spin_alm=2):
    x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
    y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
    ladder_X = (plan.abs_spin_X != X_spin_alm)
    ladder_Y = (plan.abs_spin_Y != Y_spin_alm)
    if ladder_X:
        x_fl = x_fl.copy(); x_fl[LMAX] = 0; x_fl[0] = 0; x_fl[1] = 0
    if ladder_Y:
        y_fl = y_fl.copy(); y_fl[LMAX] = 0; y_fl[0] = 0; y_fl[1] = 0

    X_pair_raw, _ = scalar_pair(T)
    Y_pair_raw, _ = pol_B_pair(B)
    Xf = np.stack([hp.almxfl(X_pair_raw[0], x_fl), hp.almxfl(X_pair_raw[1], x_fl)])
    Yf = np.stack([hp.almxfl(Y_pair_raw[0], y_fl), hp.almxfl(Y_pair_raw[1], y_fl)])
    X_maps = _alm_to_signed_pair(px, Xf, X_spin_alm, plan.abs_spin_X, LMAX)
    Y_maps = _alm_to_signed_pair(px, Yf, Y_spin_alm, plan.abs_spin_Y, LMAX)

    flip = True
    # no NP because W^- imaginary coeffs
    total = None
    for (sX, sY, sL), c in plan.coeffs.items():
        if sL != +1:
            continue
        sX_eff = -sX if flip else sX
        sY_eff = -sY if flip else sY
        Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
        My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
        contrib = complex(c) * Mx * My
        total = contrib if total is None else total + contrib
    return total

prod_gen = plan_prod(plans[0], spec) + plan_prod(plans[1], spec)

# Hand-coded TB
from symqe.engine.estimator_native import compile_tb_native
# Just for x, y filters: use _extract_pol_response
from symqe.engine.estimator_backend import _extract_pol_response
X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")
pair_coeff = complex(coeff_ref) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2

x_fl = _eval_atom(X_resp, ell, spec).real
y_fl = _eval_atom(Y_resp, ell, spec).real
X_E = hp.almxfl(T, x_fl)
Y_B = hp.almxfl(B, y_fl)
zero = np.zeros_like(T)
# Inline qe_spin_pol_defl for TB (X_E, 0, 0, Y_B)
palms = np.stack((X_E + 0, X_E - 0))  # = (X_E, X_E)
g_p2 = _gradient_spin(px, palms, LMAX, spin=+2)  # -M_+3(X_E)
g_m2 = _gradient_spin(px, palms, LMAX, spin=-2)  # +M_-1(X_E)
ymap = _rot2d(px.alm2map(np.stack((zero, Y_B)), spin=2, ncomp=2, mlmax=LMAX))
dmap_hc = -g_m2 * ymap[0] - g_p2 * ymap[1]
prod_hc = dmap_hc / 2

print(f"prod_gen median |Re|={np.median(np.abs(prod_gen.real)):.3e}")
print(f"prod_hc  median |Re|={np.median(np.abs(prod_hc.real)):.3e}")
mask = np.abs(prod_hc) > 1e-6 * np.max(np.abs(prod_hc))
r = prod_gen[mask] / prod_hc[mask]
print(f"\nratio prod_gen/prod_hc over {mask.sum()} pixels:")
print(f"  median Re={np.median(r.real):.5e}  Im={np.median(r.imag):.5e}")
print(f"  std Re={np.std(r.real):.5e}  std Im={np.std(r.imag):.5e}")
