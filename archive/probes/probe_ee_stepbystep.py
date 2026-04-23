"""Step-by-step EE: build the generic's Plans 0+1 prod (summed),
build hand-coded's dmap via _qe_spin_pol_defl, and compare."""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, f_EE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs, _alm_to_signed_pair, _eval_atom,
    _gradient_spin, _qe_spin_pol_defl, compile_ee_native, _rot2d,
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
# Plans 0 (|sX|=1, |sY|=2) and 1 (|sX|=3, |sY|=2)
plan0 = plans[0]
plan1 = plans[1]

# === Generic: build Plan 0 + Plan 1 prods (contributing to phi, sL=+1) ===
grad_l = np.sqrt(ell * (ell + 1))

def plan_prod(plan, spectra, X_spin_alm=2, Y_spin_alm=2):
    x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
    y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
    # apply mlmax truncation to match hand-coded's gradient_spin
    # ONLY on ladder legs (where abs_spin_out != spin_alm)
    ladder_X = (plan.abs_spin_X != X_spin_alm)
    ladder_Y = (plan.abs_spin_Y != Y_spin_alm)
    if ladder_X:
        x_fl = x_fl.copy(); x_fl[LMAX] = 0; x_fl[0] = 0; x_fl[1] = 0
    if ladder_Y:
        y_fl = y_fl.copy(); y_fl[LMAX] = 0; y_fl[0] = 0; y_fl[1] = 0

    Xf = np.stack([hp.almxfl(E, x_fl), hp.almxfl(E, x_fl)])
    Yf = np.stack([hp.almxfl(E, y_fl), hp.almxfl(E, y_fl)])
    X_maps = _alm_to_signed_pair(px, Xf, spin_alm_in=2,
                                 abs_spin_out=plan.abs_spin_X, lmax=LMAX)
    Y_maps = _alm_to_signed_pair(px, Yf, spin_alm_in=2,
                                 abs_spin_out=plan.abs_spin_Y, lmax=LMAX)

    # Apply flip + NP sign (same rule as compile_native)
    flip = True
    np_sign = 1
    if plan.abs_spin_X == 3: np_sign = -np_sign
    if plan.abs_spin_Y == 3: np_sign = -np_sign

    total = None
    for (sX, sY, sL), c in plan.coeffs.items():
        if sL != +1:
            continue
        sX_eff = -sX if flip else sX
        sY_eff = -sY if flip else sY
        Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
        My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
        contrib = np_sign * complex(c) * Mx * My
        total = contrib if total is None else total + contrib
    return total

prod_gen = plan_prod(plan0, spec) + plan_prod(plan1, spec)

# === Hand-coded: via compile_ee_native internals ===
from symqe.engine.estimator_backend import _extract_pol_response
X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")
import math
pair_coeff = 2 * complex(coeff_ref) * (-1) * 2 * math.sqrt(4*math.pi) * 2

x = _eval_atom(X_resp, ell, spec).real
y = _eval_atom(Y_resp, ell, spec).real
L = _eval_atom(L_atom, ell, spec).real
L_res = np.where(grad_l > 0, L / grad_l, 0.0)
X_resp_alm = hp.almxfl(E, x)
Y_resp_alm = hp.almxfl(E, y)
zero = np.zeros_like(E)
# _qe_spin_pol_defl inlined:
palms = np.stack((X_resp_alm + 1j*zero, X_resp_alm - 1j*zero))  # X_B=0
g_p2 = _gradient_spin(px, palms, LMAX, spin=+2)
g_m2 = _gradient_spin(px, palms, LMAX, spin=-2)
ymap = _rot2d(px.alm2map(np.stack((Y_resp_alm, zero)), spin=2, ncomp=2, mlmax=LMAX))
dmap_hc = -g_m2 * ymap[0] - g_p2 * ymap[1]
prod_hc = dmap_hc / 2

# === Compare ===
print(f"prod_gen stats: |Re| median={np.median(np.abs(prod_gen.real)):.3e}  "
      f"|Im| median={np.median(np.abs(prod_gen.imag)):.3e}")
print(f"prod_hc  stats: |Re| median={np.median(np.abs(prod_hc.real)):.3e}  "
      f"|Im| median={np.median(np.abs(prod_hc.imag)):.3e}")

# Per-pixel ratio
mask = np.abs(prod_hc) > 1e-6 * np.max(np.abs(prod_hc))
r = prod_gen[mask] / prod_hc[mask]
print(f"\nprod_gen / prod_hc ratio over {mask.sum()} pixels:")
print(f"  median Re={np.median(r.real):.5e}  Im={np.median(r.imag):.5e}")
print(f"  std Re={np.std(r.real):.5e}  std Im={np.std(r.imag):.5e}")
print(f"\n(if match up to constant: std Re/Im ≈ 0, median gives the constant)")

# ratio_conj: maybe prod_gen = conj(const · prod_hc)?
r_conj = np.conj(prod_gen)[mask] / prod_hc[mask]
print(f"\nconj(prod_gen) / prod_hc:")
print(f"  median Re={np.median(r_conj.real):.5e}  Im={np.median(r_conj.imag):.5e}")
print(f"  std Re={np.std(r_conj.real):.5e}  std Im={np.std(r_conj.imag):.5e}")
