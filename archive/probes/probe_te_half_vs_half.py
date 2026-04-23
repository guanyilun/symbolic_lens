"""Compare generic emitter pol-half vs hand-coded pol-half, and temp-half
vs hand-coded temp-half, INDEPENDENTLY. Also dump plan coefficient structure.

Goal: figure out whether each half of TE individually matches hand-coded (up
to a scalar) or whether there's a deeper structural mismatch.
"""
import numpy as np
import healpy as hp
import math

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCE, f_TE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, _extract_pol_response_from, _strip_sqrt_l_l_plus_1
from symqe.engine.estimator_native import (
    Pixelization, compile_te_native, compile_native, scalar_pair, pol_E_pair,
    fuse_spin_pairs, qe_pol_only, qe_temperature_only, _eval_atom,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clte = 0.5 * np.sqrt(cltt * clee)
ocltt = cltt + 1; oclee = clee + 1.5
spec = {"hCT": ocltt, "hCE": oclee, "CTE": clte, "CT": cltt, "CE": clee}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
E = hp.synalm(oclee, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TE(px=+1) / (hCT(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)

print("=== FusedPlans for TE ===")
for i, p in enumerate(plans):
    print(f"Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
    for k, v in p.coeffs.items():
        print(f"    coeff[{k}] = {complex(v):.6f}")
    print(f"    X_filter={p.X_filter}")
    print(f"    Y_filter={p.Y_filter}")
    print(f"    L_factor={p.L_factor}")
    print()

pol_terms = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
temp_terms = [t for t in terms if t.spin_X == 0 and abs(t.spin_Y) == 1]

# Hand-coded pol phi and temp phi SEPARATELY
pol_ref = next(t for t in pol_terms if abs(t.spin_X) == 1)
pol_X_resp = _extract_pol_response_from(pol_ref, "X")
pol_Y_resp = pol_ref.Y_filter
pol_L_atom = pol_ref.L_factor
pol_pair_coeff = complex(pol_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * 2

temp_ref = next(t for t in temp_terms if abs(t.spin_Y) == 1)
temp_Y_response = _strip_sqrt_l_l_plus_1(temp_ref.Y_filter, "l2")
temp_X_iv = temp_ref.X_filter
temp_L_atom = temp_ref.L_factor
temp_pair_coeff = complex(temp_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2

print(f"pol_ref.coeff={complex(pol_ref.coeff):.6f}")
print(f"temp_ref.coeff={complex(temp_ref.coeff):.6f}")
print(f"pol_pair_coeff={pol_pair_coeff:.6e}")
print(f"temp_pair_coeff={temp_pair_coeff:.6e}")
print(f"pol/temp pair_coeff ratio = {pol_pair_coeff/temp_pair_coeff:.6f}")

grad_L = np.sqrt(ell * (ell + 1))
p_x = _eval_atom(pol_X_resp, ell, spec).real
p_y = _eval_atom(pol_Y_resp, ell, spec).real
p_L = _eval_atom(pol_L_atom, ell, spec).real
p_L_res = np.where(grad_L > 0, p_L / grad_L, 0.0)
X_E = hp.almxfl(T, p_x); Y_E = hp.almxfl(E, p_y)
zero = np.zeros_like(T)
pol_pc = qe_pol_only(px, X_E, zero, Y_E, zero, LMAX)
pol_phi_hc = hp.almxfl((pol_pc[0] if pol_pc.ndim == 2 else pol_pc), p_L_res) * pol_pair_coeff

t_Y_resp = _eval_atom(temp_Y_response, ell, spec).real
t_X_iv = _eval_atom(temp_X_iv, ell, spec).real
t_L = _eval_atom(temp_L_atom, ell, spec).real
t_L_res = np.where(grad_L > 0, t_L / grad_L, 0.0)
E_as_X = hp.almxfl(E, t_Y_resp); T_as_Y = hp.almxfl(T, t_X_iv)
temp_pc = qe_temperature_only(px, E_as_X, T_as_Y, LMAX)
temp_phi_hc = hp.almxfl((temp_pc[0] if temp_pc.ndim == 2 else temp_pc), t_L_res) * temp_pair_coeff

# Generic emitter per half
gen_pol = np.asarray(compile_native(pol_terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])
gen_temp = np.asarray(compile_native(temp_terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])

def compare(label, gen, ref):
    ref_cl = hp.alm2cl(ref); gen_cl = hp.alm2cl(gen); x_cl = hp.alm2cl(gen, ref)
    print(f"\n--- {label} ---")
    print(f"{'L':>4s} {'ref_cl':>12s} {'gen_cl':>12s} {'sqrt(gen/ref)':>14s} {'xcorr':>10s}")
    for L in [5, 10, 20, 50, 100, 150, 199]:
        if ref_cl[L] > 0 and gen_cl[L] > 0:
            r = np.sqrt(gen_cl[L]/ref_cl[L])
            x = x_cl[L]/np.sqrt(ref_cl[L]*gen_cl[L])
            print(f"{L:>4d} {ref_cl[L]:>12.4e} {gen_cl[L]:>12.4e} {r:>14.6e} {x:>10.4f}")

compare("GEN POL vs HC POL (pol_phi only)", gen_pol, pol_phi_hc)
compare("GEN TEMP vs HC TEMP (temp_phi only)", gen_temp, temp_phi_hc)
