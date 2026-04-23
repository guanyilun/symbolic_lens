"""Lensing TT search via the generic run_search harness.

Same grammar / scoring as demo_search_oracle.py, but expressed as a
SearchCase so the multi-case paper runs can plug in new cases (rotation,
source, ...) by swapping the ``simulate`` callable and grammar.

Adds multi-seed averaging — single-realization oracle scores at
lmax=100 have ~4% noise on the reference ranking.  Averaging over
≥4 seeds stabilizes the top-K enough to distinguish correct from
close-wrong variants.
"""
import numpy as np
from sympy import sympify

from pixell import enmap, curvedsky as cs, lensing, utils as u

import symqe as sq
from symqe import l, l1, l2, wigner_3j, P, hCT, CT, gamma_f, a, q_plus


# --- fixed theory spectra (seed-independent) -------------------------
LMAX, NSIDE = 100, 64
ell = np.arange(LMAX + 1, dtype=float)
cltt_theory  = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt        = cltt_theory + 1.0
clphi_theory = 1e-7 / (ell + 10) ** 2
clphi_theory[:2] = 0.0


# --- simulation generator (returns X_input, Y_input, phi_true) -------
def simulate_lensed_TT(seed: int):
    T_unl    = cs.rand_alm(cltt_theory,  lmax=LMAX, seed=seed * 2 + 1)
    phi_true = cs.rand_alm(clphi_theory, lmax=LMAX, seed=seed * 2 + 2)
    shape, wcs = enmap.fullsky_geometry(res=15 * u.arcmin)
    lensed = lensing.lens_map_curved(
        (1,) + tuple(shape), wcs, phi_true, T_unl[None, :], spin=[0]
    )[0]
    T_lensed = cs.map2alm(lensed[0], lmax=LMAX)
    return (sq.scalar_pair(T_lensed),
            sq.scalar_pair(T_lensed),
            phi_true)


# --- reference candidates --------------------------------------------
def swap_l12(expr):
    return expr.subs({l1: l2, l2: l1}, simultaneous=True)

def f_TT_correct():
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_TT_wrong_ladder():
    W = -2 * a(l, 2) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)

def f_TT_wrong_m():
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 0, 1, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


# --- case definition -------------------------------------------------
case_TT = sq.SearchCase(
    name="lensing-TT",
    lmax=LMAX,
    grammar=dict(
        m_max=2,
        coeffs=(1,),
        parity_factors=(sympify(1), 1 + P),
        max_leg_factors=2,
        ladder_spins=(0, 2, -2),
        spectra_X={"CT": CT},
        spectra_Y={"CT": CT},
        restrict_m_L=(-1, 1),
        symmetrize_l1l2=True,
    ),
    simulate=simulate_lensed_TT,
    compile_kwargs=dict(
        Delta=2, hCxx=hCT, hCyy=hCT,
        user_funcs=[hCT, CT],
        px=sq.Pixelization(nside=NSIDE),
    ),
    spectra_for_fisher=(ocltt, cltt_theory),
    spectra_for_estimator={"hCT": ocltt, "CT": cltt_theory},
    fisher_L_range=(20, 80), fisher_L_ref=40,
    oracle_L_range=(20, 80),
    output_key=+1,
    top_k_for_oracle=1000,
    references=[
        ("f_TT correct",      f_TT_correct()),
        ("f_TT wrong ladder", f_TT_wrong_ladder()),
        ("f_TT wrong m-row",  f_TT_wrong_m()),
    ],
)


if __name__ == "__main__":
    result = sq.run_search(case_TT, seeds=(0, 1, 2, 3), verbose=True)
    sq.print_summary(result, n_show=15)
