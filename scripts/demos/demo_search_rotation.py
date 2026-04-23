"""Rotation-EB symbolic-search case, run through sq.run_search.

Physics: a scalar rotation field α(n̂) mixes E and B in pixel space via
(Q + iU)(n̂) → (Q + iU)(n̂) · e^{2iα(n̂)}.  With primordial C_B = 0,
only the E×B quadratic is sensitive to α; the Namikawa W^{α,+} weight
(namikawa.f_rot_EB) is the minimum-variance estimator for α.

Pipeline:
  1. Generate primordial E from C_EE (B_primordial = 0).
  2. Generate α from a scale-invariant spectrum.
  3. Rotate (Q, U) by 2α in pixel space; transform back to (E_obs, B_obs).
  4. Enumerate QE weights with E-leg / B-leg vocabulary.
  5. Fisher prune → oracle rerank against α_true.
  6. Report where f_rot_EB lands.

Note: the Namikawa f_rot_EB is the signed sum of two halves
  W·C_B(l2) − W·C_E(l1)
where the l1↔l2 swap flips sign.  Our enumerator currently emits
single-atom candidates (symmetrize_l1l2 adds only the +sign sum);
so we inject BOTH the full reference and its two halves separately
and compare.  A future enumerator extension ``sym_signs=(+1,-1)``
would capture the antisymmetric combination natively.
"""
import numpy as np
import healpy as hp
from sympy import sympify, I

from pixell import curvedsky as cs

import symqe as sq
from symqe import (
    l, l1, l2, wigner_3j, P, hCE, hCB, CE, CB,
    gamma_f, q_plus, q_minus, f_rot_EB,
)


# --- theory spectra --------------------------------------------------
LMAX, NSIDE = 100, 64
ell = np.arange(LMAX + 1, dtype=float)

clee_theory   = 40.0 / (ell + 10) ** 2 + 1e-3
clbb_theory   = np.zeros_like(clee_theory)        # primordial B = 0
clalpha_theory = 5e-5 / (ell + 10) ** 2
clalpha_theory[:2] = 0.0

oclee = clee_theory + 1.5 * np.ones_like(clee_theory)    # E + noise
oclbb = 1.5 * np.ones_like(clee_theory)                  # B is all noise + rotation


# --- simulation ------------------------------------------------------
def simulate_rotation_EB(seed: int):
    """Return (E_input, B_input, α_true_alm) for a seeded sim."""
    E_prim    = cs.rand_alm(clee_theory,    lmax=LMAX, seed=seed * 3 + 1)
    alpha_alm = cs.rand_alm(clalpha_theory, lmax=LMAX, seed=seed * 3 + 2)
    B_prim    = np.zeros_like(E_prim)

    # go to (Q, U) maps; hp.alm2map_spin on [E, B] returns [Q, U]
    Q, U = hp.alm2map_spin(
        [E_prim.astype(np.complex128), B_prim.astype(np.complex128)],
        nside=NSIDE, spin=2, lmax=LMAX,
    )

    # rotate polarization by 2α in pixel space
    alpha_map = hp.alm2map(alpha_alm.astype(np.complex128),
                           nside=NSIDE, lmax=LMAX, pol=False)
    c2 = np.cos(2 * alpha_map)
    s2 = np.sin(2 * alpha_map)
    Q_rot =  c2 * Q - s2 * U
    U_rot =  s2 * Q + c2 * U

    # back to (E_obs, B_obs) alms
    E_obs, B_obs = hp.map2alm_spin(
        [np.asarray(Q_rot, dtype=np.float64),
         np.asarray(U_rot, dtype=np.float64)],
        spin=2, lmax=LMAX,
    )

    return (sq.pol_E_pair(E_obs),
            sq.pol_B_pair(B_obs),
            alpha_alm)


# --- reference candidates -------------------------------------------
# Full Namikawa f_rot_EB with px=-1 (the sign on the swap half).
f_rot_EB_full = f_rot_EB(px=-1)

# Individual halves — what the single-atom enumerator can reach.
# W_rot_p · C_B(l2)  — the "E on l1 × C_B on l2" half.
from symqe.engine.namikawa import W_rot_p as W_rot_p_fn
f_rot_EB_halfCB = W_rot_p_fn(l1, l, l2) * CB(l2)
f_rot_EB_halfCE = W_rot_p_fn(l2, l, l1) * CE(l1)


# --- case definition ------------------------------------------------
case_rot_EB = sq.SearchCase(
    name="rotation-EB",
    lmax=LMAX,
    grammar=dict(
        m_max=2,
        coeffs=(1, 1j),                   # real/imag; other signs collapse in FOM
        parity_factors=(sympify(1), P, 1 + P, 1 - P),
        max_leg_factors=2,
        ladder_spins=(),                  # rotation base has no ladder
        spectra_X={"CE": CE, "CB": CB},   # E-leg filter vocabulary
        spectra_Y={"CE": CE, "CB": CB},   # B-leg filter vocabulary
        restrict_m_L=(0,),                # scalar output (α)
        symmetrize_l1l2=False,            # f_rot_EB is antisymmetric
    ),
    simulate=simulate_rotation_EB,
    compile_kwargs=dict(
        Delta=1, hCxx=hCE, hCyy=hCB,
        user_funcs=[hCE, hCB, CE, CB],
        nside=NSIDE,
    ),
    spectra_for_fisher=(oclee, oclbb, clee_theory, clbb_theory),
    spectra_for_estimator={
        "hCE": oclee, "hCB": oclbb,
        "CE":  clee_theory, "CB": clbb_theory,
    },
    fisher_L_range=(20, 80), fisher_L_ref=40,
    oracle_L_range=(20, 80),
    output_key=0,                         # scalar α output
    top_k_for_oracle=1000,
    references=[
        ("f_rot_EB (Namikawa full)", f_rot_EB_full),
        ("f_rot_EB half (W·CB)",     f_rot_EB_halfCB),
        ("f_rot_EB half (W·CE swap)", f_rot_EB_halfCE),
    ],
)


if __name__ == "__main__":
    result = sq.run_search(case_rot_EB, seeds=(0,), verbose=True)
    sq.print_summary(result, n_show=15)
