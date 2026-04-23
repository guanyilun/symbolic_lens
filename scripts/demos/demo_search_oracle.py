"""Two-pass search: Fisher prune + oracle cross-correlation rerank.

The first pass uses gauge-fixed Fisher (cheap, no simulation needed)
to discard catastrophically bad candidates and keep the top K.  The
second pass simulates a lensed CMB with known phi_true, applies each
surviving estimator, and ranks by ⟨r(L)⟩ between A_L-normalized
phi_hat and phi_true — a gauge-invariant FOM that resolves the
"different physical target" ambiguity Fisher alone cannot.

Goal: with the oracle in place, Hu-Okamoto f_TT correct should rise
from its Fisher rank (~top 18%) to near the top under cross-correlation.
"""
import time
import numpy as np
import healpy as hp
from sympy import sympify

from pixell import enmap, curvedsky as cs, lensing, utils as u

import symqe as sq
from symqe import l, l1, l2, wigner_3j, P, hCT, CT, gamma_f, a, q_plus


# --- simulation -------------------------------------------------------
LMAX  = 100
NSIDE = 64
L_LO_FISHER, L_HI_FISHER, L_REF = 20, 80, 40
L_LO_ORACLE, L_HI_ORACLE        = 20, 80
TOP_K_FOR_ORACLE                = 1000   # generous; references fell to rank ~700

ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
clphi_theory = 1e-7 / (ell + 10) ** 2; clphi_theory[:2] = 0
ocltt = cltt_theory + 1.0

print("Simulating lensed CMB...", flush=True)
T_unl    = cs.rand_alm(cltt_theory,  lmax=LMAX, seed=1)
phi_true = cs.rand_alm(clphi_theory, lmax=LMAX, seed=2)
shape, wcs = enmap.fullsky_geometry(res=15 * u.arcmin)
lensed_map = lensing.lens_map_curved(
    (1,) + tuple(shape), wcs, phi_true, T_unl[None, :], spin=[0]
)[0]
T_lensed = cs.map2alm(lensed_map[0], lmax=LMAX)
px = sq.Pixelization(nside=NSIDE)


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


REFERENCES = [
    ("f_TT correct",      f_TT_correct()),
    ("f_TT wrong ladder", f_TT_wrong_ladder()),
    ("f_TT wrong m-row",  f_TT_wrong_m()),
]


# --- enumerate + Fisher prune ----------------------------------------
print("Enumerating candidates...", flush=True)
t0 = time.time()
candidates = list(sq.enumerate_candidates(
    m_max=2,
    coeffs=(1,),
    parity_factors=(sympify(1), 1 + P),
    max_leg_factors=2,
    ladder_spins=(0, 2, -2),
    spectra_X={"CT": CT},
    spectra_Y={"CT": CT},
    restrict_m_L=(-1, 1),
    symmetrize_l1l2=True,
))
print(f"  {len(candidates)} candidates ({time.time()-t0:.1f}s)", flush=True)

# Inject references for tracking.
for label, expr in REFERENCES:
    candidates.append(({"label": label, "flavor": "reference"}, expr))

print("Fisher pass (gauge-fixed) + cache bundles...", flush=True)
t0 = time.time()
fisher_results = sq.score_candidates(
    candidates,
    lmax=LMAX, Delta=2,
    hCxx=hCT, hCyy=hCT,
    user_funcs=[hCT, CT],
    spectra_args=(ocltt, cltt_theory),
    L_range=range(L_LO_FISHER, L_HI_FISHER + 1),
    L_ref=L_REF,
    px=px,
    progress=False,
    return_bundles=True,
)
print(f"  ranked {len(fisher_results)} ({time.time()-t0:.1f}s)", flush=True)


# --- keep top-K by Fisher (also keep references regardless) ----------
fisher_top = fisher_results[:TOP_K_FOR_ORACLE]
fisher_top_set = {id(r[2]) for r in fisher_top}
for fom, meta, expr, bundle in fisher_results:
    if isinstance(meta, dict) and meta.get("flavor") == "reference" and id(expr) not in fisher_top_set:
        fisher_top.append((fom, meta, expr, bundle))
        print(f"  (keeping reference '{meta['label']}' which fell outside Fisher top-K)")

print(f"Oracle pass on {len(fisher_top)} candidates...", flush=True)
spec = {"hCT": ocltt, "CT": cltt_theory}
t0 = time.time()
oracle_results = sq.oracle_rerank(
    fisher_top,
    X_input=sq.scalar_pair(T_lensed),
    Y_input=sq.scalar_pair(T_lensed),
    spectra=spec,
    phi_true_alm=phi_true,
    spec_for_A_L=(ocltt, cltt_theory),
    L_range=range(L_LO_ORACLE, L_HI_ORACLE + 1),
    output_key=+1,
    progress=True,
)
print(f"  oracle done ({time.time()-t0:.1f}s)", flush=True)
print()


# --- report ----------------------------------------------------------
def summarize_meta(meta):
    if isinstance(meta, dict) and meta.get("flavor") == "reference":
        return f"REF: {meta['label']}"
    if isinstance(meta, dict):
        mL, mX, mY = meta["m_triple"]
        Xs = str(meta["X_filter"])[:24]
        Ys = str(meta["Y_filter"])[:24]
        flav = meta["flavor"]
        return f"m=({mL:+d},{mX:+d},{mY:+d}) {flav:3s} X={Xs} Y={Ys}"
    return "?"


N_SHOW = 20
print(f"Top {N_SHOW} by oracle ⟨r(L)⟩ on L ∈ [{L_LO_ORACLE},{L_HI_ORACLE}]:")
print(f"{'rk':>3s}  {'<r>':>8s}  {'fisher_FOM':>10s}  description")
print("-" * 100)
for i, (score, fom, meta, expr) in enumerate(oracle_results[:N_SHOW]):
    print(f"{i+1:>3d}  {score:>+8.4f}  {fom:>10.3e}  {summarize_meta(meta)}")

print()
print("Reference candidate positions in oracle ranking:")
for j, (label, _) in enumerate(REFERENCES):
    for i, (score, fom, meta, _expr) in enumerate(oracle_results):
        if isinstance(meta, dict) and meta.get("label") == label:
            print(f"  {label:24s}  rank {i+1:>3d}/{len(oracle_results)}   "
                  f"<r>={score:+.4f}   fisher={fom:.3e}")
            break

print()
print("Interpretation:")
print("  Fisher ranks structurally-different weights by self-consistent noise,")
print("  which doesn't capture WHICH physical target is being measured.  The")
print("  oracle cross-correlation with phi_true is the gauge-invariant rerank")
print("  that resolves this — Hu-Okamoto f_TT should now sit near the top.")
