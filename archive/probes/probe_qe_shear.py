"""Compare a candidate symbolic f_shear against falafel's qe_shear.

qe_shear is the Schaan-Ferraro-style CMB shear estimator.  Pipeline:
  X-leg: fT alm × sqrt((l-1)·l·(l+1)·(l+2))   → alm2map_spin(0, 2)
  Y-leg: T alm                                 → alm2map(spin=0)
  prod:  X_map (spin 2) × Y_map (spin 0)       → spin 2 complex map
  out:   map2alm_spin(prod, spin_alm=2, spin_transform=2) → (E-like, B-like)
         shear_alm = -2 · E-like_alm

Notable: prod's "natural" spin is 2, so the output extraction uses
spin_alm=2 (not spin_alm=0 like lensing TT does).  Our compile_native
hardcodes spin_alm=0 — this probe checks whether that matters.
"""
import numpy as np
import healpy as hp

from symqe.engine.l12_sum import l, l1, l2, P, wigner_3j
from symqe.engine.namikawa import gamma_f, hCT
from symqe.engine.estimator_native import (
    Pixelization, compile_native, scalar_pair, fuse_spin_pairs,
    _gradient_spin,  # for sanity
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1
spec = {"hCT": ocltt, "CT": cltt}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
fT = hp.almxfl(T, 1.0 / ocltt)  # IV-filtered T
px = Pixelization(nside=NSIDE)


# =======================================================================
# Reference: faithful reproduction of falafel.qe.qe_shear using our
# inlined primitives so it doesn't need falafel installed.
# =======================================================================
def qe_shear_reference(T_alm, fT_alm, lmax, px):
    """Bit-for-bit reproduction of falafel.qe.qe_shear, using only our
    inlined SHT primitives.  Returns shear_alm (1D array)."""
    from pixell import curvedsky as cs
    ells = np.arange(lmax)

    # rmapT = alm2map(T) — scalar T map
    rmapT = px.alm2map(np.stack((T_alm, T_alm)), spin=0, ncomp=1, mlmax=lmax)[0]

    # t_alm = fT × sqrt((l-1)·l·(l+1)·(l+2))
    # Note: this is FLOAT (input to np.sqrt is int product, sqrt returns float),
    # NOT int-truncated like _gradient_spin.  No truncation bug here.
    fl = np.sqrt((ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0))
    t_alm = cs.almxfl(fT_alm.copy(), fl)

    # alm2map_spin(t, spin_alm=0, spin_transform=2) → spin ±2 map pair
    rmap = px.alm2map_spin(np.stack([t_alm, t_alm]), 0, 2, ncomp=2, mlmax=lmax)

    # prod = (M+2, M-2) · T
    prodmap = rmap * rmapT
    realsp2 = prodmap[0]
    realsm2 = prodmap[1]

    # map2alm_spin(prod, spin_alm=±2, spin_transform=2)
    # Note falafel only USES res1; res2 is computed but discarded.
    res1 = px.map2alm_spin(realsp2, lmax, 2, 2)

    # rot2dalm(res1, spin=2): (-1)^spin = +1 here, so:
    #   ttalmsp2 = -(res1[0] + i·res1[1])
    #   ttalmsm2 = -(res1[0] - i·res1[1])
    #   shear_alm = ttalmsp2 + ttalmsm2 = -2 · res1[0]
    shear_alm = -2.0 * res1[0]
    return shear_alm


# =======================================================================
# Symbolic candidate f_shear.
#
# Structure: shear is the spin-2 quadratic combination of T·T.  Following
# the Namikawa W convention with the chained shear ladder factor on l1
# (instead of the lensing single-grad sqrt(l1·(l1+1))):
#
#   shear ladder = sqrt((l1-1)·l1·(l1+1)·(l1+2))
#                = sqrt(l1·(l1+1)) · sqrt((l1-1)·(l1+2))
#                = 2·|a(l1, 0)|·|a(l1, -2)| / 2       (matching factors)
#
# 3j m-pattern: output spin |sL|=2, X-leg spin 0 (input is T), Y-leg
# spin 0 (input is also T).  The 3j must enforce m_L + m_X + m_Y = 0
# and supply the spin-2 coupling.  Try (m_L, m_X, m_Y) = (2, -2, 0):
# this gives spin_L=2, spin_X=-2 (X gets ladder up by 2), spin_Y=0.
#
# The asymmetric structure (only one ladder) means f_shear is a single
# term, not a symmetric pair like f_TT.
# =======================================================================
from sympy import sqrt as sp_sqrt
from symqe.engine.namikawa import a  # Namikawa a(l, s) ladder factor

def f_shear_candidate():
    """A guess at the symbolic f_shear.

    Mirrors qe_shear's geometry: shear ladder on l1 = 2·a(l1,0)·a(l1,-2)·sqrt(2)
    (since a(l,s) carries 1/sqrt(2)), 3j with spin_L=2, spin_X=-2, spin_Y=0.
    No CT response needed (shear is purely geometric)."""
    shear_ladder = sp_sqrt((l1 - 1) * l1 * (l1 + 1) * (l1 + 2))
    return shear_ladder * gamma_f(l1, l, l2) * \
           wigner_3j(l, l1, l2, 2, -2, 0)


# =======================================================================
# Try our pipeline.
# =======================================================================
print("=" * 70)
print("qe_shear vs symbolic compile_native")
print("=" * 70)

# Compute reference
ref = qe_shear_reference(T, fT, LMAX, px)
print(f"\nReference qe_shear: shape={ref.shape}, dtype={ref.dtype}")
print(f"  rms = {np.sqrt(np.mean(np.abs(ref)**2)):.4e}")

# Compile symbolic
g_shear = f_shear_candidate() / (hCT(l1) * hCT(l2))  # divide by IV filters as in g
try:
    terms = strip_gamma(compile_estimator(g_shear))
    print(f"\nCompiled to {len(terms)} EstimatorTerm s")
    plans = fuse_spin_pairs(terms)
    print(f"Fused into {len(plans)} FusedPlan s")
    for i, p in enumerate(plans):
        print(f"  Plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}")
        print(f"    X_filter: {p.X_filter}")
        print(f"    Y_filter: {p.Y_filter}")
        print(f"    L_factor: {p.L_factor}")
        for k, v in p.coeffs.items():
            print(f"    coeff[{k}] = {complex(v):.6e}")
except Exception as e:
    print(f"\ncompile_estimator FAILED: {type(e).__name__}: {e}")
    raise

# Run compile_native (with int-trunc rule disabled, to isolate the structural issue)
import symqe.engine.estimator_native as en
_orig_detect = en._detect_truncatable_ladder
en._detect_truncatable_ladder = lambda atom: None  # disable rule 7

try:
    out = compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)
    print(f"\ncompile_native output keys: {list(out.keys())}")
    for k, v in out.items():
        v = np.asarray(v)
        print(f"  output[{k}]: shape={v.shape}, rms={np.sqrt(np.mean(np.abs(v)**2)):.4e}")
except Exception as e:
    print(f"\ncompile_native FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()


def per_L(label, gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n-- {label} --")
    print(f"{'L':>4s} {'ref_cl':>12s} {'gen_cl':>12s} {'gen/ref':>12s} {'signed_slope':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            r = np.sqrt(gc[L] / rc[L])
            s = xc[L] / rc[L]
            print(f"{L:>4d} {rc[L]:>12.4e} {gc[L]:>12.4e} {r:>12.5e} {s:>14.5e}")

# Compare each output channel against the reference
if 'out' in dir():
    for k in out:
        gen = np.asarray(out[k])
        if gen.shape == ref.shape:
            per_L(f"compile_native output[{k}] vs qe_shear", gen, ref)
