"""Empirical probe of map2alm_spin: how do the complex pixel maps
M_{+s}·Y, M_{-s}·Y, and their sums/diffs relate under map2alm_spin?

Goal: work out the linear map from a symbolic sign-signature
(s_X, s_Y, s_L) → complex pixel map contribution that, when ONE
combined complex map is fed to ONE map2alm_spin(spin=|s_L|), yields
the same phi/curl alm pair as the hand-coded falafel path.

The generic compile_native currently runs one map2alm_spin per term
and indexes (grad, curl) by sign(s_L).  By linearity the answers add,
so the question isn't "does linearity hold?" — it's "is the pixel
map we're forming the RIGHT one?"

Hand-coded TT forms exactly ONE complex dmap
    dmap = -M_{+1}(X_filt)·Y_scalar
and calls map2alm_spin(spin=1) once.

If the symbolic expansion for TT contains signatures both
(sX=+1, sY=0, sL=+1) AND (sX=-1, sY=0, sL=-1) (or similar), the
generic path forms BOTH M_{+1}·Y and M_{-1}·Y as separate prod's
and runs TWO map2alm_spin calls.  These sum correctly by linearity,
but the resulting (grad, curl) pair is NOT just the phi/curl of
dmap_hand — it includes contributions from both ±spin components.

This probe:
  (a) Inspect the sign-signature distribution of compile_estimator
      output for lensing TT — what (sX, sY, sL) tuples appear, with
      what coefficients?
  (b) For a toy scalar alm X, compute M_{+1}·X_scalar and M_{-1}·X_scalar
      as complex pixel maps and check their relationship
      (should be complex conjugate for real inputs).
  (c) Compute map2alm_spin of each and see how (grad, curl) alms
      transform.
  (d) Work out the linear combination of per-signature pixel maps
      that reproduces the hand-coded -M_{+1}·Y dmap.
"""
import numpy as np
import healpy as hp
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, f_TT
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, fuse_spin_pairs, _alm_to_signed_pair, _eval_atom,
    scalar_pair,
)

LMAX, NSIDE = 64, 64


# ---------------------------------------------------------------------
# (a) what sign-signatures does the symbolic expansion produce for TT?
# ---------------------------------------------------------------------

def inspect_signatures():
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    fused = fuse_spin_pairs(terms)
    print(f"=== Lensing TT: {len(terms)} terms → {len(fused)} fused plans ===\n")
    for i, plan in enumerate(fused):
        print(f"Plan {i}: |sX|={plan.abs_spin_X}  |sY|={plan.abs_spin_Y}  |sL|={plan.abs_spin_L}")
        for sig, c in sorted(plan.coeffs.items()):
            print(f"    sig={sig}  coeff={c:+.6g}")
    return fused


# ---------------------------------------------------------------------
# (b, c) construct M_{+1}·Y and M_{-1}·Y, send through map2alm_spin
# ---------------------------------------------------------------------

def probe_pair_relation():
    """For a real scalar X, confirm that M_{-|s|}(X) = conj(M_{+|s|}(X)).
    Then send each · Y through map2alm_spin(spin=1) and inspect the
    (grad, curl) alm pair for each."""
    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    np.random.seed(0)
    X = hp.synalm(cltt, lmax=LMAX, new=True)
    Y = hp.synalm(cltt, lmax=LMAX, new=True)

    px = Pixelization(nside=NSIDE)

    # M_{+1}, M_{-1} from alm2map_spin with spin_alm=0, spin_transform=1
    X_pair = np.stack([X, X])
    M_plus, M_minus = _alm_to_signed_pair(px, X_pair, spin_alm_in=0,
                                          abs_spin_out=1, lmax=LMAX)
    print("=== (b) Check M_{-1} = conj(M_{+1}) for real scalar X ===")
    diff = np.max(np.abs(M_minus - np.conj(M_plus)))
    print(f"    max|M_-1 - conj(M_+1)| = {diff:.3e}")

    # Y scalar pixel map
    Y_map = px.alm2map(Y, spin=0, ncomp=1, mlmax=LMAX)[0]

    # Prods
    prod_plus  = M_plus * Y_map
    prod_minus = M_minus * Y_map      # = conj(prod_plus) since Y_map real
    prod_sum   = prod_plus + prod_minus  # = 2·Re(prod_plus), purely real
    prod_diff  = prod_plus - prod_minus  # = 2i·Im(prod_plus), purely imag

    # map2alm_spin of each
    def run(prod, tag):
        a = px.map2alm_spin(prod, lmax=LMAX, spin_alm=0, spin_transform=1)
        a = np.asarray(a)
        print(f"  {tag}: grad[0:3]={a[0][:3]}  curl[0:3]={a[1][:3]}")
        return a

    print("\n=== (c) map2alm_spin outputs for each prod ===")
    a_plus  = run(prod_plus,  "M_+1·Y     ")
    a_minus = run(prod_minus, "M_-1·Y     ")
    a_sum   = run(prod_sum,   "(M_+1+M_-1)·Y")
    a_diff  = run(prod_diff,  "(M_+1-M_-1)·Y")

    # Linearity sanity
    print("\n=== linearity sanity ===")
    print(f"    max|a_sum  - (a_+ + a_-)| = {np.max(np.abs(a_sum - (a_plus + a_minus))):.3e}")
    print(f"    max|a_diff - (a_+ - a_-)| = {np.max(np.abs(a_diff - (a_plus - a_minus))):.3e}")

    # The critical question: how do (a_+, a_-) relate?
    # In particular: is a_plus[0] (grad of M_+1·Y) = ±a_minus[1] (curl of M_-1·Y) or similar?
    print("\n=== key relationships between a_plus and a_minus ===")
    print(f"    max|grad(M_+1·Y) - conj(grad(M_-1·Y))|= "
          f"{np.max(np.abs(a_plus[0] - np.conj(a_minus[0]))):.3e}")
    print(f"    max|grad(M_+1·Y) + conj(grad(M_-1·Y))|= "
          f"{np.max(np.abs(a_plus[0] + np.conj(a_minus[0]))):.3e}")
    print(f"    max|curl(M_+1·Y) - conj(curl(M_-1·Y))|= "
          f"{np.max(np.abs(a_plus[1] - np.conj(a_minus[1]))):.3e}")
    print(f"    max|curl(M_+1·Y) + conj(curl(M_-1·Y))|= "
          f"{np.max(np.abs(a_plus[1] + np.conj(a_minus[1]))):.3e}")

    # Also check: for REAL prod (=prod_sum/2), curl should be very small
    # (a real deflection map has only grad mode up to noise).
    print("\n=== purely real prod_sum: expected curl ≈ 0? ===")
    print(f"    |grad(Re·2)| median = {np.median(np.abs(a_sum[0])):.3e}")
    print(f"    |curl(Re·2)| median = {np.median(np.abs(a_sum[1])):.3e}")
    print("    (both nonzero → (Q,U) from Re(prod) alone encodes both E,B — as expected")
    print("     since we're feeding (Q,U) = (Re(prod), Im(prod)) not (Re, 0).)")
    print(f"\n    prod_sum imag part max |Im| = {np.max(np.abs(prod_sum.imag)):.3e} (should be ~0)")
    print(f"    prod_diff real part max |Re| = {np.max(np.abs(prod_diff.real)):.3e} (should be ~0)")

    return a_plus, a_minus


def main():
    inspect_signatures()
    print()
    probe_pair_relation()


if __name__ == "__main__":
    main()
