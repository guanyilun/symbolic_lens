"""
Bit-for-bit validation: symbolic TT estimator vs falafel.qe.qe_temperature_only.

The symbolic analyzer produces four EstimatorTerms for the TT lensing
recipe.  After γ-stripping, we identify the X-gradient pair
(spin_X = ±1, spin_Y = 0, spin_L = ∓1), read off the filters, and run
them through falafel.qe.qe_temperature_only.  Since X = Y = T for TT,
the Y-gradient pair (spins swapped) contributes identically, so the
emitter multiplies by 2.

Test: run the same synthetic temperature alm through
  (a) our symbolic-derived filters + falafel's qe_temperature_only
  (b) falafel's qe_temperature_only with MANUALLY derived filters
and check bit-for-bit agreement up to an overall constant that we
identify and report.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, f_TT
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, compile_tt


LMAX  = 200
NSIDE = 256


def make_test_cl(lmax):
    ell = np.arange(lmax + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    nltt = 1.0 * np.ones_like(cltt)
    return cltt, nltt


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_temperature_only, filter_alms

    cltt, nltt = make_test_cl(LMAX)
    ocltt = cltt + nltt

    # --- (1) Compile the symbolic pipeline ---
    g_TT = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = compile_estimator(g_TT)
    terms = strip_gamma(terms)
    sym_emit = compile_tt(terms, lmax=LMAX, nside=NSIDE)

    print(f"Symbolic compilation produced {len(terms)} γ-stripped terms.")
    print(f"Reference X-gradient term:")
    print(f"  coeff     = {sym_emit.ref_term.coeff}")
    print(f"  pair_coeff (×2 for X↔Y) = {sym_emit.pair_coeff}")
    print(f"  spin_X={sym_emit.ref_term.spin_X}  spin_Y={sym_emit.ref_term.spin_Y}  "
          f"spin_L={sym_emit.ref_term.spin_L}")

    # --- (2) Synthetic alm ---
    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)

    # --- (3) Symbolic pipeline output ---
    spectra = {"hCT": ocltt, "CT": cltt}
    phi_sym = sym_emit(T_alm, spectra)

    # --- (4) Hand-rolled falafel reference ---
    px = pixelization(nside=NSIDE)
    X_response = filter_alms(T_alm.copy(), cltt / ocltt)
    Y_iv       = filter_alms(T_alm.copy(), 1.0 / ocltt)
    phi_curl_falafel = qe_temperature_only(px, X_response, Y_iv, LMAX)
    phi_falafel = phi_curl_falafel[0] if phi_curl_falafel.ndim == 2 else phi_curl_falafel

    # --- (5) Compare ---
    print("\nFirst 10 alm entries (falafel, symbolic):")
    for i in range(10):
        print(f"  {i:3d}  falafel={phi_falafel[i]:+.4e}  symbolic={phi_sym[i]:+.4e}")

    mask = np.abs(phi_falafel) > 1e-6
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"\nSymbolic / falafel ratio statistics:")
    print(f"  median:  {np.median(np.abs(ratios)):.6e}")
    print(f"  mean:    {np.mean(np.abs(ratios)):.6e}")
    print(f"  std:     {np.std(np.abs(ratios)):.6e}")
    print(f"  std/mean (should be ~0 if ratio is constant):  "
          f"{np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
