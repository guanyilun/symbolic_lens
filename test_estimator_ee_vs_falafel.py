"""Bit-for-bit validation of the symbolic EE estimator against falafel.qe.qe_pol_only."""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCE, f_EE
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, compile_ee


LMAX  = 200
NSIDE = 256


def make_test_cl(lmax):
    ell = np.arange(lmax + 1, dtype=float)
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    nlee = 1.5 * np.ones_like(clee)
    return clee, nlee


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_pol_only, filter_alms

    clee, nlee = make_test_cl(LMAX)
    oclee = clee + nlee

    g_EE = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = compile_estimator(g_EE)
    terms = strip_gamma(terms)
    sym_emit = compile_ee(terms, lmax=LMAX, nside=NSIDE)

    print(f"pair_coeff = {sym_emit.pair_coeff}")

    np.random.seed(42)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True)

    # --- symbolic ---
    spectra = {"hCE": oclee, "CE": clee}
    phi_sym = sym_emit(E_alm, spectra)

    # --- falafel reference ---
    px = pixelization(nside=NSIDE)
    X_resp = filter_alms(E_alm.copy(), clee / oclee)
    Y_iv   = filter_alms(E_alm.copy(), 1.0 / oclee)
    zero = np.zeros_like(E_alm)
    phi_curl = qe_pol_only(px, X_resp, zero, Y_iv, zero, LMAX)
    phi_falafel = phi_curl[0] if phi_curl.ndim == 2 else phi_curl

    # --- compare ---
    print("\nFirst 10 alm entries (falafel, symbolic):")
    for i in range(10):
        print(f"  {i:3d}  falafel={phi_falafel[i]:+.4e}  symbolic={phi_sym[i]:+.4e}")

    mask = np.abs(phi_falafel) > 1e-8
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"\nSymbolic / falafel ratio statistics:")
    print(f"  median:  {np.median(np.abs(ratios)):.6e}")
    print(f"  mean:    {np.mean(np.abs(ratios)):.6e}")
    print(f"  std:     {np.std(np.abs(ratios)):.6e}")
    print(f"  std/mean: {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
