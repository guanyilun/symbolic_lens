"""Bit-for-bit validation of the symbolic EB estimator against falafel.

Note: falafel's EB path (``pest(xalm('e0'), 0, 0, fBalm)``) assumes the
primordial C_B = 0, so we set ``CB = 0`` in the test spectra for apples-
to-apples comparison.  This zeroes out the "B-response" term of f^{EB}
and leaves only the "E-response" term that falafel computes.
"""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, hCB, f_EB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, compile_eb


LMAX, NSIDE = 200, 256


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_pol_only, filter_alms

    ell = np.arange(LMAX + 1, dtype=float)
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb_for_noise = 5.0 / (ell + 10) ** 2 + 1e-4  # used only to synth B alm
    nlee = 1.5 * np.ones_like(clee)
    nlbb = 1.5 * np.ones_like(clbb_for_noise)
    oclee = clee + nlee
    oclbb = clbb_for_noise + nlbb

    # For f^{EB} under the C_B=0 assumption, the first term (W^-·C_B)
    # vanishes; only the second (W^-(swap)·C_E) survives.
    g_EB = f_EB(px=+1) / (hCE(l1) * hCB(l2))    # Δ = 1 for EB
    terms = strip_gamma(compile_estimator(g_EB))
    print(f"Compiled {len(terms)} EB terms.")
    sym_emit = compile_eb(terms, lmax=LMAX, nside=NSIDE)

    np.random.seed(42)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True)

    # C_B = 0 in the response spectrum (not the synth spectrum).
    spectra = {"hCE": oclee, "hCB": oclbb, "CE": clee,
               "CB": np.zeros_like(clee)}
    phi_sym = sym_emit(E_alm, B_alm, spectra)

    px = pixelization(nside=NSIDE)
    X_resp = filter_alms(E_alm.copy(), clee / oclee)
    Y_iv   = filter_alms(B_alm.copy(), 1.0 / oclbb)
    zero = np.zeros_like(E_alm)
    phi_curl = qe_pol_only(px, X_resp, zero, zero, Y_iv, LMAX)
    phi_falafel = phi_curl[0] if phi_curl.ndim == 2 else phi_curl

    mask = np.abs(phi_falafel) > 1e-10
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"pair_coeff = {sym_emit.pair_coeff}")
    print(f"EB symbolic / falafel:")
    print(f"  median |ratio|: {np.median(np.abs(ratios)):.6e}")
    print(f"  median ratio (complex): {np.median(ratios.real):.6e} + {np.median(ratios.imag):.6e}j")
    print(f"  std/mean of |ratio|:    {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
