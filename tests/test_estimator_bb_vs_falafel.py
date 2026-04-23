"""Bit-for-bit validation of the symbolic BB estimator against falafel."""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCB, f_BB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, compile_bb


LMAX, NSIDE = 200, 256


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_pol_only, filter_alms

    ell = np.arange(LMAX + 1, dtype=float)
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    nlbb = 1.5 * np.ones_like(clbb)
    oclbb = clbb + nlbb

    g_BB = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g_BB))
    sym_emit = compile_bb(terms, lmax=LMAX, nside=NSIDE)

    np.random.seed(42)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True)

    spectra = {"hCB": oclbb, "CB": clbb}
    phi_sym = sym_emit(B_alm, spectra)

    px = pixelization(nside=NSIDE)
    X_resp = filter_alms(B_alm.copy(), clbb / oclbb)
    Y_iv   = filter_alms(B_alm.copy(), 1.0 / oclbb)
    zero = np.zeros_like(B_alm)
    phi_curl = qe_pol_only(px, zero, X_resp, zero, Y_iv, LMAX)
    phi_falafel = phi_curl[0] if phi_curl.ndim == 2 else phi_curl

    mask = np.abs(phi_falafel) > 1e-10
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"pair_coeff = {sym_emit.pair_coeff}")
    print(f"BB symbolic / falafel:")
    print(f"  median |ratio|: {np.median(np.abs(ratios)):.6e}")
    print(f"  std/mean:       {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
