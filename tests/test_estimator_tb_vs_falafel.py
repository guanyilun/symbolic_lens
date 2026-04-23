"""Bit-for-bit validation of the symbolic TB estimator against falafel."""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCB, f_TB, CTE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma, compile_tb


LMAX, NSIDE = 200, 256


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_pol_only, filter_alms

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clbb = 5.0  / (ell + 10) ** 2 + 1e-4
    clte = 0.5 * np.sqrt(cltt * clbb + 1e-10)
    nltt = 1.0 * np.ones_like(cltt)
    nlbb = 1.5 * np.ones_like(clbb)
    ocltt = cltt + nltt
    oclbb = clbb + nlbb

    # f^{TB} uses the hCT·hCB denominator (X=T, Y=B inv-var filtered).
    g_TB = f_TB(px=+1) / (hCT(l1) * hCB(l2))     # Δ = 1 for TB
    terms = strip_gamma(compile_estimator(g_TB))
    print(f"Compiled {len(terms)} TB terms.")
    sym_emit = compile_tb(terms, lmax=LMAX, nside=NSIDE)

    # synthetic alms (independent T and B; TB uses CTE as response)
    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True)

    spectra = {"hCT": ocltt, "hCB": oclbb, "CTE": clte}
    phi_sym = sym_emit(T_alm, B_alm, spectra)

    # falafel reference: dmap('Ptb') = pest(xalm('e_t0'), 0, 0, fBalm)
    #   xalm('e_t0') = filter(T_iv, CTE) = T/ocltt · CTE
    #   fBalm = B/oclbb
    px = pixelization(nside=NSIDE)
    fTalm = filter_alms(T_alm.copy(), 1.0 / ocltt)     # iv T
    fBalm = filter_alms(B_alm.copy(), 1.0 / oclbb)     # iv B
    X_pseudo_E = filter_alms(fTalm.copy(), clte)       # T·CTE/ocltt
    zero = np.zeros_like(T_alm)
    phi_curl = qe_pol_only(px, X_pseudo_E, zero, zero, fBalm, LMAX)
    phi_falafel = phi_curl[0] if phi_curl.ndim == 2 else phi_curl

    mask = np.abs(phi_falafel) > 1e-10
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"pair_coeff = {sym_emit.pair_coeff}")
    print(f"TB symbolic / falafel:")
    print(f"  median |ratio|: {np.median(np.abs(ratios)):.6e}")
    print(f"  median ratio (complex): {np.median(ratios.real):.6e} + {np.median(ratios.imag):.6e}j")
    print(f"  std/mean of |ratio|:    {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
