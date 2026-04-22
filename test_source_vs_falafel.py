"""
Validate the symbolic source (amplitude / ε) TT estimator against
``falafel.qe.qe_source``, on CAR geometry.

Note on convention: falafel's ``qe_source`` by default computes
``sum_{l,l'} 3j · T_l/ĉ · T_l'/ĉ`` with NO response-weighting — i.e.,
both legs are plain inverse-variance filtered T.  Our Namikawa-style
``f_ampl_TT`` includes C_T as a response on one leg
(``W·C_T^{l'} + p_ε W(swap)·C_T^l``), so the recipe has response
weighting baked in.  To compare apples-to-apples we pass the SAME alm
to falafel on both legs after applying our (response × iv) filter.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, f_ampl_TT
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import compile_source_tt_native, Pixelization


LMAX = 200


def main():
    import healpy as hp
    from pixell import enmap
    from falafel.qe import pixelization as falafel_pix, qe_source

    res = np.deg2rad(12.0 / 60.0)
    shape, wcs = enmap.fullsky_geometry(res=res)
    px_fal = falafel_pix(shape=shape, wcs=wcs)
    px_nat = Pixelization(shape=shape, wcs=wcs)

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    nltt = 1.0 * np.ones_like(cltt)
    ocltt = cltt + nltt

    g = f_ampl_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    emit = compile_source_tt_native(terms, LMAX, px=px_nat)
    print(f"Source compiled to {len(terms)} terms, pair_coeff = {emit.pair_coeff}")

    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)

    # Native: symbolic emitter applies the two filters internally
    src_native = emit(T_alm, {"hCT": ocltt, "CT": cltt})

    # Falafel reference: qe_source(px, mlmax, fTalm, xfTalm).
    # Our symbolic f splits filters as (response × iv).  Apply the response
    # to one copy of T and iv to the other, pass as the two inputs.
    from falafel.qe import filter_alms
    T_response = filter_alms(T_alm.copy(), cltt / ocltt)
    T_iv       = filter_alms(T_alm.copy(), 1.0 / ocltt)
    src_falafel = qe_source(px_fal, LMAX, T_iv, xfTalm=T_response)

    mask = np.abs(src_falafel) > 1e-10
    ratios = src_native[mask] / src_falafel[mask]
    print(f"Source Native / Falafel:")
    print(f"  median |ratio|: {np.median(np.abs(ratios)):.6e}")
    print(f"  std/mean:       {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
