"""
Regression: the generic compile_native emitter vs the hand-coded
per-estimator compile_*_native functions.  All 8 estimators (TT, EE, BB,
TB, EB, TE, rotation-EB, source-TT) should produce identical outputs
once the channel-extraction logic and any overall scaling are applied.

This test also surfaces any constant offset: if generic = scale ·
hand-coded, we identify and document ``scale`` per estimator.  The
goal is to arrive at generic = hand-coded bit-for-bit across all of
them; any discrepancy reveals a fusion/coefficient bookkeeping bug.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import (
    hCT, hCE, hCB, hCTE, CT, CE, CB, CTE,
    f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
    f_ampl_TT, f_rot_EB,
)
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    # hand-coded emitters (regression oracle)
    compile_tt_native, compile_ee_native, compile_bb_native,
    compile_tb_native, compile_eb_native, compile_te_native,
    compile_rot_eb_native, compile_source_tt_native,
    # generic emitter
    compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    Pixelization,
)


LMAX, NSIDE = 200, 256


def compare(alm_ref, alm_gen, label, tol=1e-10):
    alm_ref = np.asarray(alm_ref); alm_gen = np.asarray(alm_gen)
    mask = np.abs(alm_ref) > 1e-8
    if mask.sum() == 0:
        print(f"  {label:12s}  NO SIGNAL")
        return
    ratios = alm_gen[mask] / alm_ref[mask]
    median = np.median(np.abs(ratios))
    std_rel = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
    status = "MATCH" if (std_rel < tol and abs(median - 1) < tol) else f"offset={median:.4e}"
    print(f"  {label:12s}  median={median:.6e}  std/mean={std_rel:.3e}  {status}")


def main():
    import healpy as hp

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    clte = 0.5 * np.sqrt(cltt * clee)
    nltt, nlee, nlbb = np.ones_like(cltt), 1.5 * np.ones_like(clee), 1.5 * np.ones_like(clbb)
    ocltt, oclee, oclbb = cltt + nltt, clee + nlee, clbb + nlbb

    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    E = hp.synalm(oclee, lmax=LMAX, new=True)
    B = hp.synalm(oclbb, lmax=LMAX, new=True)

    px = Pixelization(nside=NSIDE)

    print("Generic compile_native vs hand-coded emitters (HEALPix):\n")

    # --- TT ---
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCT": ocltt, "CT": cltt}

    ref = compile_tt_native(terms, LMAX, px=px)(T, spec)
    gen = compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)
    # Generic output is a dict; for lensing TT phi is at +1 key
    compare(ref, gen[+1], "TT (phi)")

    # --- EE ---
    g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCE": oclee, "CE": clee}

    ref = compile_ee_native(terms, LMAX, px=px)(E, spec)
    gen = compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)
    compare(ref, gen[+1], "EE (phi)")

    # --- BB ---
    g = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCB": oclbb, "CB": clbb}

    ref = compile_bb_native(terms, LMAX, px=px)(B, spec)
    gen = compile_native(terms, LMAX, px=px)(pol_B_pair(B), pol_B_pair(B), spec)
    compare(ref, gen[+1], "BB (phi)")

    # --- Rotation EB ---
    g = f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)}

    ref = compile_rot_eb_native(terms, LMAX, px=px)(E, B, spec)
    gen = compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)
    compare(ref, gen[0], "ROT-EB (α)")

    # --- Source TT ---
    g = f_ampl_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCT": ocltt, "CT": cltt}

    ref = compile_source_tt_native(terms, LMAX, px=px)(T, spec)
    gen = compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)
    compare(ref, gen[0], "SRC-TT")


if __name__ == "__main__":
    main()
