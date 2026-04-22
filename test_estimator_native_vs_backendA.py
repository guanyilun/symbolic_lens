"""
Regression: the native SHT backend (no falafel runtime dep) must match
the falafel-delegating backend A bit-for-bit on all lensing estimators.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, hCB, f_TT, f_EE, f_BB
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, compile_tt, compile_ee, compile_bb
from sym_utils.estimator_native import (
    compile_tt_native, compile_ee_native, compile_bb_native,
)

LMAX, NSIDE = 200, 256


def compare(phi_A, phi_B, label):
    mask = np.abs(phi_A) > 1e-8
    ratios = phi_B[mask] / phi_A[mask]
    std_rel = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
    print(f"  {label:6s}  median={np.median(np.abs(ratios)):.6e}   "
          f"std/mean={std_rel:.3e}  "
          f"{'MATCH' if std_rel < 1e-10 and abs(np.median(np.abs(ratios)) - 1) < 1e-10 else 'MISMATCH'}")


def main():
    import healpy as hp

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0  / (ell + 10) ** 2 + 1e-4
    nltt = np.ones_like(cltt); nlee = 1.5*np.ones_like(clee); nlbb = 1.5*np.ones_like(clbb)
    ocltt, oclee, oclbb = cltt+nltt, clee+nlee, clbb+nlbb

    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True)

    print("Native vs backend-A (falafel-delegating):")

    # TT
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    phi_A      = compile_tt       (terms, lmax=LMAX, nside=NSIDE)(T_alm, {"hCT": ocltt, "CT": cltt})
    phi_native = compile_tt_native(terms, lmax=LMAX, nside=NSIDE)(T_alm, {"hCT": ocltt, "CT": cltt})
    compare(phi_A, phi_native, "TT")

    # EE
    g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    phi_A      = compile_ee       (terms, lmax=LMAX, nside=NSIDE)(E_alm, {"hCE": oclee, "CE": clee})
    phi_native = compile_ee_native(terms, lmax=LMAX, nside=NSIDE)(E_alm, {"hCE": oclee, "CE": clee})
    compare(phi_A, phi_native, "EE")

    # BB
    g = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    phi_A      = compile_bb       (terms, lmax=LMAX, nside=NSIDE)(B_alm, {"hCB": oclbb, "CB": clbb})
    phi_native = compile_bb_native(terms, lmax=LMAX, nside=NSIDE)(B_alm, {"hCB": oclbb, "CB": clbb})
    compare(phi_A, phi_native, "BB")


if __name__ == "__main__":
    main()
