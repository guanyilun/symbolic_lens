"""
Regression: the native SHT backend (no falafel runtime dep) must match
the falafel-delegating backend A bit-for-bit on all lensing estimators.
"""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import (
    hCT, hCE, hCB, f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import (
    strip_gamma,
    compile_tt as compile_tt_bA, compile_ee as compile_ee_bA,
    compile_bb as compile_bb_bA, compile_tb as compile_tb_bA,
    compile_eb as compile_eb_bA, compile_te as compile_te_bA,
)
from symqe.engine.estimator_native import (
    compile_tt, compile_ee, compile_bb,
    compile_tb, compile_eb, compile_te,
)

LMAX, NSIDE = 200, 256


def compare(phi_A, phi_B, label):
    mask = np.abs(phi_A) > 1e-8
    ratios = phi_B[mask] / phi_A[mask]
    std_rel = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
    median = np.median(np.abs(ratios))
    status = "MATCH" if (std_rel < 1e-10 and abs(median - 1) < 1e-10) else "MISMATCH"
    print(f"  {label:4s}  median={median:.6e}  std/mean={std_rel:.3e}  {status}")


def main():
    import healpy as hp

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    clte = 0.5 * np.sqrt(cltt * clee)
    nltt, nlee, nlbb = (np.ones_like(cltt), 1.5*np.ones_like(clee), 1.5*np.ones_like(clbb))
    ocltt, oclee, oclbb = cltt + nltt, clee + nlee, clbb + nlbb

    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    E = hp.synalm(oclee, lmax=LMAX, new=True)
    B = hp.synalm(oclbb, lmax=LMAX, new=True)

    print("Native vs backend-A (falafel-delegating):")

    # TT
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCT": ocltt, "CT": cltt}
    compare(compile_tt_bA    (terms, LMAX, NSIDE)(T, spec),
            compile_tt(terms, LMAX, nside=NSIDE)(T, spec), "TT")

    # EE
    g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCE": oclee, "CE": clee}
    compare(compile_ee_bA    (terms, LMAX, NSIDE)(E, spec),
            compile_ee(terms, LMAX, nside=NSIDE)(E, spec), "EE")

    # BB
    g = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCB": oclbb, "CB": clbb}
    compare(compile_bb_bA    (terms, LMAX, NSIDE)(B, spec),
            compile_bb(terms, LMAX, nside=NSIDE)(B, spec), "BB")

    # TB
    g = f_TB(px=+1) / (hCT(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCT": ocltt, "hCB": oclbb, "CTE": clte}
    compare(compile_tb_bA    (terms, LMAX, NSIDE)(T, B, spec),
            compile_tb(terms, LMAX, nside=NSIDE)(T, B, spec), "TB")

    # EB
    g = f_EB(px=+1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)}
    compare(compile_eb_bA    (terms, LMAX, NSIDE)(E, B, spec),
            compile_eb(terms, LMAX, nside=NSIDE)(E, B, spec), "EB")

    # TE
    g = f_TE(px=+1) / (hCT(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    spec = {"hCT": ocltt, "hCE": oclee, "CTE": clte}
    compare(compile_te_bA    (terms, LMAX, NSIDE)(T, E, spec),
            compile_te(terms, LMAX, nside=NSIDE)(T, E, spec), "TE")


if __name__ == "__main__":
    main()
