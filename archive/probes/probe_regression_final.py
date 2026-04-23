"""Final regression: symbolic compile_native vs hand-coded compile_{xy}_native
for all 6 lensing estimators + ROT + SRC, after baking the int-truncation
ladder rule into compile_native."""
import numpy as np
import healpy as hp

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import (
    hCT, hCE, hCB, f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    compile_tt_native, compile_ee_native, compile_bb_native,
    compile_tb_native, compile_eb_native, compile_te_native,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
ocltt = cltt + 1; oclee = clee + 1.5; oclbb = clbb + 1.5
spec = {"hCT": ocltt, "hCE": oclee, "hCB": oclbb,
        "CT": cltt, "CE": clee, "CB": clbb,
        "CTE": 0.5 * np.sqrt(cltt * clee)}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
E = hp.synalm(oclee, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)


def summarize(label, gen, ref):
    gc = hp.alm2cl(gen); rc = hp.alm2cl(ref); xc = hp.alm2cl(gen, ref)
    valid = (gc > 0) & (rc > 0)
    ratios = xc[valid] / rc[valid]  # signed
    mean = np.mean(ratios)
    sm = np.std(ratios) / np.abs(mean) if np.abs(mean) > 0 else float('nan')
    corr = np.corrcoef(gen.view(np.float64), ref.view(np.float64))[0, 1]
    print(f"  {label:6s}: signed_slope = {mean:+.5e}   std/mean = {sm:.3e}   xcorr = {corr:+.6f}")


print("=" * 70)
print("Final regression: symbolic compile_native vs hand-coded compile_*_native")
print("=" * 70)

# TT
g = f_TT() / (hCT(l1) * hCT(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)[+1])
ref = np.asarray(compile_tt_native(terms, LMAX, px=px)(T, spec))
summarize("TT", gen, ref)

# TE
g = f_TE() / (hCT(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1])
ref = np.asarray(compile_te_native(terms, LMAX, px=px)(T, E, spec))
summarize("TE", gen, ref)

# TB
g = f_TB() / (hCT(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(scalar_pair(T), pol_B_pair(B), spec)[+1])
ref = np.asarray(compile_tb_native(terms, LMAX, px=px)(T, B, spec))
summarize("TB", gen, ref)

# EE
g = f_EE() / (hCE(l1) * hCE(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1])
ref = np.asarray(compile_ee_native(terms, LMAX, px=px)(E, spec))
summarize("EE", gen, ref)

# BB
g = f_BB() / (hCB(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(pol_B_pair(B), pol_B_pair(B), spec)[+1])
ref = np.asarray(compile_bb_native(terms, LMAX, px=px)(B, spec))
summarize("BB", gen, ref)

# EB
g = f_EB() / (hCE(l1) * hCB(l2))
terms = strip_gamma(compile_estimator(g))
gen = np.asarray(compile_native(terms, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1])
ref = np.asarray(compile_eb_native(terms, LMAX, px=px)(E, B, spec))
summarize("EB", gen, ref)
