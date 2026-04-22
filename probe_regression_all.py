"""Full regression: all 6 lensing estimators, generic emitter vs hand-coded."""
import numpy as np
import healpy as hp

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, hCB, f_TT, f_TE, f_TB, f_EE, f_EB, f_BB
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    compile_tt_native, compile_te_native, compile_tb_native,
    compile_ee_native, compile_eb_native, compile_bb_native,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
clte = 0.5 * np.sqrt(cltt * clee)
ocltt = cltt + 1; oclee = clee + 1.5; oclbb = clbb + 1.5
spec = {"hCT": ocltt, "hCE": oclee, "hCB": oclbb,
        "CTE": clte, "CT": cltt, "CE": clee, "CB": clbb}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
E = hp.synalm(oclee, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

def summary(label, gen, ref):
    mask = np.abs(ref) > 1e-6 * np.max(np.abs(ref))
    r = gen[mask] / ref[mask]
    med = np.median(np.abs(r))
    rel = np.std(np.abs(r)) / np.mean(np.abs(r))
    # Cross-correlation in alm space
    ref_cl = hp.alm2cl(ref); gen_cl = hp.alm2cl(gen); x_cl = hp.alm2cl(gen, ref)
    mask_L = (ref_cl > 0) & (gen_cl > 0)
    xcorr = np.mean(x_cl[mask_L] / np.sqrt(ref_cl[mask_L] * gen_cl[mask_L]))
    print(f"{label:4s}  median|r|={med:.4e}  std/mean={rel:.4e}  xcorr={xcorr:.5f}")

g = f_TT() / (hCT(l1) * hCT(l2)); t = strip_gamma(compile_estimator(g))
summary("TT",
        np.asarray(compile_native(t, LMAX, px=px)(scalar_pair(T), scalar_pair(T), spec)[+1]),
        np.asarray(compile_tt_native(t, LMAX, px=px)(T, spec)))

g = f_EE() / (hCE(l1) * hCE(l2)); t = strip_gamma(compile_estimator(g))
summary("EE",
        np.asarray(compile_native(t, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1]),
        np.asarray(compile_ee_native(t, LMAX, px=px)(E, spec)))

g = f_BB() / (hCB(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
summary("BB",
        np.asarray(compile_native(t, LMAX, px=px)(pol_B_pair(B), pol_B_pair(B), spec)[+1]),
        np.asarray(compile_bb_native(t, LMAX, px=px)(B, spec)))

g = f_TE() / (hCT(l1) * hCE(l2)); t = strip_gamma(compile_estimator(g))
summary("TE",
        np.asarray(compile_native(t, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1]),
        np.asarray(compile_te_native(t, LMAX, px=px)(T, E, spec)))

g = f_TB() / (hCT(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
summary("TB",
        np.asarray(compile_native(t, LMAX, px=px)(scalar_pair(T), pol_B_pair(B), spec)[+1]),
        np.asarray(compile_tb_native(t, LMAX, px=px)(T, B, spec)))

g = f_EB() / (hCE(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
summary("EB",
        np.asarray(compile_native(t, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1]),
        np.asarray(compile_eb_native(t, LMAX, px=px)(E, B, spec)))
