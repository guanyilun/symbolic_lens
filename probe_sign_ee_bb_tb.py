"""Isolate the global sign issue: does a uniform -1 on EE/BB/TB bring them closer?"""
import numpy as np
import healpy as hp

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, hCB, f_EE, f_BB, f_TB, f_EB
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization, compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    compile_ee_native, compile_eb_native, compile_bb_native, compile_tb_native,
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

def analyze(label, gen, ref):
    # Per-L signed ratio gen/ref (using cross / ref_cl)
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n--- {label} ---")
    print(f"{'L':>4s} {'ref_cl':>12s} {'gen_cl':>12s} {'gen/ref':>12s} {'signed':>12s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            r2 = gc[L] / rc[L]
            signed = xc[L] / rc[L]  # signed ratio
            print(f"{L:>4d} {rc[L]:>12.4e} {gc[L]:>12.4e} {np.sqrt(r2):>12.5e} {np.sqrt(abs(signed))*np.sign(signed):>12.5e}")

def pair(g, t, in_x, in_y, ref_fn, ref_args):
    gen = np.asarray(compile_native(t, LMAX, px=px)(in_x, in_y, spec)[+1])
    ref = np.asarray(ref_fn(t, LMAX, px=px)(*ref_args))
    return gen, ref

g = f_EE() / (hCE(l1) * hCE(l2)); t = strip_gamma(compile_estimator(g))
gen, ref = pair(g, t, pol_E_pair(E), pol_E_pair(E), compile_ee_native, (E, spec))
analyze("EE", gen, ref)

g = f_BB() / (hCB(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
gen, ref = pair(g, t, pol_B_pair(B), pol_B_pair(B), compile_bb_native, (B, spec))
analyze("BB", gen, ref)

g = f_TB() / (hCT(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
gen, ref = pair(g, t, scalar_pair(T), pol_B_pair(B), compile_tb_native, (T, B, spec))
analyze("TB", gen, ref)

g = f_EB() / (hCE(l1) * hCB(l2)); t = strip_gamma(compile_estimator(g))
gen, ref = pair(g, t, pol_E_pair(E), pol_B_pair(B), compile_eb_native, (E, B, spec))
analyze("EB", gen, ref)
