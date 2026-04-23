"""Test the NP-sign rule on all lensing pol estimators.

Rule: for each leg, if |s_leg| == 3 apply -1, elif |s_leg| == 1 apply +1,
otherwise +1.  Net plan factor = X_sign * Y_sign.

Verify by applying to current generic emitter post-hoc and checking
whether the ratio to hand-coded becomes a constant (std/mean → 0).
"""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import (
    hCT, hCE, hCB, CT, CE, CB, CTE,
    f_TT, f_EE, f_BB, f_TB, f_EB, f_TE,
)
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs,
    compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    compile_tt_native, compile_ee_native, compile_bb_native,
    compile_tb_native, compile_eb_native, compile_te_native,
)

LMAX, NSIDE = 200, 256


def np_sign_per_leg(abs_spin):
    """Return the Newman-Penrose ladder sign for a pol-ladder leg."""
    if abs_spin == 3:
        return -1
    return +1


def run(label, gen, ref):
    gen = np.asarray(gen); ref = np.asarray(ref)
    mask = np.abs(ref) > 1e-8
    r = gen[mask] / ref[mask]
    med = np.median(np.abs(r))
    s = np.std(np.abs(r)) / (np.mean(np.abs(r)) + 1e-30)
    print(f"  {label:16s}  median={med:.4e}  std/mean={s:.3e}  "
          f"{'MATCH' if s < 1e-10 else 'STRUCTURAL DIFF'}")


def main():
    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    clte = 0.5 * np.sqrt(cltt * clee)
    nltt = np.ones_like(cltt); nlee = 1.5 * np.ones_like(clee); nlbb = 1.5 * np.ones_like(clbb)
    ocltt = cltt + nltt; oclee = clee + nlee; oclbb = clbb + nlbb

    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    E = hp.synalm(oclee, lmax=LMAX, new=True)
    B = hp.synalm(oclbb, lmax=LMAX, new=True)
    px = Pixelization(nside=NSIDE)

    cases = [
        ("EE (phi)", f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2)),
         {"hCE": oclee, "CE": clee},
         lambda t: compile_ee_native(t, LMAX, px=px)(E, spec),
         lambda t: compile_native(t, LMAX, px=px)(pol_E_pair(E), pol_E_pair(E), spec)[+1]),
        ("BB (phi)", f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2)),
         {"hCB": oclbb, "CB": clbb},
         lambda t: compile_bb_native(t, LMAX, px=px)(B, spec),
         lambda t: compile_native(t, LMAX, px=px)(pol_B_pair(B), pol_B_pair(B), spec)[+1]),
        ("TB (phi)", f_TB(px=+1) / (hCT(l1) * hCB(l2)),
         {"hCT": ocltt, "hCB": oclbb, "CTE": clte, "CT": cltt, "CB": clbb},
         lambda t: compile_tb_native(t, LMAX, px=px)(T, B, spec),
         lambda t: compile_native(t, LMAX, px=px)(scalar_pair(T), pol_B_pair(B), spec)[+1]),
        ("EB (phi)", f_EB(px=+1) / (hCE(l1) * hCB(l2)),
         {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": clbb},
         lambda t: compile_eb_native(t, LMAX, px=px)(E, B, spec),
         lambda t: compile_native(t, LMAX, px=px)(pol_E_pair(E), pol_B_pair(B), spec)[+1]),
        ("TE (phi)", f_TE(px=+1) / (hCT(l1) * hCE(l2)),
         {"hCT": ocltt, "hCE": oclee, "CTE": clte, "CT": cltt, "CE": clee},
         lambda t: compile_te_native(t, LMAX, px=px)(T, E, spec),
         lambda t: compile_native(t, LMAX, px=px)(scalar_pair(T), pol_E_pair(E), spec)[+1]),
    ]

    print("CURRENT (no NP-sign rule applied):\n")
    for label, g_sym, spec, ref_fn, gen_fn in cases:
        terms = strip_gamma(compile_estimator(g_sym))
        globals()["spec"] = spec
        ref = ref_fn(terms)
        gen = gen_fn(terms)
        run(label, gen, ref)

    print("\nPlan breakdown (sig → NP factor) for reference:\n")
    for label, g_sym, spec, _, _ in cases:
        terms = strip_gamma(compile_estimator(g_sym))
        plans = fuse_spin_pairs(terms)
        print(f"  {label}:")
        for i, p in enumerate(plans):
            x_s = np_sign_per_leg(p.abs_spin_X)
            y_s = np_sign_per_leg(p.abs_spin_Y)
            print(f"    plan {i}: |sX|={p.abs_spin_X} |sY|={p.abs_spin_Y} |sL|={p.abs_spin_L}  "
                  f"NP_X={x_s}  NP_Y={y_s}  net={x_s*y_s}")


if __name__ == "__main__":
    main()
