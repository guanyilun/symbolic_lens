"""Empirically test all 4 sign-flip combinations to find the right
convention relating our symbolic sign(spin_X), sign(spin_Y) to the
M_+|s| vs M_-|s| component of the alm2map_spin output."""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, hCB, f_TT, f_EE, f_BB, f_rot_EB, f_ampl_TT
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    compile_tt_native, compile_ee_native, compile_bb_native,
    compile_rot_eb_native, compile_source_tt_native,
    Pixelization, fuse_spin_pairs, _eval_atom, _alm_to_signed_pair,
    scalar_pair, pol_E_pair, pol_B_pair,
)
import healpy as hp

LMAX, NSIDE = 200, 256


def generic_emit(terms, lmax, px, flip_X, flip_Y):
    """Generic emitter with configurable sign convention."""
    fused = fuse_spin_pairs(terms)
    def run(X_input, Y_input, spectra):
        X_pair, X_spin_alm = X_input
        Y_pair, Y_spin_alm = Y_input
        ell = np.arange(lmax + 1, dtype=float)
        X_pair = np.asarray(X_pair, dtype=np.complex128)
        Y_pair = np.asarray(Y_pair, dtype=np.complex128)
        outputs = {}
        for plan in fused:
            x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
            y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
            L_fl = _eval_atom(plan.L_factor, ell, spectra).real.astype(np.float64)
            Xf = np.stack([hp.almxfl(X_pair[0], x_fl), hp.almxfl(X_pair[1], x_fl)])
            Yf = np.stack([hp.almxfl(Y_pair[0], y_fl), hp.almxfl(Y_pair[1], y_fl)])
            X_maps = _alm_to_signed_pair(px, Xf, X_spin_alm, plan.abs_spin_X, lmax)
            Y_maps = _alm_to_signed_pair(px, Yf, Y_spin_alm, plan.abs_spin_Y, lmax)
            for (sX, sY, sL), coeff in plan.coeffs.items():
                sX_eff = -sX if flip_X else sX
                sY_eff = -sY if flip_Y else sY
                Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
                My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
                prod = complex(coeff) * Mx * My
                if plan.abs_spin_L == 0:
                    m = prod.real
                    if px.hpix:
                        alm = px.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)
                    else:
                        from pixell import enmap
                        alm = px.map2alm(enmap.enmap(m, px.wcs), lmax=lmax)
                    alm = hp.almxfl(alm, L_fl)
                    outputs[0] = alm if 0 not in outputs else outputs[0] + alm
                else:
                    pair = px.map2alm_spin(prod, lmax=lmax, spin_alm=0, spin_transform=plan.abs_spin_L)
                    pick = pair[0] if sL >= 0 else pair[1]
                    pick = hp.almxfl(pick, L_fl)
                    outputs[sL] = pick if sL not in outputs else outputs[sL] + pick
        return outputs
    return run


def ratio_stats(alm_ref, alm_gen):
    alm_ref = np.asarray(alm_ref); alm_gen = np.asarray(alm_gen)
    mask = np.abs(alm_ref) > 1e-8
    r = alm_gen[mask] / alm_ref[mask]
    return np.median(np.abs(r)), np.std(np.abs(r))/np.mean(np.abs(r))


def main():
    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    nltt, nlee, nlbb = np.ones_like(cltt), 1.5*np.ones_like(clee), 1.5*np.ones_like(clbb)
    ocltt, oclee, oclbb = cltt+nltt, clee+nlee, clbb+nlbb
    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    E = hp.synalm(oclee, lmax=LMAX, new=True)
    B = hp.synalm(oclbb, lmax=LMAX, new=True)
    px = Pixelization(nside=NSIDE)

    cases = []
    # TT
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    cases.append(("TT", terms, (scalar_pair(T), scalar_pair(T)),
                  compile_tt_native(terms, LMAX, px=px)(T, {"hCT": ocltt, "CT": cltt}), +1))
    # EE
    g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    cases.append(("EE", terms, (pol_E_pair(E), pol_E_pair(E)),
                  compile_ee_native(terms, LMAX, px=px)(E, {"hCE": oclee, "CE": clee}), +1))
    # BB
    g = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    cases.append(("BB", terms, (pol_B_pair(B), pol_B_pair(B)),
                  compile_bb_native(terms, LMAX, px=px)(B, {"hCB": oclbb, "CB": clbb}), +1))
    # ROT-EB
    g = f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    cases.append(("ROT", terms, (pol_E_pair(E), pol_B_pair(B)),
                  compile_rot_eb_native(terms, LMAX, px=px)(E, B, {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)}), 0))
    # SRC
    g = f_ampl_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    cases.append(("SRC", terms, (scalar_pair(T), scalar_pair(T)),
                  compile_source_tt_native(terms, LMAX, px=px)(T, {"hCT": ocltt, "CT": cltt}), 0))

    for flipX in (False, True):
        for flipY in (False, True):
            label = f"flip_X={flipX} flip_Y={flipY}"
            print(f"\n{label}:")
            for name, terms, (X_in, Y_in), ref, ch in cases:
                emit = generic_emit(terms, LMAX, px, flipX, flipY)
                spec_kwargs = {"ROT": {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)},
                               "TT": {"hCT": ocltt, "CT": cltt},
                               "EE": {"hCE": oclee, "CE": clee},
                               "BB": {"hCB": oclbb, "CB": clbb},
                               "SRC": {"hCT": ocltt, "CT": cltt}}[name]
                gen = emit(X_in, Y_in, spec_kwargs)
                if ch in gen:
                    med, stdm = ratio_stats(ref, gen[ch])
                    print(f"  {name:4s} ch={ch:+d}   median={med:.4e}  std/mean={stdm:.3e}")


if __name__ == "__main__":
    main()
