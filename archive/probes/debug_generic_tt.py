"""Debug: for TT with flip_X=flip_Y=True, the per-ell ratio should be
exactly -2c ≈ 0.141 (constant).  If not constant, there's a bug beyond
the sign convention."""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, f_TT
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    compile_tt_native, Pixelization, fuse_spin_pairs, _eval_atom,
    _alm_to_signed_pair, scalar_pair,
)
import healpy as hp

LMAX, NSIDE = 200, 256


def generic_with_flip(terms, lmax, px, flip_X=True, flip_Y=True):
    fused = fuse_spin_pairs(terms)
    def run(X_input, Y_input, spectra):
        X_pair, X_spin_alm = X_input
        Y_pair, Y_spin_alm = Y_input
        X_pair = np.asarray(X_pair, dtype=np.complex128)
        Y_pair = np.asarray(Y_pair, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        outputs = {}
        for pi, plan in enumerate(fused):
            x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
            y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
            L_fl = _eval_atom(plan.L_factor, ell, spectra).real.astype(np.float64)
            Xf = np.stack([hp.almxfl(X_pair[0], x_fl), hp.almxfl(X_pair[1], x_fl)])
            Yf = np.stack([hp.almxfl(Y_pair[0], y_fl), hp.almxfl(Y_pair[1], y_fl)])
            X_maps = _alm_to_signed_pair(px, Xf, X_spin_alm, plan.abs_spin_X, lmax)
            Y_maps = _alm_to_signed_pair(px, Yf, Y_spin_alm, plan.abs_spin_Y, lmax)
            print(f"\n  [Fused plan {pi}]  |sX|={plan.abs_spin_X}  |sY|={plan.abs_spin_Y}  |sL|={plan.abs_spin_L}")
            print(f"    coeffs: {plan.coeffs}")
            for (sX, sY, sL), coeff in plan.coeffs.items():
                sX_eff = -sX if flip_X else sX
                sY_eff = -sY if flip_Y else sY
                Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
                My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
                prod = complex(coeff) * Mx * My
                pair = px.map2alm_spin(prod, lmax=lmax, spin_alm=0, spin_transform=plan.abs_spin_L)
                pick = pair[0] if sL >= 0 else pair[1]
                pick = hp.almxfl(pick, L_fl)
                outputs[sL] = pick if sL not in outputs else outputs[sL] + pick
                print(f"    term {(sX,sY,sL)} coeff={coeff:+.4e} → pick=pair[{0 if sL>=0 else 1}]  contribution magnitude at l=10: {np.abs(pick[hp.Alm.getidx(lmax, 10, 5)]):.4e}")
        return outputs
    return run


def main():
    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    nltt = np.ones_like(cltt)
    ocltt = cltt + nltt

    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    px = Pixelization(nside=NSIDE)

    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))

    ref = compile_tt_native(terms, LMAX, px=px)(T, {"hCT": ocltt, "CT": cltt})
    print(f"\nhand-coded ref[0:5]: {ref[:5]}")

    emit = generic_with_flip(terms, LMAX, px, flip_X=True, flip_Y=True)
    out = emit(scalar_pair(T), scalar_pair(T), {"hCT": ocltt, "CT": cltt})
    gen_phi = out[+1]
    print(f"\ngeneric phi[0:5]: {gen_phi[:5]}")

    # per-L ratio using only m=0 entries
    print("\nPer-L ratios (using m=0 alms):")
    for L in [2, 5, 10, 20, 50, 100, 150]:
        idx = hp.Alm.getidx(LMAX, L, 0)
        if abs(ref[idx]) > 1e-8:
            r = gen_phi[idx] / ref[idx]
            print(f"  L={L:3d}  ref={ref[idx]:+.3e}  gen={gen_phi[idx]:+.3e}  ratio={r}")


if __name__ == "__main__":
    main()
