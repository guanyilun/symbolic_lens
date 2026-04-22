"""
CAR regression: native SHT backend vs the falafel-delegating backend A,
on CAR geometry, for all 6 lensing estimators.

Mirror of ``test_estimator_native_vs_backendA.py`` but with:
  - ``Pixelization(shape=, wcs=)`` for native (CAR)
  - ``falafel.qe.pixelization(shape=, wcs=)`` for backend A (CAR)

Since compile_tt / compile_ee / etc. in backend A currently hardcode a
HEALPix pixelization, we need to override their internal px with a CAR
one.  This is done via monkey-patching: we build the emitter with
``nside=1``, then replace ``px`` with a CAR instance before calling.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import (
    hCT, hCE, hCB, f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
)
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import (
    Pixelization,
    compile_tt_native, compile_ee_native, compile_bb_native,
    compile_tb_native, compile_eb_native, compile_te_native,
)

LMAX = 200


def compare(phi_ref, phi_test, label):
    mask = np.abs(phi_ref) > 1e-8
    ratios = phi_test[mask] / phi_ref[mask]
    median = np.median(np.abs(ratios))
    std_rel = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))
    status = "MATCH" if (std_rel < 1e-4 and abs(median - 1) < 1e-4) else "MISMATCH"
    print(f"  {label:4s}  median={median:.6e}  std/mean={std_rel:.3e}  {status}")


def _native_emit(builder, terms, lmax, shape, wcs):
    return builder(terms, lmax, shape=shape, wcs=wcs)


def _falafel_emit(backend_builder, terms, lmax, shape, wcs):
    """Backend A's emitters hardcode HEALPix inside; we construct one
    with a dummy nside and then swap its internal px for a CAR one.

    This requires each emitter closure to hold its ``px`` in a known
    attribute or capture we can replace.  For the backend A closures
    that use ``from falafel.qe import pixelization``, we instead
    reconstruct the emitter by running its path with a CAR
    pixelization at module level."""
    from falafel.qe import pixelization as falafel_pix
    # Dirty but effective: the backend A emitters construct their own
    # ``px`` from ``nside``.  Rather than monkey-patch, we run the
    # identical compile path with a CAR px by re-importing and calling
    # falafel's primitives directly — delegated below per-estimator.
    raise NotImplementedError("use per-estimator helpers below")


def main():
    from pixell import enmap
    from falafel.qe import (
        pixelization as falafel_pix, qe_temperature_only, qe_pol_only,
        filter_alms,
    )
    import healpy as hp
    import math

    res = np.deg2rad(12.0 / 60.0)
    shape, wcs = enmap.fullsky_geometry(res=res)

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb = 5.0 / (ell + 10) ** 2 + 1e-4
    clte = 0.5 * np.sqrt(cltt * clee)
    nltt = np.ones_like(cltt)
    nlee = 1.5 * np.ones_like(clee)
    nlbb = 1.5 * np.ones_like(clbb)
    ocltt, oclee, oclbb = cltt + nltt, clee + nlee, clbb + nlbb

    np.random.seed(42)
    T = hp.synalm(ocltt, lmax=LMAX, new=True)
    E = hp.synalm(oclee, lmax=LMAX, new=True)
    B = hp.synalm(oclbb, lmax=LMAX, new=True)

    px_fal = falafel_pix(shape=shape, wcs=wcs)
    px_nat = Pixelization(shape=shape, wcs=wcs)

    print(f"Geometry: CAR full-sky, shape={tuple(int(x) for x in shape)}, lmax={LMAX}")
    print("Native CAR vs Backend-A CAR (hand-rolled for shape/wcs):")

    # --- TT ---
    # backend-A pipeline inlined for CAR (same pair_coeff as compile_tt):
    g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = strip_gamma(compile_estimator(g))
    xgrad = next(t for t in terms if t.spin_Y == 0 and abs(t.spin_X) == 1)
    pair_coeff = 2 * complex(xgrad.coeff) * (-1) * 2 * math.sqrt(4 * math.pi)
    # falafel CAR:
    X_resp = filter_alms(T.copy(), cltt / ocltt)
    Y_iv   = filter_alms(T.copy(), 1.0 / ocltt)
    phi_curl = qe_temperature_only(px_fal, X_resp, Y_iv, LMAX)
    phi_fal = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
    phi_fal = pair_coeff * phi_fal
    phi_nat = compile_tt_native(terms, LMAX, shape=shape, wcs=wcs)(
                T, {"hCT": ocltt, "CT": cltt})
    compare(phi_fal, phi_nat, "TT")

    # --- EE ---
    from sym_utils.estimator_backend import _extract_pol_response
    g = f_EE(px=+1) / (sympify(2) * hCE(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    X_resp_atom, Y_resp_atom, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")
    pair_coeff_EE = 2 * complex(coeff_ref) * (-1) * 2 * math.sqrt(4 * math.pi) * 2
    x_fl = np.where(np.arange(LMAX+1) >= 2, clee / oclee, 0)
    X_fal = filter_alms(E.copy(), x_fl)
    Y_fal = filter_alms(E.copy(), 1.0 / oclee)
    zero = np.zeros_like(E)
    phi_curl = qe_pol_only(px_fal, X_fal, zero, Y_fal, zero, LMAX)
    phi_fal = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
    phi_fal = pair_coeff_EE * phi_fal
    phi_nat = compile_ee_native(terms, LMAX, shape=shape, wcs=wcs)(
                E, {"hCE": oclee, "CE": clee})
    compare(phi_fal, phi_nat, "EE")

    # --- BB ---
    g = f_BB(px=+1) / (sympify(2) * hCB(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    _, _, _, coeff_ref = _extract_pol_response(terms, "X", "l1")
    pair_coeff_BB = 2 * complex(coeff_ref) * (-1) * 2 * math.sqrt(4 * math.pi) * 2
    X_fal = filter_alms(B.copy(), clbb / oclbb)
    Y_fal = filter_alms(B.copy(), 1.0 / oclbb)
    phi_curl = qe_pol_only(px_fal, zero, X_fal, zero, Y_fal, LMAX)
    phi_fal = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
    phi_fal = pair_coeff_BB * phi_fal
    phi_nat = compile_bb_native(terms, LMAX, shape=shape, wcs=wcs)(
                B, {"hCB": oclbb, "CB": clbb})
    compare(phi_fal, phi_nat, "BB")

    # --- TB ---
    g = f_TB(px=+1) / (hCT(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    _, _, _, coeff_ref = _extract_pol_response(terms, "X", "l1")
    pair_coeff_TB = complex(coeff_ref) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2
    fT = filter_alms(T.copy(), 1.0 / ocltt)
    fB = filter_alms(B.copy(), 1.0 / oclbb)
    X_pseudo_E = filter_alms(fT.copy(), clte)
    phi_curl = qe_pol_only(px_fal, X_pseudo_E, zero, zero, fB, LMAX)
    phi_fal = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
    phi_fal = pair_coeff_TB * phi_fal
    phi_nat = compile_tb_native(terms, LMAX, shape=shape, wcs=wcs)(
                T, B, {"hCT": ocltt, "hCB": oclbb, "CTE": clte})
    compare(phi_fal, phi_nat, "TB")

    # --- EB (assuming C_B primordial = 0) ---
    g = f_EB(px=+1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g))
    from sym_utils.estimator_backend import _extract_pol_response_from
    xgrad = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    ref = next(t for t in xgrad if abs(t.spin_X) == 1)
    X_resp_atom = _extract_pol_response_from(ref, "X")
    pair_coeff_EB = complex(ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2
    X_E = filter_alms(E.copy(), clee / oclee)
    Y_B = filter_alms(B.copy(), 1.0 / oclbb)
    phi_curl = qe_pol_only(px_fal, X_E, zero, zero, Y_B, LMAX)
    phi_fal = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
    phi_fal = pair_coeff_EB * phi_fal
    phi_nat = compile_eb_native(terms, LMAX, shape=shape, wcs=wcs)(
                E, B, {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)})
    compare(phi_fal, phi_nat, "EB")

    # --- TE (pol half + temp half) ---
    from falafel.qe import qe_spin_temperature_deflection, qe_spin_pol_deflection, deflection_map_to_phi_curl_alms
    g = f_TE(px=+1) / (hCT(l1) * hCE(l2))
    terms = strip_gamma(compile_estimator(g))
    pol_terms = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    temp_terms = [t for t in terms if t.spin_X == 0 and abs(t.spin_Y) == 1]
    pol_ref = next(t for t in pol_terms if abs(t.spin_X) == 1)
    temp_ref = next(t for t in temp_terms if abs(t.spin_Y) == 1)
    pol_pair_coeff = complex(pol_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * 2
    temp_pair_coeff = complex(temp_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2
    fT = filter_alms(T.copy(), 1.0 / ocltt)
    fE = filter_alms(E.copy(), 1.0 / oclee)
    xalm_t_e0 = filter_alms(fE.copy(), clte)
    xalm_e_t0 = filter_alms(fT.copy(), clte)
    dmap_T = qe_spin_temperature_deflection(px_fal, xalm_t_e0, fT, LMAX)
    dmap_P = qe_spin_pol_deflection(px_fal, xalm_e_t0,
                                     np.zeros_like(xalm_e_t0),
                                     fE, np.zeros_like(xalm_e_t0), LMAX)
    phi_fal_T = deflection_map_to_phi_curl_alms(px_fal, dmap_T, LMAX)[0] * temp_pair_coeff
    phi_fal_P = deflection_map_to_phi_curl_alms(px_fal, dmap_P, LMAX)[0] * pol_pair_coeff
    phi_fal = phi_fal_T + phi_fal_P
    phi_nat = compile_te_native(terms, LMAX, shape=shape, wcs=wcs)(
                T, E, {"hCT": ocltt, "hCE": oclee, "CTE": clte})
    compare(phi_fal, phi_nat, "TE")


if __name__ == "__main__":
    main()
