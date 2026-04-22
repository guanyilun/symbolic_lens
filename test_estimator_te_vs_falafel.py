"""Bit-for-bit validation of the symbolic TE estimator against falafel."""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, hCE, f_TE
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, compile_te


LMAX, NSIDE = 200, 256


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_temperature_only, qe_pol_only, filter_alms

    ell = np.arange(LMAX + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clte = 0.5 * np.sqrt(np.abs(cltt * clee))
    nltt = 1.0 * np.ones_like(cltt)
    nlee = 1.5 * np.ones_like(clee)
    ocltt = cltt + nltt
    oclee = clee + nlee

    g_TE = f_TE(px=+1) / (hCT(l1) * hCE(l2))       # Δ = 1 for TE
    terms = strip_gamma(compile_estimator(g_TE))
    print(f"Compiled {len(terms)} TE terms.")
    sym_emit = compile_te(terms, lmax=LMAX, nside=NSIDE)

    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True)

    spectra = {"hCT": ocltt, "hCE": oclee, "CTE": clte}
    phi_sym = sym_emit(T_alm, E_alm, spectra)

    # falafel reference: TE = kfunc(Tte0 + Pte0)
    #   Tte0 = test(xalm('t_e0'))     = qe_spin_temperature_deflection(E·CTE/hCE, fTalm)
    #   Pte0 = pest(xalm('e_t0'), 0, fEalm, 0) = qe_spin_pol_deflection(T·CTE/hCT, 0, E/hCE, 0)
    px = pixelization(nside=NSIDE)
    fTalm = filter_alms(T_alm.copy(), 1.0 / ocltt)
    fEalm = filter_alms(E_alm.copy(), 1.0 / oclee)
    xalm_t_e0 = filter_alms(fEalm.copy(), clte)           # E·CTE/hCE
    xalm_e_t0 = filter_alms(fTalm.copy(), clte)           # T·CTE/hCT

    # Tte0
    dmap_T = pol_prep_test = None
    from falafel.qe import qe_spin_temperature_deflection, qe_spin_pol_deflection, deflection_map_to_phi_curl_alms
    dmap_T = qe_spin_temperature_deflection(px, xalm_t_e0, fTalm, LMAX)
    dmap_P = qe_spin_pol_deflection(px, xalm_e_t0, np.zeros_like(xalm_e_t0),
                                     fEalm, np.zeros_like(xalm_e_t0), LMAX)
    dmap_total = dmap_T + dmap_P
    phi_curl = deflection_map_to_phi_curl_alms(px, dmap_total, LMAX)
    phi_falafel = phi_curl[0] if phi_curl.ndim == 2 else phi_curl

    mask = np.abs(phi_falafel) > 1e-10
    ratios = phi_sym[mask] / phi_falafel[mask]
    print(f"pol_pair_coeff  = {sym_emit.pol_pair_coeff}")
    print(f"temp_pair_coeff = {sym_emit.temp_pair_coeff}")
    print(f"TE symbolic / falafel:")
    print(f"  median |ratio|: {np.median(np.abs(ratios)):.6e}")
    print(f"  median ratio (complex): {np.median(ratios.real):.6e} + {np.median(ratios.imag):.6e}j")
    print(f"  std/mean of |ratio|:    {np.std(np.abs(ratios))/np.mean(np.abs(ratios)):.3e}")


if __name__ == "__main__":
    main()
