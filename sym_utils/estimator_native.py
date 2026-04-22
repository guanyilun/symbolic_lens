"""
estimator_native.py — SHT backend for EstimatorTerm lists, without a
falafel runtime dependency.

The SHT primitives (rot2d / irot2d / rot2dalm and the ``Pixelization``
wrapper) are inlined directly from falafel.qe (BSD-licensed) so the
sign/phase conventions are identical to the falafel-delegating
backend A and thus to pytempura's established QE conventions.  The
only thing this module does NOT require at runtime is ``import falafel``.

Use this backend when:
  * You want to ship code to a site that doesn't have falafel installed.
  * You are building an estimator for new physics (rotation, patchy τ,
    source, custom W) that falafel doesn't have a hand-coded primitive
    for — the recipe compilation is fully symbolic, the emitter here
    just executes it.

Regression: the TT/EE/BB/TB/EB/TE validation scripts
``test_estimator_*_vs_falafel.py`` can all be repointed at this
backend and should match at machine precision.
"""
from __future__ import annotations
import numpy as np

from .atom import AtomicFactor, FuncApp, Pow, Var, Mul
from .estimator import EstimatorTerm
from .estimator_backend import strip_gamma, _eval_atom


# ======================================================================
# Inlined SHT bookkeeping — exact copy of falafel.qe conventions.
# ======================================================================

def _rot2d(fmap):
    """Real pair (f0, f1) → complex ±|s| pair  (f0 + i f1, f0 - i f1)."""
    return np.stack((fmap[0] + fmap[1] * 1j, fmap[0] - fmap[1] * 1j))


def _irot2d(fmap, spin):
    """Complex ±|s| alm pair → real alm pair for hp.alm2map_spin."""
    return -np.stack((
        (fmap[0] + ((-1) ** spin) * fmap[1]) / 2.0,
        (fmap[0] - ((-1) ** spin) * fmap[1]) / 2.0 / 1j,
    ))


class Pixelization:
    """Minimal inlined equivalent of ``falafel.qe.pixelization``.

    HEALPix only for now; CAR support is straightforward but not wired.
    """
    def __init__(self, nside, dtype=np.float32, iter=0):
        import healpy as hp
        self.nside = nside
        self.dtype = dtype
        self.iter = iter
        self.hpix = True

    def alm2map(self, alm, spin, ncomp, mlmax):
        import healpy as hp
        if spin != 0:
            return hp.alm2map_spin(alm, nside=self.nside, spin=spin, lmax=mlmax)
        return hp.alm2map(alm.astype(np.complex128), nside=self.nside, pol=False)[None]

    def alm2map_spin(self, alm, spin_alm, spin_transform, ncomp, mlmax):
        import healpy as hp
        ap_am = _irot2d(alm, spin=spin_alm)
        res = hp.alm2map_spin(ap_am.astype(np.complex128), nside=self.nside,
                              spin=abs(spin_transform), lmax=mlmax)
        return _rot2d(res)

    def map2alm(self, imap, lmax):
        import healpy as hp
        return hp.map2alm(imap.astype(np.float64), lmax=lmax, iter=self.iter)

    def map2alm_spin(self, imap, lmax, spin_alm, spin_transform):
        import healpy as hp
        dmap = -_irot2d(np.stack((imap, imap.conj())), spin=spin_alm).real
        return hp.map2alm_spin(dmap.astype(np.float64), lmax=lmax,
                               spin=spin_transform)


# =================================================================
# Helpers mirroring falafel.qe (gradient_spin, deflection→phi).
# =================================================================

def _gradient_spin(px, alm, mlmax, spin):
    """Inlined equivalent of falafel.qe.gradient_spin (uses pixell.curvedsky.almxfl
    for the alm multiplication so convention matches exactly)."""
    from pixell import curvedsky as cs
    ells = np.arange(0, mlmax)
    # NOTE: falafel.qe.gradient_spin initializes fl via ``ells * 0`` which is
    # an INTEGER array (ells is from np.arange without dtype).  Subsequent
    # in-place float assignments get silently truncated.  We reproduce that
    # behavior exactly so our output matches falafel bit-for-bit.  If/when
    # falafel fixes this, flip these to ``np.zeros_like(ells, dtype=float)``.
    if spin == 0:
        fl = np.sqrt(ells * (ells + 1.0))   # already float because of the 1.0
        spin_out, comp, sign = 1, 0, 1
    elif spin == -2:
        fl = ells * 0                       # integer zeros, matches falafel
        fl[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
        spin_out, comp, sign = -1, 1, 1
    elif spin == 2:
        fl = ells * 0                       # integer zeros, matches falafel
        fl[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
        spin_out, comp, sign = 3, 0, -1
    else:
        raise ValueError(f"unsupported spin {spin}")
    fl[ells < 2] = 0
    alm_arr = np.asarray(alm)
    salms = cs.almxfl(alm_arr.copy(), fl)
    return sign * px.alm2map_spin(salms, spin, spin_out, ncomp=2, mlmax=mlmax)[comp]


def _deflection_to_phi_curl(px, dmap, mlmax):
    """Inlined falafel.qe.deflection_map_to_phi_curl_alms."""
    from pixell import curvedsky as cs
    res = px.map2alm_spin(dmap, lmax=mlmax, spin_alm=0, spin_transform=1)
    ells = np.arange(0, mlmax)
    fl = np.sqrt(ells * (ells + 1.0))
    return cs.almxfl(np.asarray(res), fl)


def _qe_spin_temp_defl(px, Xalm, Yalm, mlmax):
    grad = _gradient_spin(px, np.stack((Xalm, Xalm)), mlmax, spin=0)
    ymap = px.alm2map(Yalm, spin=0, ncomp=1, mlmax=mlmax)[0]
    return -grad * ymap


def _qe_spin_pol_defl(px, X_E, X_B, Y_E, Y_B, mlmax):
    def pol_alms(E, B): return np.stack((E + 1j * B, E - 1j * B))
    palms = pol_alms(X_E, X_B)
    g_p2 = _gradient_spin(px, palms, mlmax, spin=+2)
    g_m2 = _gradient_spin(px, palms, mlmax, spin=-2)
    ymap = _rot2d(px.alm2map(np.stack((Y_E, Y_B)), spin=2, ncomp=2, mlmax=mlmax))
    prod = -g_m2 * ymap[0] - g_p2 * ymap[1]
    return prod / 2


def qe_temperature_only(px, Xalm, Yalm, mlmax):
    """Public inlined version: phi_curl alms."""
    return _deflection_to_phi_curl(px, _qe_spin_temp_defl(px, Xalm, Yalm, mlmax), mlmax)


def qe_pol_only(px, X_E, X_B, Y_E, Y_B, mlmax):
    return _deflection_to_phi_curl(px, _qe_spin_pol_defl(px, X_E, X_B, Y_E, Y_B, mlmax), mlmax)


# =====================================================================
# Compile helpers — reuse the milestone-2 emitters but swap the falafel
# primitives for the inlined equivalents above.
# =====================================================================

def compile_tt_native(terms, lmax, nside=2048):
    """Native TT emitter — same logic as estimator_backend.compile_tt but
    without falafel at runtime."""
    import healpy as hp
    import math

    xgrad_terms = [t for t in terms if t.spin_Y == 0 and abs(t.spin_X) == 1]
    ref = xgrad_terms[0]
    pair_coeff = 2 * complex(ref.coeff) * (-1) * 2 * math.sqrt(4 * math.pi)

    px = Pixelization(nside=nside)

    def compiled(T_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        x_fl = _eval_atom(ref.X_filter, ell, spectra).real
        y_fl = _eval_atom(ref.Y_filter, ell, spectra).real
        L_fl = _eval_atom(ref.L_factor, ell, spectra).real

        grad_l = np.sqrt(ell * (ell + 1))
        grad_L = np.sqrt(ell * (ell + 1))
        x_response_fl = np.where(grad_l > 0, x_fl / grad_l, 0.0)
        L_residual_fl = np.where(grad_L > 0, L_fl / grad_L, 0.0)

        X_resp = hp.almxfl(T_alm, x_response_fl)
        Y_iv   = hp.almxfl(T_alm, y_fl)

        phi_curl = qe_temperature_only(px, X_resp, Y_iv, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        phi = hp.almxfl(phi, L_residual_fl)
        return pair_coeff * phi

    compiled.pair_coeff = pair_coeff
    return compiled


def compile_pol_same_field_native(terms, lmax, nside, field):
    """Native EE/BB emitter."""
    import healpy as hp
    import math
    from .estimator_backend import _extract_pol_response
    assert field in ('E', 'B')

    X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")
    pair_coeff = 2 * complex(coeff_ref) * (-1) * 2 * math.sqrt(4*math.pi) * 2

    px = Pixelization(nside=nside)

    def compiled(alm, spectra):
        alm = np.asarray(alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        x = _eval_atom(X_resp, ell, spectra).real
        y = _eval_atom(Y_resp, ell, spectra).real
        L = _eval_atom(L_atom, ell, spectra).real
        grad_L = np.sqrt(ell * (ell + 1))
        L_res = np.where(grad_L > 0, L / grad_L, 0.0)
        X_resp_alm = hp.almxfl(alm, x)
        Y_resp_alm = hp.almxfl(alm, y)
        zero = np.zeros_like(alm)
        if field == 'E':
            phi_curl = qe_pol_only(px, X_resp_alm, zero, Y_resp_alm, zero, lmax)
        else:
            phi_curl = qe_pol_only(px, zero, X_resp_alm, zero, Y_resp_alm, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        return pair_coeff * hp.almxfl(phi, L_res)

    compiled.pair_coeff = pair_coeff
    return compiled


def compile_ee_native(terms, lmax, nside=2048):
    return compile_pol_same_field_native(terms, lmax, nside, 'E')


def compile_bb_native(terms, lmax, nside=2048):
    return compile_pol_same_field_native(terms, lmax, nside, 'B')
