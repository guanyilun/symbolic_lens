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
    """Inlined equivalent of ``falafel.qe.pixelization`` supporting BOTH
    HEALPix and CAR (pixell) geometries.

    CAR is the canonical pixelization for SO / ACT pipelines; use it by
    passing ``shape`` and ``wcs`` (both obtained from
    ``pixell.enmap.fullsky_geometry`` or similar).  HEALPix is kept for
    back-compat by passing ``nside`` instead.

    Conventions match falafel.qe.pixelization exactly so the native
    backend is bit-compatible with falafel on both geometries.
    """
    def __init__(self, shape=None, wcs=None, nside=None,
                 dtype=np.float32, iter=0):
        if shape is not None:
            assert wcs is not None, "CAR pixelization needs both shape and wcs"
            assert nside is None,    "pass either shape+wcs or nside, not both"
            self.hpix = False
            self.shape = shape[-2:]
            self.wcs = wcs
        else:
            assert wcs is None
            assert nside is not None
            self.hpix = True
            self.nside = nside
        self.dtype = dtype
        self.iter = iter

    # ---- alm → map ----

    def alm2map(self, alm, spin, ncomp, mlmax):
        if self.hpix:
            import healpy as hp
            if spin != 0:
                return hp.alm2map_spin(alm, nside=self.nside, spin=spin, lmax=mlmax)
            return hp.alm2map(alm.astype(np.complex128), nside=self.nside,
                              pol=False)[None]
        from pixell import curvedsky as cs, enmap
        omap = enmap.empty((ncomp,) + self.shape, self.wcs, dtype=self.dtype)
        return cs.alm2map(alm, omap, spin=spin)

    def alm2map_spin(self, alm, spin_alm, spin_transform, ncomp, mlmax):
        ap_am = _irot2d(alm, spin=spin_alm)
        if self.hpix:
            import healpy as hp
            res = hp.alm2map_spin(ap_am.astype(np.complex128), nside=self.nside,
                                  spin=abs(spin_transform), lmax=mlmax)
        else:
            from pixell import curvedsky as cs, enmap
            omap = enmap.empty((ncomp,) + self.shape, self.wcs, dtype=self.dtype)
            res = cs.alm2map(ap_am, omap, spin=abs(spin_transform))
        return _rot2d(res)

    # ---- map → alm ----

    def map2alm(self, imap, lmax):
        if self.hpix:
            import healpy as hp
            return hp.map2alm(np.asarray(imap, dtype=np.float64), lmax=lmax, iter=self.iter)
        from pixell import curvedsky as cs
        return cs.map2alm(imap, lmax=lmax)

    def map2alm_spin(self, imap, lmax, spin_alm, spin_transform):
        dmap = -_irot2d(np.stack((imap, imap.conj())), spin=spin_alm).real
        if self.hpix:
            import healpy as hp
            return hp.map2alm_spin(np.asarray(dmap, dtype=np.float64),
                                   lmax=lmax, spin=spin_transform)
        from pixell import curvedsky as cs, enmap
        return cs.map2alm(enmap.enmap(dmap, self.wcs),
                          spin=spin_transform, lmax=lmax)


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


def compile_tb_native(terms, lmax, nside=2048):
    """Native TB emitter. Mirrors estimator_backend.compile_tb."""
    import healpy as hp
    import math
    from .estimator_backend import _extract_pol_response

    X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")
    pair_coeff = complex(coeff_ref) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2

    px = Pixelization(nside=nside)

    def compiled(T_alm, B_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        B_alm = np.asarray(B_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        x = _eval_atom(X_resp, ell, spectra).real
        y = _eval_atom(Y_resp, ell, spectra).real
        L = _eval_atom(L_atom, ell, spectra).real
        grad_L = np.sqrt(ell * (ell + 1))
        L_res = np.where(grad_L > 0, L / grad_L, 0.0)
        X_E = hp.almxfl(T_alm, x)
        Y_B = hp.almxfl(B_alm, y)
        zero = np.zeros_like(T_alm)
        phi_curl = qe_pol_only(px, X_E, zero, zero, Y_B, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        return pair_coeff * hp.almxfl(phi, L_res)

    compiled.pair_coeff = pair_coeff
    return compiled


def compile_eb_native(terms, lmax, nside=2048):
    """Native EB emitter. Mirrors estimator_backend.compile_eb."""
    import healpy as hp
    import math
    from .estimator_backend import _extract_pol_response_from

    xgrad = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    ref = next(t for t in xgrad if abs(t.spin_X) == 1)
    X_resp = _extract_pol_response_from(ref, "X")
    Y_resp = ref.Y_filter
    L_atom = ref.L_factor
    pair_coeff = complex(ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2

    px = Pixelization(nside=nside)

    def compiled(E_alm, B_alm, spectra):
        E_alm = np.asarray(E_alm, dtype=np.complex128)
        B_alm = np.asarray(B_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        x = _eval_atom(X_resp, ell, spectra).real
        y = _eval_atom(Y_resp, ell, spectra).real
        L = _eval_atom(L_atom, ell, spectra).real
        grad_L = np.sqrt(ell * (ell + 1))
        L_res = np.where(grad_L > 0, L / grad_L, 0.0)
        X_E = hp.almxfl(E_alm, x)
        Y_B = hp.almxfl(B_alm, y)
        zero = np.zeros_like(E_alm)
        phi_curl = qe_pol_only(px, X_E, zero, zero, Y_B, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        return pair_coeff * hp.almxfl(phi, L_res)

    compiled.pair_coeff = pair_coeff
    return compiled


def compile_rot_eb_native(terms, lmax, nside=None, shape=None, wcs=None, px=None):
    """Native backend for the CMB rotation (α) EB estimator.

    Pass either ``nside`` (HEALPix) or ``shape`` + ``wcs`` (CAR), or a
    pre-built ``px`` object.  CAR is the canonical SO/ACT pipeline
    geometry and is recommended for production.

    Structurally different from lensing EB:
      - spin-2 SHT on both E and B legs (no gradient/sqrt filters)
      - scalar (spin-0) map2alm for the output α_LM
      - no sqrt(L(L+1)) post-multiplier (L_factor = 1 in the recipe)

    This primitive is NOT present in falafel.qe as a single function,
    though falafel.qe.qe_rot provides an equivalent CAR-only implementation
    we validate against.  The emitter assembles the primitive from the
    symbolic recipe (spin tuples, filters, coefficients); the underlying
    W^{α,+} weight is defined in namikawa.py.
    """
    import healpy as hp
    if px is None:
        if nside is not None:
            px = Pixelization(nside=nside)
        else:
            px = Pixelization(shape=shape, wcs=wcs)

    def compiled(E_alm, B_alm, spectra):
        E_alm = np.asarray(E_alm, dtype=np.complex128)
        B_alm = np.asarray(B_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        # Accumulate the coefficient-weighted complex map in real space.  We
        # can't take Re(prod) per term because the ±spin pair members have
        # conjugate prod's with identical real parts — only their imaginary
        # parts carry the signal that the ±I rotation coefficients extract.
        total_map = None
        for t in terms:
            x_fl = _eval_atom(t.X_filter, ell, spectra).real.astype(np.float64)
            y_fl = _eval_atom(t.Y_filter, ell, spectra).real.astype(np.float64)

            Ef = hp.almxfl(E_alm, x_fl)
            Bf = hp.almxfl(B_alm, y_fl)

            palms_E = np.stack([Ef, Ef])
            xmap_pair = px.alm2map_spin(palms_E, spin_alm=2, spin_transform=2,
                                        ncomp=2, mlmax=lmax)
            x_map = xmap_pair[0] if t.spin_X > 0 else xmap_pair[1]

            palms_B = np.stack([1j * Bf, -1j * Bf])
            ymap_pair = px.alm2map_spin(palms_B, spin_alm=2, spin_transform=2,
                                        ncomp=2, mlmax=lmax)
            y_map = ymap_pair[0] if t.spin_Y > 0 else ymap_pair[1]

            prod = x_map * y_map
            contrib = complex(t.coeff) * prod
            total_map = contrib if total_map is None else total_map + contrib

        # α is a real scalar field — take the real part of the summed map
        # and apply the (constant) L_factor once.
        L_fl = _eval_atom(terms[0].L_factor, ell, spectra).real.astype(np.float64)
        alpha_map = total_map.real
        if px.hpix:
            alpha_alm = px.map2alm(np.asarray(alpha_map, dtype=np.float64), lmax=lmax)
        else:
            from pixell import enmap
            alpha_alm = px.map2alm(enmap.enmap(alpha_map, px.wcs), lmax=lmax)
        alpha_alm = hp.almxfl(alpha_alm, L_fl)
        # Undo the γ residual 1/sqrt(4π) that our symbolic f/g carries
        # explicitly; falafel.qe.qe_rot does not include this factor.
        import math
        return alpha_alm * math.sqrt(4 * math.pi)

    compiled.n_terms = len(terms)
    return compiled


def compile_te_native(terms, lmax, nside=2048):
    """Native TE emitter. Mirrors estimator_backend.compile_te (pol half
    + temp half summed)."""
    import healpy as hp
    import math
    from .estimator_backend import _extract_pol_response_from, _strip_sqrt_l_l_plus_1

    pol_terms  = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    temp_terms = [t for t in terms if t.spin_X == 0 and abs(t.spin_Y) == 1]

    pol_ref = next(t for t in pol_terms if abs(t.spin_X) == 1)
    pol_X_resp = _extract_pol_response_from(pol_ref, "X")
    pol_Y_resp = pol_ref.Y_filter
    pol_L_atom = pol_ref.L_factor
    pol_pair_coeff = complex(pol_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * 2

    temp_ref = next(t for t in temp_terms if abs(t.spin_Y) == 1)
    temp_Y_response = _strip_sqrt_l_l_plus_1(temp_ref.Y_filter, "l2")
    temp_X_iv       = temp_ref.X_filter
    temp_L_atom     = temp_ref.L_factor
    temp_pair_coeff = complex(temp_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2

    px = Pixelization(nside=nside)

    def compiled(T_alm, E_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        E_alm = np.asarray(E_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        grad_L = np.sqrt(ell * (ell + 1))

        # pol half
        p_x = _eval_atom(pol_X_resp, ell, spectra).real
        p_y = _eval_atom(pol_Y_resp, ell, spectra).real
        p_L = _eval_atom(pol_L_atom, ell, spectra).real
        p_L_res = np.where(grad_L > 0, p_L / grad_L, 0.0)
        X_E = hp.almxfl(T_alm, p_x)
        Y_E = hp.almxfl(E_alm, p_y)
        zero = np.zeros_like(T_alm)
        pol_pc = qe_pol_only(px, X_E, zero, Y_E, zero, lmax)
        pol_phi = (pol_pc[0] if pol_pc.ndim == 2 else pol_pc)
        pol_phi = hp.almxfl(pol_phi, p_L_res) * pol_pair_coeff

        # temp half
        t_Y_resp = _eval_atom(temp_Y_response, ell, spectra).real
        t_X_iv   = _eval_atom(temp_X_iv, ell, spectra).real
        t_L      = _eval_atom(temp_L_atom, ell, spectra).real
        t_L_res  = np.where(grad_L > 0, t_L / grad_L, 0.0)
        E_as_X = hp.almxfl(E_alm, t_Y_resp)
        T_as_Y = hp.almxfl(T_alm, t_X_iv)
        temp_pc = qe_temperature_only(px, E_as_X, T_as_Y, lmax)
        temp_phi = (temp_pc[0] if temp_pc.ndim == 2 else temp_pc)
        temp_phi = hp.almxfl(temp_phi, t_L_res) * temp_pair_coeff

        return pol_phi + temp_phi

    compiled.pol_pair_coeff = pol_pair_coeff
    compiled.temp_pair_coeff = temp_pair_coeff
    return compiled
