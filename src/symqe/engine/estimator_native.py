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

from .atom import AtomicFactor, FuncApp, Pow, Var, Mul, Add, Const
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


def _resolve_px(px=None, nside=None, shape=None, wcs=None):
    """Build a Pixelization from whichever of (px, nside, shape+wcs) the
    caller supplied.  Enables CAR and HEALPix side-by-side across all
    native emitters with a single signature."""
    if px is not None:
        return px
    if nside is not None:
        return Pixelization(nside=nside)
    if shape is not None and wcs is not None:
        return Pixelization(shape=shape, wcs=wcs)
    raise ValueError("compile_*_native needs one of: px, nside, shape+wcs")


# ======================================================================
# Generic emitter pipeline: fuse_spin_pairs → compile_native
# ======================================================================
#
# Goal: accept any list[EstimatorTerm] from compile_estimator and execute
# it via SHT primitives with NO per-estimator hand-coding and NO hand-tuned
# pair_coeff.  The symbolic structure (spins, filters, coefficients)
# dictates the execution; the user provides only the input alms packed as
# (±|s| pair, spin_alm value) per leg.
#
# The ±spin pair fusion is the enabler: multiple EstimatorTerms that
# differ only by the sign of (s_X, s_Y, s_L) collapse into one FusedPlan
# so that a single pair of alm2map_spin / map2alm_spin calls handles all
# of them.  Without fusion, a generic emitter can't combine the ±spin
# conjugate products correctly — as seen in the rotation debugging.

from dataclasses import dataclass


def _sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


@dataclass(frozen=True)
class FusedPlan:
    """One group of EstimatorTerms that share absolute spins, filters and
    L-factor, differing only in the sign of ``(s_X, s_Y, s_L)``.

    Fields:
      abs_spin_X, abs_spin_Y, abs_spin_L : nonnegative int
        Absolute values of the spin signatures.
      X_filter, Y_filter, L_factor : AtomicFactor
      coeffs : dict[tuple[int, int, int], complex]
        Maps each sign signature ``(sign(s_X), sign(s_Y), sign(s_L))`` —
        each entry in {-1, 0, +1} — to the summed complex coefficient of
        all EstimatorTerms with that signature.
    """
    abs_spin_X: int
    abs_spin_Y: int
    abs_spin_L: int
    X_filter: AtomicFactor
    Y_filter: AtomicFactor
    L_factor: AtomicFactor
    coeffs: dict


def fuse_spin_pairs(terms):
    """Group a list of EstimatorTerms into FusedPlans by
    (|s_X|, |s_Y|, |s_L|, structural filter keys).

    Terms that differ only in sign(s_*) are merged into the same
    FusedPlan; their coefficients are recorded against the sign
    signature.  Within a group, coefficients for the same signature
    are summed (e.g., from independent (1+P)/2 expansions producing
    the same effective atomic term)."""
    from collections import defaultdict
    groups = defaultdict(lambda: {"coeffs": defaultdict(lambda: 0j), "meta": None})
    for t in terms:
        absX, absY, absL = abs(t.spin_X), abs(t.spin_Y), abs(t.spin_L)
        key = (absX, absY, absL,
               t.X_filter.structural_key(),
               t.Y_filter.structural_key(),
               t.L_factor.structural_key())
        sig = (_sign(t.spin_X), _sign(t.spin_Y), _sign(t.spin_L))
        groups[key]["coeffs"][sig] += complex(t.coeff)
        if groups[key]["meta"] is None:
            groups[key]["meta"] = (absX, absY, absL,
                                    t.X_filter, t.Y_filter, t.L_factor)

    plans = []
    for g in groups.values():
        aX, aY, aL, xf, yf, Lf = g["meta"]
        plans.append(FusedPlan(aX, aY, aL, xf, yf, Lf, dict(g["coeffs"])))
    return plans


# -------- input-alm-pair constructors (convenience) --------

def scalar_pair(alm):
    """Pack a scalar (spin-0) alm as a ±0 pair.

    Returns ``((alm, alm), spin_alm=0)`` for direct use in compile_native.
    Use for temperature (T) or any spin-0 field."""
    alm = np.asarray(alm, dtype=np.complex128)
    return np.stack([alm, alm]), 0


def pol_E_pair(E_alm):
    """Pack a pure-E polarization alm as a ±2 pair.  Spin_alm = 2."""
    E_alm = np.asarray(E_alm, dtype=np.complex128)
    return np.stack([E_alm, E_alm]), 2


def pol_B_pair(B_alm):
    """Pack a pure-B polarization alm as a ±2 pair.  Spin_alm = 2."""
    B_alm = np.asarray(B_alm, dtype=np.complex128)
    return np.stack([1j * B_alm, -1j * B_alm]), 2


def pol_EB_pair(E_alm, B_alm):
    """Pack a polarization (E, B) pair as the ±2 alm pair."""
    E = np.asarray(E_alm, dtype=np.complex128)
    B = np.asarray(B_alm, dtype=np.complex128)
    return np.stack([E + 1j * B, E - 1j * B]), 2


# -------- int-truncation bug mirroring (falafel _gradient_spin compat) --------
#
# Falafel's _gradient_spin (spin=±2) initializes its ℓ-filter via
# ``fl = ells * 0`` which — because ``ells`` is an int ndarray — produces
# INTEGER zeros.  Subsequent ``fl[...] = np.sqrt(...)`` assignments are
# silently truncated to int.  This is a real bug in falafel, but
# hand-coded emitters mirror it bit-for-bit (see ``_gradient_spin`` here
# in this file).  The symbolic emitter, by contrast, evaluates the
# Namikawa a-factor as a true float, so the two paths disagree at the
# per-ℓ level (~1% drift for low-ℓ, decaying at high-ℓ).  To stay
# bit-for-bit with falafel, we mirror the truncation on the LADDER leg
# of every plan whose X_filter / Y_filter structurally contains one of
# the two int-truncated ladder factors:
#     raising  : sqrt((l-2)(l+3))   (spin +2 → +3)
#     lowering : sqrt((l-1)(l+2))   (spin -2 → -1)
# The spin-0 temperature ladder sqrt(l(l+1)) is already computed as
# float in falafel (``ells*(ells+1.0)``), so we do NOT truncate it.

def _detect_truncatable_ladder(atom):
    """Return 'raising' | 'lowering' | None based on whether the
    AtomicFactor contains the int-truncatable polarization ladder factors.

    Looks for sqrt((l-2))·sqrt((l+3)) (raising, spin=+2 gradient) or
    sqrt((l-1))·sqrt((l+2)) (lowering, spin=-2 gradient) as subfactors.

    Returns ``None`` if the filter is a *chained* ladder (e.g. the shear
    weight sqrt((l-1)·l·(l+1)·(l+2)) = sqrt(l-1)·sqrt(l)·sqrt(l+1)·
    sqrt(l+2)).  In a chained case falafel would evaluate the whole
    product as a single ``np.sqrt(int_product)`` — no int-truncation
    happens because the int-product is computed first and the sqrt then
    returns float.  The int-truncation bug only applies when the
    polarization ladder factor is isolated (so it goes through
    _gradient_spin's fl=ells*0 initialization).  Presence of sqrt(l) AND
    sqrt(l+1) alongside the polarization ladder is the signal of
    chaining.
    """
    import sympy as sp
    if not isinstance(atom, Mul):
        return None
    has_lm1 = has_lm2 = has_l = has_lp1 = has_lp2 = has_lp3 = False
    for f in atom.factors:
        if not (isinstance(f, Pow) and f.exp == sp.Rational(1, 2)):
            continue
        b = f.base
        if isinstance(b, Var):
            # sqrt(l) — bare variable base
            has_l = True
            continue
        if not isinstance(b, Add):
            continue
        const_vals = [t.value for t in b.terms if isinstance(t, Const)]
        if sp.Integer(-1) in const_vals: has_lm1 = True
        if sp.Integer(-2) in const_vals: has_lm2 = True
        if sp.Integer(1)  in const_vals: has_lp1 = True
        if sp.Integer(2)  in const_vals: has_lp2 = True
        if sp.Integer(3)  in const_vals: has_lp3 = True
    # Chained ladder check: both the spin-0 grad (sqrt(l)·sqrt(l+1)) AND
    # a polarization ladder are present → falafel evaluates as one
    # sqrt(int_product), no truncation.
    is_chained = has_l and has_lp1
    if is_chained:
        return None
    if has_lm2 and has_lp3:
        return 'raising'
    if has_lm1 and has_lp2:
        return 'lowering'
    return None


def _int_trunc_ratio(kind, lmax):
    """Return per-ℓ ratio ``int_fl / float_fl`` of the int-truncated
    vs. true-float ladder factor.  Length ``lmax+1`` (zeroed at l=lmax
    to match falafel's length-mlmax convention)."""
    ells = np.arange(0, lmax)
    if kind == 'raising':
        ff = np.zeros_like(ells, dtype=float)
        ff[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
        fi = ells * 0
        fi[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
    elif kind == 'lowering':
        ff = np.zeros_like(ells, dtype=float)
        ff[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
        fi = ells * 0
        fi[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
    else:
        return None
    ff_full = np.zeros(lmax + 1); ff_full[:lmax] = ff
    fi_full = np.zeros(lmax + 1); fi_full[:lmax] = fi.astype(float)
    return np.where(ff_full != 0, fi_full / ff_full, 1.0)


# -------- the generic emitter --------

def compile_native(terms, lmax, *, px=None, nside=None, shape=None, wcs=None):
    """Compile EstimatorTerms into a generic callable.

    Returns a function with signature::

        result = emit(X_input, Y_input, spectra)

    where ``X_input`` and ``Y_input`` are 2-tuples
    ``(alm_pair, spin_alm)`` as produced by ``scalar_pair``,
    ``pol_E_pair``, ``pol_B_pair`` or ``pol_EB_pair``.

    The output ``result`` is a dict keyed by output-channel sign:

      * abs_spin_L == 0:  ``{0: scalar_alm}``
      * abs_spin_L  > 0:  ``{+1: a_plus_alm, -1: a_minus_alm}``

    Status
    ------
    - **abs_spin_L == 0** (scalar output, e.g. rotation α, source,
      patchy τ): works cleanly.  Output differs from the corresponding
      hand-coded emitter by a CONSTANT overall factor
      (absorbed γ-residual and Δ normalization); the relationship is
      documented per estimator in the hand-coded ``pair_coeff``.
    - **abs_spin_L > 0** (phi/curl output, e.g. lensing TT/EE/BB/TB/EB/TE):
      currently NOT bit-for-bit with hand-coded.  The ±spin pair
      members on the X/Y legs feed different map2alm_spin(spin=aL)
      channels in a way that doesn't decompose into a single global
      sign convention — the hand-coded path uses ONE complex SHT call
      per estimator and extracts phi/curl via gradient/curl modes; the
      generic per-term path currently breaks this coupling.  See
      ``test_generic_emitter.py`` for the regression; for lensing
      continue to use the hand-coded ``compile_{tt,ee,bb,tb,eb,te}_native``.
    """
    import healpy as hp

    px = _resolve_px(px, nside, shape, wcs)
    fused = fuse_spin_pairs(terms)

    def emit(X_input, Y_input, spectra):
        X_pair, X_spin_alm = X_input
        Y_pair, Y_spin_alm = Y_input
        X_pair = np.asarray(X_pair, dtype=np.complex128)
        Y_pair = np.asarray(Y_pair, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        # Output accumulators, keyed by output-channel sign.
        outputs = {}

        for plan in fused:
            x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
            y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
            L_fl = _eval_atom(plan.L_factor, ell, spectra).real.astype(np.float64)

            # Falafel-compat mlmax truncation: falafel.qe.gradient_spin and
            # deflection_map_to_phi_curl_alms build their internal filter via
            # ``ells = np.arange(0, mlmax)`` which yields a length-mlmax array,
            # silently zeroing the l=mlmax mode via cs.almxfl.  Also
            # gradient_spin has ``fl[ells<2]=0``.  These only fire on the
            # LADDER-APPLIED leg (the one where abs_spin_out differs from
            # the field's spin_alm — i.e., the one that goes through
            # _gradient_spin).  The non-ladder leg (abs_spin_out == spin_alm)
            # goes through direct alm2map and has no such truncation.
            # For abs_spin_L == 0 paths (rotation, source, τ) there's no
            # gradient_spin/deflection path at all, so nothing to truncate.
            ladder_X = (plan.abs_spin_X != X_spin_alm)
            ladder_Y = (plan.abs_spin_Y != Y_spin_alm)
            if plan.abs_spin_L > 0:
                if ladder_X and plan.abs_spin_X > 0:
                    x_fl = x_fl.copy(); x_fl[lmax] = 0.0
                    x_fl[0] = 0.0; x_fl[1] = 0.0
                if ladder_Y and plan.abs_spin_Y > 0:
                    y_fl = y_fl.copy(); y_fl[lmax] = 0.0
                    y_fl[0] = 0.0; y_fl[1] = 0.0
                # L-side mlmax truncation: falafel's
                # deflection_map_to_phi_curl_alms builds a length-mlmax
                # sqrt(l(l+1)) filter that zeros l=mlmax.  Mirror this
                # only when plan.L_factor is NONTRIVIAL (i.e. carries a
                # genuine grad factor).  For estimators with L_factor =
                # Const(1) (shear, source, etc.), there's no such filter
                # and we must NOT zero l=lmax on the output side.
                if not (isinstance(plan.L_factor, Const)
                        and complex(plan.L_factor.evaluate({})) == 1):
                    L_fl = L_fl.copy(); L_fl[lmax] = 0.0

                # Falafel-compat int-truncation of polarization ladder
                # factors.  Hand-coded _gradient_spin(spin=±2) builds its
                # ℓ-filter as an int array (``fl = ells * 0``) then assigns
                # sqrt(...) values to it, silently truncating to int.
                # Mirror that here by substituting the truncated ratio into
                # x_fl / y_fl on any ladder leg whose filter structurally
                # carries one of the two int-truncated ladder factors.
                if ladder_X and plan.abs_spin_X > 0:
                    kind = _detect_truncatable_ladder(plan.X_filter)
                    if kind is not None:
                        x_fl = x_fl * _int_trunc_ratio(kind, lmax)
                if ladder_Y and plan.abs_spin_Y > 0:
                    kind = _detect_truncatable_ladder(plan.Y_filter)
                    if kind is not None:
                        y_fl = y_fl * _int_trunc_ratio(kind, lmax)

            Xf = np.stack([hp.almxfl(X_pair[0], x_fl),
                           hp.almxfl(X_pair[1], x_fl)])
            Yf = np.stack([hp.almxfl(Y_pair[0], y_fl),
                           hp.almxfl(Y_pair[1], y_fl)])

            X_maps = _alm_to_signed_pair(px, Xf, X_spin_alm,
                                          plan.abs_spin_X, lmax)
            Y_maps = _alm_to_signed_pair(px, Yf, Y_spin_alm,
                                          plan.abs_spin_Y, lmax)

            # Three convention rules applied on top of the symbolic coeff:
            #
            # 1. ``flip_*`` (pair-member selection): for lensing plans
            #    (abs_spin_L>0), hand-coded's _gradient_spin picks
            #    M_+|s_out| (comp=0) for T-leg spin=0 inputs, and picks
            #    specific ±|s_out| members on pol legs via its comp/sign
            #    convention.  Symbolically, sig sX=-1 naively picks M_-|s|;
            #    flipping makes it pick M_+|s|.  Verified bit-for-bit on TT.
            #
            # 2. ``plan_factor`` NP-ladder sign (W^+ plans only): the
            #    symbolic Namikawa a(l, s) = -sqrt((l-s)(l+s+1)/2) carries
            #    a uniform minus sign.  But Newman-Penrose ladder operators
            #    have OPPOSITE signs:
            #        ð (raising):  -sqrt((l-s)(l+s+1))
            #        ð̄ (lowering): +sqrt((l+s)(l-s+1))
            #    Hand-coded falafel's _gradient_spin encodes this via
            #    sign=-1 on spin=+2 (→ |s_out|=3) branch.  The symbolic
            #    emitter restores the relative sign on any W^+ plan with
            #    |s|=3 on a ladder leg.
            #
            # 3. ``plan_factor`` (-1j) rotation (W^- plans only): W^-
            #    coefficients carry a ζ_-=i factor.  The pixel product
            #    then has an imaginary prefactor; map2alm_spin(j·prod)
            #    evaluates to the CURL mode, not the grad mode — a 90°
            #    rotation in (grad, curl) space.  Hand-coded's pair_coeff
            #    for TB/EB includes a (-1j) factor; the symbolic emitter
            #    applies the same to each W^- plan's prod.  Also, the
            #    (1-P)/2 parity of W^- already supplies the relative sign
            #    between |sX|=1 and |sX|=3 branches, so NP flip is NOT
            #    applied on W^- plans.
            #
            # Detection: W^+ plans have REAL coefficients (ζ_+=1);
            # W^- plans have IMAGINARY coefficients (ζ_-=i).
            flip_X = (plan.abs_spin_L > 0) and (plan.abs_spin_X > 0)
            flip_Y = (plan.abs_spin_L > 0) and (plan.abs_spin_Y > 0)
            is_W_plus = all(abs(complex(c).imag) < 1e-12 * (abs(complex(c).real) + 1e-30)
                            for c in plan.coeffs.values())
            plan_factor = 1 + 0j
            if plan.abs_spin_L > 0:
                if is_W_plus:
                    if plan.abs_spin_X == 3:
                        plan_factor = -plan_factor
                    if plan.abs_spin_Y == 3:
                        plan_factor = -plan_factor
                else:
                    plan_factor = -1j * plan_factor
                # Cross-half sign rule (TE-pol-like plans, W^+ only).  A W^+
                # plan where the X input is scalar (spin_alm_X=0) but the X
                # output is |sX|>0 AND the Y input is polarization
                # (|spin_alm_Y|=2, |sY|>0) originates from a column-swapped
                # W_lens_p (as in f_TE's pol half:
                # W_lens_p(l2, l, l1) · CTE(l1)).  The symbolic engine
                # absorbs the column swap via P²=1, but the resulting per-
                # plan sign convention is opposite to hand-coded
                # qe_pol_only's sign convention.  Flip to align with the
                # temp half (which matches hand-coded without flip).
                # Restricted to W^+: for W^- estimators like TB the same
                # spin pattern has the correct sign already (verified
                # empirically: TB xcorr=+1 without this rule).
                if (is_W_plus and X_spin_alm == 0 and plan.abs_spin_X > 0
                        and abs(Y_spin_alm) == 2 and plan.abs_spin_Y > 0):
                    plan_factor = -plan_factor
                # Pol-pol W^+ sign rule (EE/BB).  When BOTH legs carry
                # polarization input (|spin_alm_X|=|spin_alm_Y|=2) and the
                # plan is W^+, hand-coded qe_pol_only's ``prod / 2 = -g_m2
                # · ymap[0] - g_p2 · ymap[1]`` generates an overall sign
                # opposite to the symbolic's (sX, sY) sign assignment.
                # This does not affect EB (W^-) or TE-pol (scalar X).
                if (is_W_plus and abs(X_spin_alm) == 2
                        and abs(Y_spin_alm) == 2
                        and plan.abs_spin_X > 0 and plan.abs_spin_Y > 0):
                    plan_factor = -plan_factor
            for (sX, sY, sL), coeff in plan.coeffs.items():
                sX_eff = -sX if flip_X else sX
                sY_eff = -sY if flip_Y else sY
                Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
                My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
                prod = plan_factor * complex(coeff) * Mx * My

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
                    alm_pair = px.map2alm_spin(prod, lmax=lmax,
                                                spin_alm=0,
                                                spin_transform=plan.abs_spin_L)
                    pick = alm_pair[0] if sL >= 0 else alm_pair[1]
                    pick = hp.almxfl(pick, L_fl)
                    outputs[sL] = pick if sL not in outputs else outputs[sL] + pick

        return outputs

    emit.fused_plans = fused
    return emit


# ----------------------------------------------------------------------
# High-level wrapper: build_estimator
# ----------------------------------------------------------------------
#
# compile_native returns a callable with a raw (alm_pair, spin_alm) input
# convention.  That's the right primitive but it's awkward for the common
# physicist use case — "I have a T alm, I want a phi alm".  build_estimator
# wraps it with an explicit ``X=`` / ``Y=`` tag so the returned callable
# takes raw alms directly and does the ±spin pair packing internally.

def build_estimator(g_or_recipe, lmax, *,
                    X=None, Y=None,
                    nside=None, shape=None, wcs=None, px=None):
    """Build the QE estimator callable from a symbolic weight or a
    pre-compiled recipe.

    Parameters
    ----------
    g_or_recipe : sympy expression OR list[EstimatorTerm]
        Either the symbolic weight ``g(l, L, l')`` (which is compiled
        internally via ``compile_estimator``) or a recipe already produced
        by ``compile_estimator``.  Use the two-step form when you want to
        inspect the recipe first (``symqe.pretty(recipe)``).
    lmax : int
    X, Y : {'T', 'E', 'B'}, optional
        The physical field on each input leg.  When both are given, the
        returned estimator takes raw alms directly::

            estimator(X_alm, Y_alm, spectra) -> phi_alm

        When omitted, the returned estimator exposes the lower-level
        (alm_pair, spin_alm) interface — useful for custom input packings.
    nside, shape, wcs, px
        Pixelization, same options as ``compile_native``.

    Returns
    -------
    estimator : callable
        The QE estimator.  Carries ``.recipe`` (the EstimatorTerm list)
        and ``.fused_plans`` for introspection.
    """
    if isinstance(g_or_recipe, list):
        recipe = g_or_recipe
    else:
        from .estimator import compile_estimator
        recipe = compile_estimator(g_or_recipe)

    raw = compile_native(recipe, lmax, px=px, nside=nside, shape=shape, wcs=wcs)

    if X is None and Y is None:
        raw.recipe = recipe
        return raw

    if X is None or Y is None:
        raise ValueError("Pass both X and Y, or neither.")

    pack_X = _make_packer(X)
    pack_Y = _make_packer(Y)

    def estimator(X_alm, Y_alm, spectra):
        return raw(pack_X(X_alm), pack_Y(Y_alm), spectra)

    estimator.recipe = recipe
    estimator.fused_plans = raw.fused_plans
    estimator.raw = raw
    return estimator


def _make_packer(field):
    if field == 'T':
        return scalar_pair
    if field == 'E':
        return pol_E_pair
    if field == 'B':
        return pol_B_pair
    raise ValueError(f"Unknown field {field!r}; expected 'T', 'E', or 'B'.")


def _alm_to_signed_pair(px, filtered_pair, spin_alm_in, abs_spin_out, lmax):
    """Produce (M_+|s_out|, M_-|s_out|) complex-map pair from a filtered alm pair.

    For abs_spin_out = 0, returns (scalar_map, scalar_map) — the ±0
    components are identical.
    """
    if abs_spin_out == 0:
        m = px.alm2map(filtered_pair[0], spin=0, ncomp=1, mlmax=lmax)[0]
        m = m.astype(np.complex128)
        return (m, m)
    pair_map = px.alm2map_spin(filtered_pair, spin_alm=spin_alm_in,
                               spin_transform=abs_spin_out,
                               ncomp=2, mlmax=lmax)
    return (pair_map[0], pair_map[1])


# =====================================================================
# Per-field convenience wrappers
# =====================================================================
#
# Each of these wraps compile_native with a natural input signature
# (e.g. ``compile_tt(terms, ...)(T_alm, spec)``) and applies a known
# per-estimator scale factor on top of compile_native's output so it
# matches the falafel / pytempura convention bit-for-bit.
#
# For 7 of the 8 estimators (TT/EE/BB/TB/TE/ROT-EB/SRC-TT) the only
# difference between compile_native and the hand-coded falafel path is
# a single constant: γ-residual × X↔Y symmetry × Δ.  The scale cancels
# identically in any A_L-normalized estimate, so for search/ranking
# prefer compile_qe (which carries the normalization) or compile_native
# directly.  Use these wrappers when you need raw alms that cross-check
# bit-for-bit against falafel/pytempura.
#
# EB is the exception: compile_native disagrees with the hand-coded
# path by more than a scalar (Hu-Okamoto structural difference — see
# memory 'generic_emitter_state.md'), so compile_eb keeps its own
# hand-coded SHT recipe.

import math

_SCALE_LENSING = 4 * math.sqrt(math.pi)   # TT, EE, BB, TB, TE
_SCALE_ROT_EB  = 2 * math.sqrt(math.pi)   # rotation α (EB)
_SCALE_SRC_TT  =     math.sqrt(math.pi)   # amplitude / source (TT)


def compile_tt(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """TT lensing estimator.  Returns ``compiled(T_alm, spectra) -> phi_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(T_alm, spectra):
        return _SCALE_LENSING * emit(scalar_pair(T_alm), scalar_pair(T_alm), spectra)[+1]
    compiled.scale = _SCALE_LENSING
    return compiled


def compile_ee(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """EE lensing estimator.  Returns ``compiled(E_alm, spectra) -> phi_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(E_alm, spectra):
        return _SCALE_LENSING * emit(pol_E_pair(E_alm), pol_E_pair(E_alm), spectra)[+1]
    compiled.scale = _SCALE_LENSING
    return compiled


def compile_bb(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """BB lensing estimator.  Returns ``compiled(B_alm, spectra) -> phi_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(B_alm, spectra):
        return _SCALE_LENSING * emit(pol_B_pair(B_alm), pol_B_pair(B_alm), spectra)[+1]
    compiled.scale = _SCALE_LENSING
    return compiled


def compile_tb(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """TB lensing estimator.  Returns ``compiled(T_alm, B_alm, spectra) -> phi_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(T_alm, B_alm, spectra):
        return _SCALE_LENSING * emit(scalar_pair(T_alm), pol_B_pair(B_alm), spectra)[+1]
    compiled.scale = _SCALE_LENSING
    return compiled


def compile_te(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """TE lensing estimator.  Returns ``compiled(T_alm, E_alm, spectra) -> phi_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(T_alm, E_alm, spectra):
        return _SCALE_LENSING * emit(scalar_pair(T_alm), pol_E_pair(E_alm), spectra)[+1]
    compiled.scale = _SCALE_LENSING
    return compiled


def compile_rot_eb(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """Rotation (α) EB estimator.  Returns ``compiled(E_alm, B_alm, spectra) -> alpha_alm``."""
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(E_alm, B_alm, spectra):
        return _SCALE_ROT_EB * emit(pol_E_pair(E_alm), pol_B_pair(B_alm), spectra)[0]
    compiled.scale = _SCALE_ROT_EB
    return compiled


def compile_source_tt(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """Source / amplitude TT estimator.  Returns ``compiled(T_alm, spectra) -> src_alm``.

    The Namikawa-style ε estimator — spin-0 output, no sqrt(L(L+1))
    post-multiplier.  Compared against ``falafel.qe.qe_source`` modulo
    an inverse-variance filter convention (see test_source_vs_falafel.py).
    """
    emit = compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    def compiled(T_alm, spectra):
        return _SCALE_SRC_TT * emit(scalar_pair(T_alm), scalar_pair(T_alm), spectra)[0]
    compiled.scale = _SCALE_SRC_TT
    return compiled


# --- EB: kept as hand-coded SHT (compile_native structurally diverges) ---

def compile_eb(terms, lmax, *, nside=None, shape=None, wcs=None, px=None):
    """EB lensing estimator.  Returns ``compiled(E_alm, B_alm, spectra) -> phi_alm``.

    Kept as a dedicated hand-coded SHT recipe because the generic
    ``compile_native`` disagrees with falafel by more than a constant
    here (Hu-Okamoto structural difference — see memory
    'generic_emitter_state.md').  Until that gap is closed in
    compile_native, this is the bit-for-bit EB path.
    """
    import healpy as hp
    from .estimator_backend import _extract_pol_response_from

    xgrad = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    ref = next(t for t in xgrad if abs(t.spin_X) == 1)
    X_resp = _extract_pol_response_from(ref, "X")
    Y_resp = ref.Y_filter
    L_atom = ref.L_factor
    pair_coeff = complex(ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * (-1j) * 2

    px = _resolve_px(px, nside, shape, wcs)

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
