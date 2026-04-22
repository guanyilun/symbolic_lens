"""
estimator_backend.py — emit a symbolic estimator recipe against falafel's
hand-tested SHT primitives.

Philosophy
----------
Rather than reinvent signed-spin bookkeeping (which was the source of
bugs in an earlier attempt), this backend uses falafel's primitives
directly — ``gradient_spin``, ``qe_temperature_only``,
``qe_spin_pol_deflection``, ``deflection_map_to_phi_curl_alms``. Those
functions encode sign/phase conventions (``rot2d``, ``irot2d``, choice
of ``comp=0`` vs ``comp=1``) that have already been debugged and
validated at the pytempura-normalization level.

Our job here is to:
  1. Take a list of EstimatorTerms produced by ``compile_estimator`` and
     strip the γ factors that pixell's SHT normalization reproduces.
  2. Recognize structural patterns that correspond to falafel's existing
     callable primitives — currently only the TT "single-gradient" case.
  3. Derive the right ell-space filters from the symbolic recipe and
     pass them to the falafel primitive.

This keeps the symbolic engine as the source of truth for which filters
/ which spins go where, and keeps the SHT numerics in falafel where
they're already validated.

Assumptions (TT emitter)
------------------------
The TT lensing recipe, after γ-stripping, collapses to four
EstimatorTerms: the ``(spin_X, spin_Y, spin_L)`` triples are
``(±1, 0, ∓1)`` twice (for the X-gradient pair) plus ``(0, ±1, ∓1)``
twice (for the Y-gradient pair). For X = Y = T the two pairs are
related by relabeling, so we pick the X-gradient pair and apply
falafel's ``qe_temperature_only`` with the filters our symbolic recipe
dictates, multiplied by 2 to account for the X↔Y symmetry.

This emitter therefore does NOT generalize automatically to arbitrary
estimators; it is the TT-validation milestone. Extension to polarization
will plug into different falafel primitives (``gradient_spin`` for
spin-±2 plus ``qe_spin_pol_deflection``).
"""
from __future__ import annotations
import numpy as np
from sympy import Rational

from .atom import AtomicFactor, Const, Var, Pow, Mul, Add, FuncApp, ONE
from .atom import mul as atom_mul
from .estimator import EstimatorTerm


# -------------------------------------------------------------- γ-strip

def strip_gamma(terms: list[EstimatorTerm]) -> list[EstimatorTerm]:
    """Remove the sqrt(2·var+1) factor from each leg of each term.

    These factors come from splitting γ_{l1 L l2} = sqrt((2l1+1)(2L+1)(2l2+1)/(4π))
    across the three legs at W-construction time.  They are absorbed by
    pixell/healpy's SHT normalization and must not be re-applied as
    explicit filter multiplications.
    """
    return [
        EstimatorTerm(
            coeff    = t.coeff,
            L_factor = _strip_sqrt_2var_plus_1(t.L_factor, 'l'),
            X_filter = _strip_sqrt_2var_plus_1(t.X_filter, 'l1'),
            Y_filter = _strip_sqrt_2var_plus_1(t.Y_filter, 'l2'),
            spin_L=t.spin_L, spin_X=t.spin_X, spin_Y=t.spin_Y,
        )
        for t in terms
    ]


def _strip_sqrt_2var_plus_1(atom: AtomicFactor, var_name: str) -> AtomicFactor:
    half = Rational(1, 2)

    def is_target(f):
        if not isinstance(f, Pow):    return False
        if f.exp != half:             return False
        base = f.base
        if not isinstance(base, Add): return False
        if len(base.terms) != 2:      return False
        has_one = any(isinstance(t, Const) and t.value == 1 for t in base.terms)
        def is_two_var(t):
            if not isinstance(t, Mul):                      return False
            if len(t.factors) != 2:                         return False
            has_two = any(isinstance(f_, Const) and f_.value == 2 for f_ in t.factors)
            has_var = any(isinstance(f_, Var)   and f_.name  == var_name for f_ in t.factors)
            return has_two and has_var
        return has_one and any(is_two_var(t) for t in base.terms)

    if isinstance(atom, Mul):
        kept = [f for f in atom.factors if not is_target(f)]
        return atom_mul(*kept) if kept else ONE
    return ONE if is_target(atom) else atom


# -------------------------------------------------- atom → numpy array

def _eval_atom(atom: AtomicFactor, ell: np.ndarray, spectra: dict) -> np.ndarray:
    """Evaluate an atomic factor over ``ell`` given the named spectra.

    Values where the expression diverges or is undefined (sqrt of negative
    arg, 1/0) are zeroed.
    """
    fv = atom.free_vars()
    if not fv:
        return np.full_like(ell, complex(atom.evaluate({})), dtype=complex)
    assert len(fv) == 1, f"multi-var atom: {fv}"
    (name,) = fv
    env = dict(spectra); env[name] = ell.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        arr = np.asarray(atom.evaluate(env), dtype=complex)
    return np.where(np.isfinite(arr), arr, 0.0)


# ---------------------------------------------- TT emitter (milestone 1)

def compile_tt(terms: list[EstimatorTerm], lmax: int, nside: int = 2048):
    """Emit a callable for the TT lensing estimator using
    falafel.qe.qe_temperature_only as the SHT primitive.

    The γ-stripped recipe is expected to have the "canonical" TT structure
    (X-gradient pair with spin_Y = 0).  The emitter identifies the pair
    from the recipe, extracts the ell-space filters, and hands them to
    falafel.

    Returns a callable ``compiled(T_alm, spectra_dict) -> phi_alm``.
    """
    from falafel.qe import pixelization, qe_temperature_only
    import healpy as hp

    xgrad_terms = [t for t in terms if t.spin_Y == 0 and abs(t.spin_X) == 1]
    assert xgrad_terms, "No X-gradient pair found in TT recipe"

    # The ±spin pair members have the same filters; pick one.
    ref = xgrad_terms[0]
    # Coefficient bookkeeping:
    #   ref.coeff                is the per-term atomic coefficient, incl.
    #                            the 1/Δ factor from g = f/(Δ·ĉ·ĉ), the
    #                            residual 1/sqrt(4π) from γ, and the -2
    #                            sign from W_lens_0's prefactor.
    #   × 2                      X↔Y swap symmetry (X = Y = T for TT).
    #   × (-1)                   falafel.qe.qe_spin_temperature_deflection
    #                            bakes in the "-grad * ymap" sign, so our
    #                            symbolic sign must be REMOVED to avoid
    #                            double-counting.
    #   × Δ = 2                  falafel returns the f-pipeline output, not
    #                            the g-pipeline; our 1/Δ is not wanted.
    #   × sqrt(4π)               falafel's SHT conventions reproduce γ
    #                            implicitly; our residual 1/sqrt(4π) is
    #                            not wanted.
    import math
    pair_coeff = 2 * complex(ref.coeff) * (-1) * 2 * math.sqrt(4 * math.pi)

    px = pixelization(nside=nside)

    def compiled(T_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        # Decompose the X filter into
        #   X_filter(l) = gradient_piece · response_piece
        # where gradient_piece = sqrt(l*(l+1)) (applied internally by
        # falafel.gradient_spin).
        x_fl = _eval_atom(ref.X_filter, ell, spectra).real
        y_fl = _eval_atom(ref.Y_filter, ell, spectra).real
        L_fl = _eval_atom(ref.L_factor, ell, spectra).real

        # The gradient factor and L-post-mul that falafel ALREADY supplies
        # internally; divide them out so we don't double-apply.
        grad_l  = np.sqrt(ell * (ell + 1))
        grad_L  = np.sqrt(ell * (ell + 1))
        x_response_fl = np.where(grad_l > 0, x_fl / grad_l, 0.0)
        y_iv_fl       = y_fl
        L_residual_fl = np.where(grad_L > 0, L_fl / grad_L, 0.0)  # expect ≈ 1 if structure matches

        X_response = hp.almxfl(T_alm, x_response_fl)
        Y_iv       = hp.almxfl(T_alm, y_iv_fl)

        phi_curl = qe_temperature_only(px, X_response, Y_iv, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        phi = hp.almxfl(phi, L_residual_fl)     # should be a no-op at structural match

        return pair_coeff * phi

    # Expose the internal pieces for inspection / debugging.
    compiled.ref_term = ref
    compiled.pair_coeff = pair_coeff
    return compiled
