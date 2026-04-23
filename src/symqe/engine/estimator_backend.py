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


# ------------------------------------- polarization emitters (milestone 2)

def _extract_pol_response(terms: list[EstimatorTerm], gradient_leg: str, l_var: str):
    """From a γ-stripped polarization recipe, find a term belonging to the
    X-gradient (or Y-gradient) pair whose X (resp. Y) filter includes
    sqrt((l-1)(l+2)), and extract the "response" filter — the filter minus
    the gradient-spin factor that falafel's gradient_spin(spin=∓2) applies
    internally.

    Returns (X_response_atom, Y_response_atom, L_factor_atom, coeff_ref)
    where the X/Y response atoms are what to pre-multiply the E/B alms by
    before passing to qe_pol_only.  For EE/BB/EB/TB the Y leg is plain
    (spin ±2 alm2map) and carries the IV filter; for the swap pair the
    roles exchange.
    """
    import sympy as sp
    from .atom import Pow, Add, Mul, Var, Const, ONE
    from .atom import mul as atom_mul

    # pick an "m1" gradient term: spin_X of ∓1 (for X-gradient) or spin_Y of ∓1 (Y-gradient)
    if gradient_leg == "X":
        pair = [t for t in terms if abs(t.spin_X) == 1 and abs(t.spin_Y) == 2]
    else:
        pair = [t for t in terms if abs(t.spin_Y) == 1 and abs(t.spin_X) == 2]
    assert pair, f"no {gradient_leg}-gradient |m|=1 pair found"
    ref = pair[0]

    # Strip the gradient factor sqrt((l-1)(l+2)) from the gradient leg's filter.
    def strip_grad_m1(atom: AtomicFactor, var_name: str) -> AtomicFactor:
        """Remove Pow(Mul(Pow(Add(-1, Var), 1/2), Pow(Add(2, Var), 1/2)), ...)
        i.e. sqrt(l-1) * sqrt(l+2) — grad_spin(spin=-2) filter.
        """
        half = sp.Rational(1, 2)
        def is_sqrt_of(base_expr):
            def match(f):
                return isinstance(f, Pow) and f.exp == half and _add_equal(f.base, base_expr)
            return match
        need_lm1 = is_sqrt_of(_make_add(Const(sp.Integer(-1)), Var(var_name)))
        need_lp2 = is_sqrt_of(_make_add(Const(sp.Integer(2)),  Var(var_name)))
        if not isinstance(atom, Mul): return atom
        kept, found_m1, found_p2 = [], False, False
        for f in atom.factors:
            if not found_m1 and need_lm1(f):  found_m1 = True;  continue
            if not found_p2 and need_lp2(f):  found_p2 = True;  continue
            kept.append(f)
        if not (found_m1 and found_p2):
            return atom   # not the m=1 variant
        return atom_mul(*kept) if kept else ONE

    if gradient_leg == "X":
        X_response = strip_grad_m1(ref.X_filter, "l1")
        Y_response = ref.Y_filter
    else:
        X_response = ref.X_filter
        Y_response = strip_grad_m1(ref.Y_filter, "l2")
    return X_response, Y_response, ref.L_factor, ref.coeff


def _make_add(*args: AtomicFactor) -> Add:
    return Add(args)


def _add_equal(a: AtomicFactor, b: AtomicFactor) -> bool:
    """Order-insensitive equality for Add nodes (structural up to term order)."""
    if type(a) != type(b): return False
    if isinstance(a, Add):
        ka = sorted(t.structural_key() for t in a.terms)
        kb = sorted(t.structural_key() for t in b.terms)
        return ka == kb
    return a.structural_key() == b.structural_key()


def compile_pol_same_field(terms: list[EstimatorTerm], lmax: int, nside: int,
                           field: str):
    """Same-field polarization estimator (EE or BB) via qe_pol_only.

    The symbolic recipe has 8 terms: 4 for the X-gradient and 4 for the
    Y-gradient.  falafel's qe_spin_pol_deflection packs each 4-bundle
    (both |m|=1 and |m|=3 gradient flavors × ±spin pair) into one call.
    For X = Y (same field) the two bundles contribute identically, so we
    use the X-gradient bundle and multiply by 2.

    `field` is 'E' or 'B'; determines which slot of qe_pol_only's
    (X_E, X_B, Y_E, Y_B) receives the input alm.
    """
    from falafel.qe import pixelization, qe_pol_only
    import healpy as hp
    import math

    assert field in ('E', 'B')

    X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")

    # Same bookkeeping as compile_tt, plus an extra factor 2 for falafel's
    # /2 at the end of qe_spin_pol_deflection (from the irot2d on
    # spin_alm=±2 with B_alm=0, which leaves the ± components both equal).
    pair_coeff = 2 * complex(coeff_ref) * (-1) * 2 * math.sqrt(4 * math.pi) * 2

    px = pixelization(nside=nside)

    def compiled(alm, spectra):
        alm = np.asarray(alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        x_resp_fl = _eval_atom(X_resp, ell, spectra).real
        y_resp_fl = _eval_atom(Y_resp, ell, spectra).real
        L_fl      = _eval_atom(L_atom, ell, spectra).real

        grad_L = np.sqrt(ell * (ell + 1))
        L_residual_fl = np.where(grad_L > 0, L_fl / grad_L, 0.0)

        X_resp_alm = hp.almxfl(alm, x_resp_fl)
        Y_resp_alm = hp.almxfl(alm, y_resp_fl)
        zero = np.zeros_like(alm)

        if field == 'E':
            phi_curl = qe_pol_only(px, X_resp_alm, zero, Y_resp_alm, zero, lmax)
        else:  # 'B'
            phi_curl = qe_pol_only(px, zero, X_resp_alm, zero, Y_resp_alm, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        phi = hp.almxfl(phi, L_residual_fl)
        return pair_coeff * phi

    compiled.pair_coeff = pair_coeff
    compiled.X_response_atom = X_resp
    compiled.Y_response_atom = Y_resp
    return compiled


def compile_ee(terms, lmax, nside=2048):
    return compile_pol_same_field(terms, lmax, nside, 'E')


def compile_bb(terms, lmax, nside=2048):
    return compile_pol_same_field(terms, lmax, nside, 'B')


def compile_eb(terms: list[EstimatorTerm], lmax: int, nside: int = 2048):
    """EB estimator via falafel.qe.qe_pol_only.

    f^{EB} has TWO W-weighted terms (E-response with B-iv, and B-response
    with E-iv).  The standard CMB-lensing assumption that C^{BB}_primordial
    is negligible kills the second term, leaving only the X-gradient (E
    with response, B with iv) pair — which is exactly what falafel's
    ``pest(xalm('e0'), zero, zero, fBalm)`` computes.

    The ζ⁻ = i carries through; paired with pol_alms(0, B) = (iB, -iB)'s
    own i factor, we get a real net result (phi channel).
    """
    from falafel.qe import pixelization, qe_pol_only
    import healpy as hp
    import math

    # Pick the X-gradient pair whose X_filter has the CE (response on E side).
    # _extract_pol_response picks pair[0]; that may come from plans 5-8
    # (Y-gradient) if the recipe ordering puts those first. We explicitly
    # search for the X-gradient subset.
    xgrad = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    assert xgrad, "no X-gradient pair found in EB recipe"

    # Among the X-gradient pair, pick the |m_X|=1 flavor (used for strip_grad_m1
    # to succeed).
    ref = next(t for t in xgrad if abs(t.spin_X) == 1)
    X_resp = _extract_pol_response_from(ref, "X")
    Y_resp = ref.Y_filter
    L_atom = ref.L_factor

    pair_coeff = complex(ref.coeff) * (-1) * 1 * math.sqrt(4 * math.pi) * 2 * (-1j) * 2

    px = pixelization(nside=nside)

    def compiled(E_alm, B_alm, spectra):
        E_alm = np.asarray(E_alm, dtype=np.complex128)
        B_alm = np.asarray(B_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        x_resp_fl = _eval_atom(X_resp, ell, spectra).real
        y_resp_fl = _eval_atom(Y_resp, ell, spectra).real
        L_fl      = _eval_atom(L_atom, ell, spectra).real

        grad_L = np.sqrt(ell * (ell + 1))
        L_residual_fl = np.where(grad_L > 0, L_fl / grad_L, 0.0)

        X_E = hp.almxfl(E_alm, x_resp_fl)
        Y_B = hp.almxfl(B_alm, y_resp_fl)
        zero = np.zeros_like(E_alm)

        phi_curl = qe_pol_only(px, X_E, zero, zero, Y_B, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        phi = hp.almxfl(phi, L_residual_fl)
        return pair_coeff * phi

    compiled.pair_coeff = pair_coeff
    return compiled


def compile_te(terms: list[EstimatorTerm], lmax: int, nside: int = 2048):
    """TE estimator via the sum of falafel's qe_temperature_only and
    qe_pol_only, following qe_all's ``Tte0 + Pte0`` recipe.

    f^{TE} has two terms — W⁰·C^{TE}(l') and W⁺(swap)·C^{TE}(l) —
    which split into two disjoint subsets of the 6-term symbolic recipe:

      * Plans 1-4  (spin_X ∈ {±1, ±3}, spin_Y = ±2):
            T gradient, E plain → pol primitive.
            Falafel: ``pest(xalm('e_t0'), 0, fEalm, 0)``
                     with xalm('e_t0') = T·C^{TE}/Ĉ^{TT}.
      * Plans 5-6  (spin_X = 0, spin_Y = ±1):
            E gradient, T plain → temperature primitive.
            Falafel: ``test(xalm('t_e0'), fTalm)``
                     with xalm('t_e0') = E·C^{TE}/Ĉ^{EE},  fTalm = T/Ĉ^{TT}.
            Note our (X, Y) = (T, E) while falafel's test() takes
            gradient-side first, so the X/Y roles swap for this half.
    """
    from falafel.qe import pixelization, qe_temperature_only, qe_pol_only
    import healpy as hp
    import math

    # ---- split ----
    pol_terms  = [t for t in terms if abs(t.spin_X) in (1, 3) and abs(t.spin_Y) == 2]
    temp_terms = [t for t in terms if t.spin_X == 0 and abs(t.spin_Y) == 1]
    assert pol_terms and temp_terms, "TE recipe missing expected subsets"

    # -- pol-half reference (X-gradient, |m_X|=1) --
    pol_ref = next(t for t in pol_terms if abs(t.spin_X) == 1)
    pol_X_resp = _extract_pol_response_from(pol_ref, "X")     # T filter pre-gradient
    pol_Y_resp = pol_ref.Y_filter                              # E IV
    pol_L_atom = pol_ref.L_factor
    # Same bookkeeping as compile_eb (no X↔Y symmetry; Δ=1; ζ⁻ absent here
    # because TE uses W⁺ for this half).  Factor 2 at end for the f-vs-g
    # Δ scaling (falafel returns f-pipeline).
    pol_pair_coeff = complex(pol_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2 * 2

    # -- temp-half reference (Y-gradient, |m_Y|=1; X is spin-0 plain) --
    temp_ref = next(t for t in temp_terms if abs(t.spin_Y) == 1)
    # Strip sqrt(l*(l+1)) from Y_filter — falafel.gradient_spin(spin=0)
    # adds that internally.  Use the same m=1 stripper as the pol helper
    # but targeting (0 - 0 type) Var relations.
    # Here the grad factor is sqrt(l*(l+1)) = sqrt(l)·sqrt(l+1).
    temp_Y_response = _strip_sqrt_l_l_plus_1(temp_ref.Y_filter, "l2")
    temp_X_iv       = temp_ref.X_filter
    temp_L_atom     = temp_ref.L_factor
    # Bookkeeping (no X↔Y symmetry for TE; Δ=1).  The extra ×2 accounts
    # for the ±spin pair fusion inside falafel's single gradient_spin call
    # (our plans 5 and 6 are the ±L pair; falafel's gradient_spin(spin=0)
    # with map2alm_spin(spin=1) produces both at once and takes comp=0 =
    # phi alone, so our ref.coeff = -1/(4√π) needs doubling to match).
    temp_pair_coeff = complex(temp_ref.coeff) * (-1) * 1 * math.sqrt(4*math.pi) * 2

    px = pixelization(nside=nside)

    def compiled(T_alm, E_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        E_alm = np.asarray(E_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        # -- pol half --
        p_x = _eval_atom(pol_X_resp, ell, spectra).real
        p_y = _eval_atom(pol_Y_resp, ell, spectra).real
        p_L = _eval_atom(pol_L_atom, ell, spectra).real
        grad_L = np.sqrt(ell * (ell + 1))
        p_L_res = np.where(grad_L > 0, p_L / grad_L, 0.0)
        X_E = hp.almxfl(T_alm, p_x)      # T·CTE/hCT  → feed into falafel's X_E slot
        Y_E = hp.almxfl(E_alm, p_y)      # E/hCE       → feed into falafel's Y_E slot
        zero = np.zeros_like(T_alm)
        pol_pc = qe_pol_only(px, X_E, zero, Y_E, zero, lmax)
        pol_phi = pol_pc[0] if pol_pc.ndim == 2 else pol_pc
        pol_phi = hp.almxfl(pol_phi, p_L_res) * pol_pair_coeff

        # -- temp half (gradient on E side; falafel takes gradient-side as X) --
        t_Y_resp = _eval_atom(temp_Y_response, ell, spectra).real   # CTE/hCE
        t_X_iv   = _eval_atom(temp_X_iv, ell, spectra).real          # 1/hCT
        t_L      = _eval_atom(temp_L_atom, ell, spectra).real
        t_L_res  = np.where(grad_L > 0, t_L / grad_L, 0.0)
        E_as_X = hp.almxfl(E_alm, t_Y_resp)     # E·CTE/hCE  → falafel's X (gradient side)
        T_as_Y = hp.almxfl(T_alm, t_X_iv)       # T/hCT       → falafel's Y (plain side)
        temp_phi = qe_temperature_only(px, E_as_X, T_as_Y, lmax)
        if temp_phi.ndim == 2: temp_phi = temp_phi[0]
        temp_phi = hp.almxfl(temp_phi, t_L_res) * temp_pair_coeff

        return pol_phi + temp_phi

    compiled.pol_pair_coeff = pol_pair_coeff
    compiled.temp_pair_coeff = temp_pair_coeff
    return compiled


def _strip_sqrt_l_l_plus_1(atom: AtomicFactor, var_name: str) -> AtomicFactor:
    """Remove sqrt(l)·sqrt(l+1) from an atom (the spin-0 gradient filter)."""
    from .atom import Pow, Var, Add, Const, Mul, ONE
    from .atom import mul as atom_mul
    import sympy as sp
    half = sp.Rational(1, 2)

    def is_sqrt_l(f):
        return (isinstance(f, Pow) and f.exp == half
                and isinstance(f.base, Var) and f.base.name == var_name)
    def is_sqrt_lp1(f):
        if not (isinstance(f, Pow) and f.exp == half): return False
        b = f.base
        if not isinstance(b, Add) or len(b.terms) != 2: return False
        has_one = any(isinstance(t, Const) and t.value == 1 for t in b.terms)
        has_var = any(isinstance(t, Var) and t.name == var_name for t in b.terms)
        return has_one and has_var

    if not isinstance(atom, Mul): return atom
    kept, found_l, found_lp1 = [], False, False
    for f in atom.factors:
        if not found_l and is_sqrt_l(f):      found_l = True; continue
        if not found_lp1 and is_sqrt_lp1(f):  found_lp1 = True; continue
        kept.append(f)
    return atom_mul(*kept) if kept else ONE


def _extract_pol_response_from(ref: EstimatorTerm, leg: str) -> AtomicFactor:
    """Strip the sqrt((l-1)(l+2)) gradient factor from a specific EstimatorTerm."""
    from .atom import Pow, Add, Mul, Var, Const, ONE
    from .atom import mul as atom_mul
    import sympy as sp
    half = sp.Rational(1, 2)
    var_name = "l1" if leg == "X" else "l2"
    filt = ref.X_filter if leg == "X" else ref.Y_filter

    def is_sqrt_of(base_expr):
        return lambda f: (isinstance(f, Pow) and f.exp == half
                          and _add_equal(f.base, base_expr))
    need_lm1 = is_sqrt_of(_make_add(Const(sp.Integer(-1)), Var(var_name)))
    need_lp2 = is_sqrt_of(_make_add(Const(sp.Integer(2)),  Var(var_name)))

    if not isinstance(filt, Mul): return filt
    kept, found_m1, found_p2 = [], False, False
    for f in filt.factors:
        if not found_m1 and need_lm1(f):  found_m1 = True; continue
        if not found_p2 and need_lp2(f):  found_p2 = True; continue
        kept.append(f)
    return atom_mul(*kept) if kept else ONE


def compile_tb(terms: list[EstimatorTerm], lmax: int, nside: int = 2048):
    """TB estimator via falafel.qe.qe_pol_only.

    f^{TB} has only ONE W-weighted term (no "+swap") — the recipe's
    8 total atomic terms collapse to plans 1-4 only (X-gradient on the
    T side, Y plain on the B side).  X = T, Y = B, so there is no
    X↔Y swap symmetry; Δ = 1.  The ζ⁻ = i factor on W⁻ is carried as
    a complex coefficient; falafel's qe_pol_only with (X_E, 0, 0, Y_B)
    on spin-±2 alms also produces i factors via pol_alms, and the two
    cancel.
    """
    from falafel.qe import pixelization, qe_pol_only
    import healpy as hp
    import math

    X_resp, Y_resp, L_atom, coeff_ref = _extract_pol_response(terms, "X", "l1")

    # Bookkeeping (TB-specific):
    #   × coeff_ref  carries 1/Δ (Δ=1), γ residual, -I/2 from ζ⁻·q⁻
    #   × (-1)       undo falafel's "-grad*ymap" sign
    #   × Δ = 1      no f-vs-g rescaling
    #   × sqrt(4π)   strip γ residual
    #   × 2          undo falafel's /2 at end of qe_spin_pol_deflection
    #   × (-I)       undo the i from pol_alms(0, B) = (iB, -iB)
    #   × 2          extra factor because f_TB has only ONE W term (no swap)
    #                — our symbolic recipe has 4 plans (|m|=1 and |m|=3
    #                gradient flavors) while falafel's pol call produces
    #                the analogue of 8 products (all ±spin combinations);
    #                empirically confirmed at bit-for-bit precision.
    pair_coeff = complex(coeff_ref) * (-1) * 1 * math.sqrt(4 * math.pi) * 2 * (-1j) * 2

    px = pixelization(nside=nside)

    def compiled(T_alm, B_alm, spectra):
        T_alm = np.asarray(T_alm, dtype=np.complex128)
        B_alm = np.asarray(B_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        x_resp_fl = _eval_atom(X_resp, ell, spectra).real
        y_resp_fl = _eval_atom(Y_resp, ell, spectra).real
        L_fl      = _eval_atom(L_atom, ell, spectra).real

        grad_L = np.sqrt(ell * (ell + 1))
        L_residual_fl = np.where(grad_L > 0, L_fl / grad_L, 0.0)

        X_E = hp.almxfl(T_alm, x_resp_fl)   # T·CTE/hCT, used in the E-slot
        Y_B = hp.almxfl(B_alm, y_resp_fl)   # B/hCB
        zero = np.zeros_like(T_alm)

        phi_curl = qe_pol_only(px, X_E, zero, zero, Y_B, lmax)
        phi = phi_curl[0] if phi_curl.ndim == 2 else phi_curl
        phi = hp.almxfl(phi, L_residual_fl)
        return pair_coeff * phi

    compiled.pair_coeff = pair_coeff
    return compiled
