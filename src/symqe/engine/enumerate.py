"""Grammar enumerator for CMB quadratic-estimator weight candidates.

BFS over the DSL the symbolic compiler already accepts (see memory
`search_space.md` for the full scope analysis).  Each enumerated
candidate is a sympy expression of the form

    coeff · γ_f(l1, l, l2) · X_filter(l1) · Y_filter(l2)
          · w3j(l, l1, l2; m_L, m_X, m_Y) · P^{0|1}

with m_L + m_X + m_Y = 0 and |m_*| ≤ ``m_max``.  Plain structural
dedup: two expressions that compile to the same canonical set of
EstimatorTerms are considered identical.

Downstream: pair with ``compile_qe_from_f`` + ``fisher_fom`` to score
and rank (see ``scripts/demos/demo_search_enum.py``).

Open questions deferred to callers (see ``search_space.md``):
  * field_gating — whether to auto-restrict spectrum atoms to the
    leg's expected field type.  ``permissive`` (default) lets the
    enumerator propose any combination; ``strict`` requires
    ``allowed_spectra`` to name the valid atoms per leg.
  * unknown_spectra — user-declared ``sympy.Function`` instances can
    be passed via ``extra_spectra`` and will be enumerated alongside
    the standard CMB spectra.  Their runtime values are the caller's
    problem.
"""
from __future__ import annotations
from itertools import product, combinations_with_replacement
from typing import Iterator

from sympy import sympify, I, Expr

from .l12_sum import l, l1, l2, P, wigner_3j
from .namikawa import gamma_f, a, CT, CE, CB, CTE
from .estimator import compile_estimator


# --- vocabulary --------------------------------------------------------

_DEFAULT_LADDER_SPINS = (-2, -1, 0, 1, 2)
_DEFAULT_SPECTRA = {"CT": CT, "CE": CE, "CB": CB, "CTE": CTE}
_DEFAULT_COEFFS = (1, -1, I, -I)


def m_triples(m_max: int = 3) -> list[tuple[int, int, int]]:
    """All (m_L, m_X, m_Y) with Σm=0 and |m_*| ≤ m_max."""
    out = []
    for mL in range(-m_max, m_max + 1):
        for mX in range(-m_max, m_max + 1):
            mY = -mL - mX
            if abs(mY) <= m_max:
                out.append((mL, mX, mY))
    return out


def leg_atoms(ell_sym, *, ladder_spins=_DEFAULT_LADDER_SPINS,
              spectra=None):
    """All single atomic factors that can appear on one leg."""
    if spectra is None:
        spectra = _DEFAULT_SPECTRA
    atoms = [sympify(1)]
    for s in ladder_spins:
        atoms.append(a(ell_sym, s))
    for Cl in spectra.values():
        atoms.append(Cl(ell_sym))
    return atoms


def leg_filters(ell_sym, *, max_factors: int = 2,
                ladder_spins=_DEFAULT_LADDER_SPINS,
                spectra=None):
    """All products of ≤ max_factors leg atoms (unordered, with repetition)."""
    assert max_factors >= 1
    atoms = [x for x in leg_atoms(ell_sym, ladder_spins=ladder_spins,
                                  spectra=spectra) if x != 1]
    yield sympify(1)
    for atom in atoms:
        yield atom
    if max_factors >= 2:
        for a1, a2 in combinations_with_replacement(atoms, 2):
            yield a1 * a2


# --- canonical key for dedup ------------------------------------------

def canonical_key(expr: Expr):
    """Return a hashable key that identifies this weight up to the
    equivalences the engine already normalizes (3j column perms, P²=1,
    factor ordering, coefficient placement).

    Returns None if the expression doesn't fit the grammar
    (compile_estimator rejects it) or reduces to zero.
    """
    try:
        terms = compile_estimator(expr)
    except Exception:
        return None
    if not terms:
        return None
    summaries = []
    for t in terms:
        summaries.append((
            t.spin_L, t.spin_X, t.spin_Y,
            t.L_factor.structural_key(),
            t.X_filter.structural_key(),
            t.Y_filter.structural_key(),
            repr(complex(t.coeff)),
        ))
    return tuple(sorted(summaries))


# --- enumerator -------------------------------------------------------

_DEFAULT_PARITY_FACTORS = (sympify(1), P, 1 + P, 1 - P)


def enumerate_candidates(
    *,
    m_max: int = 3,
    coeffs=_DEFAULT_COEFFS,
    parity_factors=_DEFAULT_PARITY_FACTORS,
    max_leg_factors: int = 2,
    ladder_spins=_DEFAULT_LADDER_SPINS,
    spectra_X=None,
    spectra_Y=None,
    field_gating: str = "permissive",
    restrict_m_L: tuple | None = None,
    symmetrize_l1l2: bool = False,
) -> Iterator[tuple[dict, Expr]]:
    """Yield distinct (metadata, sympy_expr) pairs.

    Parameters
    ----------
    m_max : int
        Max absolute value of 3j m-indices.
    coeffs : iterable
        Complex coefficients to try (default ±1, ±i).
    parity_factors : iterable of sympy.Expr
        P-parity factors to multiply the base term by.  Default is
        ``(1, P, 1+P, 1-P)``: the first two are the plain P⁰/P¹ powers,
        the last two are the Namikawa q± = (1±P)/2 selectors (up to
        scale).  Terms containing sums like ``1+P`` will compile into
        multi-term EstimatorTerm lists, which is how q_plus-style
        weights become a single candidate.
    max_leg_factors : int
        Max number of atoms multiplied on each leg filter.
    ladder_spins : iterable of int
        Which a(ell, s) spins to enumerate.
    spectra_X, spectra_Y : dict[str, sympy.Function] or None
        Spectrum atoms allowed on X/Y legs.  ``None`` means all of
        ``_DEFAULT_SPECTRA``.
    field_gating : 'permissive' | 'strict'
        'permissive' (default): spectra_X/Y control vocabulary only,
        arbitrary products allowed.  'strict': reserved for future
        per-leg-field checks (currently behaves the same).
    restrict_m_L : tuple of int or None
        If given, only enumerate m-triples whose m_L is in this set.
        Useful e.g. ``(-1, 1)`` for lensing-like scalar output.
    symmetrize_l1l2 : bool
        If True, also emit ``expr + swap_l1l2(expr)`` for each
        candidate.  Useful for TT/EE/BB-style symmetric estimators
        (Hu-Okamoto f_TT is exactly a symmetrization of a half-weight).

    Yields
    ------
    (metadata, expr) where metadata is a dict of the candidate's
    generating parameters (coeff, m_triple, parity, x_filter,
    y_filter, flavor).
    """
    assert field_gating in ("permissive", "strict")

    mlist = m_triples(m_max)
    if restrict_m_L is not None:
        allowed_mL = set(restrict_m_L)
        mlist = [t for t in mlist if t[0] in allowed_mL]

    X_filters = list(leg_filters(l1, max_factors=max_leg_factors,
                                 ladder_spins=ladder_spins,
                                 spectra=spectra_X))
    Y_filters = list(leg_filters(l2, max_factors=max_leg_factors,
                                 ladder_spins=ladder_spins,
                                 spectra=spectra_Y))

    seen = set()
    for coeff in coeffs:
        for (mL, mX, mY) in mlist:
            for parity in parity_factors:
                for X_filter in X_filters:
                    for Y_filter in Y_filters:
                        base = (coeff * gamma_f(l1, l, l2) * X_filter * Y_filter
                                * wigner_3j(l, l1, l2, mL, mX, mY) * parity)
                        exprs = [("raw", base)]
                        if symmetrize_l1l2:
                            swapped = base.subs({l1: l2, l2: l1},
                                                simultaneous=True)
                            exprs.append(("sym", base + swapped))

                        for tag, expr in exprs:
                            key = canonical_key(expr)
                            if key is None or key in seen:
                                continue
                            seen.add(key)
                            meta = dict(
                                coeff=coeff,
                                m_triple=(mL, mX, mY),
                                parity=parity,
                                X_filter=X_filter,
                                Y_filter=Y_filter,
                                flavor=tag,
                            )
                            yield meta, expr


# --- small scoring convenience ---------------------------------------

def score_candidates(
    candidates,
    *,
    lmax,
    Delta,
    hCxx,
    hCyy,
    user_funcs,
    spectra_args,
    L_range,
    L_ref,
    px=None,
    nside=None,
    shape=None,
    wcs=None,
    progress: bool = False,
):
    """Compile each candidate via compile_qe_from_f, score via fisher_fom,
    return a sorted list of (score, metadata, expr).

    ``spectra_args`` is the positional spectrum tuple matching
    ``user_funcs`` (same convention as the returned A_L_fn / N_L_phi_fn
    of compile_qe_from_f).  Candidates that fail to compile or score
    non-finite get ``-inf``.
    """
    import numpy as np
    from .compile_qe import compile_qe_from_f
    from .scoring import fisher_fom

    results = []
    cands = list(candidates)
    for i, (meta, expr) in enumerate(cands):
        if progress and (i % 50 == 0):
            print(f"  scoring {i}/{len(cands)}...", flush=True)
        try:
            _, A_L_fn, N_L_phi_fn = compile_qe_from_f(
                expr, lmax, Delta=Delta, hCxx=hCxx, hCyy=hCyy,
                user_funcs=user_funcs, px=px, nside=nside,
                shape=shape, wcs=wcs,
            )
            N = N_L_phi_fn(*spectra_args)
            A = A_L_fn(*spectra_args)
            fom = fisher_fom(N, A, L_range=L_range, L_ref=L_ref)
        except Exception:
            fom = -np.inf
        results.append((fom, meta, expr))
    results.sort(key=lambda r: r[0], reverse=True)
    return results
