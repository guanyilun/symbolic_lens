"""Unified compilation of a symbolic QE weight into BOTH the estimator
and the Hu-Okamoto normalization A_L, from one symbolic expression.

Two entry points:

  * ``compile_qe(g, ...)``         — take the filter g; derive f = g·Δ·hC·hC.
  * ``compile_qe_from_f(f, ...)``  — take the physical weight f; derive
                                     the minimum-variance filter g = f/(Δ·hC·hC).

The ``_from_f`` variant is preferred for search work because it forces
apples-to-apples ranking: the engine always pairs each candidate f with
its minimum-variance g, so Fisher comparisons reflect structural
differences in f rather than filter-choice artifacts.

Both variants return ``(estimator, A_L_fn, N_L_phi_fn)``.  See
``symqe.engine.scoring.fisher_fom`` for a gauge-fixed scalar FOM
suitable for ranking candidates.
"""
import numpy as np
from sympy import cancel, Symbol, Function, I as _sympy_I


def _flip_I(expr):
    """Return expr with every ``I`` replaced by ``-I``.

    Equivalent to complex conjugation for our symbolic weights: the
    only non-real atom in the DSL (sympy spectra, wigner_3j, l/l1/l2,
    P, ladder a(l,s), gamma_f) is the imaginary unit ``I`` that rides
    on ζ^- = i.  A full sympy ``conjugate()`` call leaves an opaque
    ``conjugate(...)`` wrapper that the atomic-tree bridge cannot
    convert — subs-ing ``I → -I`` is the semantically-equivalent
    operation that stays inside the DSL.
    """
    return expr.subs(_sympy_I, -_sympy_I)

from .l12_sum import l, l1, l2, NormCompiler
from .estimator import compile_estimator
from .estimator_backend import strip_gamma
from .estimator_native import compile_native


def compile_qe(
    g_symbolic,
    lmax,
    *,
    Delta,
    hCxx,
    hCyy,
    user_funcs,
    rlmin=2,
    rlmax=None,
    px=None,
    nside=None,
    shape=None,
    wcs=None,
):
    """Compile one symbolic g into (estimator, normalization).

    Parameters
    ----------
    g_symbolic : sympy.Expr
        The QE weight, expected to contain ``1/hCxx(l1)/hCyy(l2)`` factors.
    lmax : int
        Output L range for both estimator and normalization.
    Delta : int
        QE normalization factor (2 for XX-type estimators, 1 for XY).
    hCxx, hCyy : sympy.Function
        The IV-filter Function classes used in ``g_symbolic`` on the X and Y
        legs (e.g., ``hCT``, ``hCE``).
    user_funcs : list[sympy.Function]
        Ordered list of all sympy Function classes (e.g.
        ``[hCT, CT]`` for TT) that ``g_symbolic`` references.  This fixes
        the positional order of the spectra arguments to the returned
        normalization callable.
    rlmin, rlmax : int
        GL-quadrature range for the L12Sum compiler.  Default rlmax=lmax.
    px / nside / shape+wcs : pixell-compat SHT context.  Pass one of them.

    Returns
    -------
    estimator : callable (X_pair, Y_pair, spec_dict) -> {sL: alm}
    A_L_fn : callable (*spectra_arrays) -> A_L array
        Hu-Okamoto normalization: A_L = 1 / [(1/(2L+1)) sum f·g].
    N_L_phi_fn : callable (*spectra_arrays) -> N_L^phi array
        Reconstruction noise power spectrum:
            N_L^phi = A_L² · (1/(2L+1)) · sum g² · hCxx(l1) · hCyy(l2)
        This IS the right figure of merit for search: smaller = better
        reconstruction, and it is invariant under g -> α·g (response
        and noise both scale by α² and cancel).
        For the OPTIMAL g = f/(Δ · hC · hC), N_L^phi = A_L/Δ (standard
        Hu-Okamoto result).  For non-optimal g, the two diverge.

    Spectra are positional, in the order of ``user_funcs``.
    """
    return _compile_qe_core(
        g_symbolic=g_symbolic, f_symbolic=None,
        lmax=lmax, Delta=Delta, hCxx=hCxx, hCyy=hCyy,
        user_funcs=user_funcs, rlmin=rlmin, rlmax=rlmax,
        px=px, nside=nside, shape=shape, wcs=wcs,
    )


def compile_qe_from_f(
    f_symbolic,
    lmax,
    *,
    Delta,
    hCxx,
    hCyy,
    user_funcs,
    rlmin=2,
    rlmax=None,
    px=None,
    nside=None,
    shape=None,
    wcs=None,
):
    """Compile a physical weight ``f`` into (estimator, normalization, noise).

    The engine derives the minimum-variance filter ``g = f/(Δ·hCxx·hCyy)``
    internally, so you cannot accidentally pair a candidate ``f`` with a
    non-optimal filter.  Use this variant for search/ranking work.

    Parameters and return value are the same as :func:`compile_qe`, but
    the first argument is the physical weight ``f_symbolic`` rather than
    the filter ``g_symbolic``.

    Response (A_L⁻¹) uses the gauge-correct form ``Σ g·f / (2L+1)``
    (equivalent to ``Σ f²/(Δ·hC·hC)/(2L+1)`` with the optimal g);
    variance uses ``Σ g²·hC·hC / (2L+1)``.  Together with
    :func:`~symqe.engine.scoring.fisher_fom` the outputs give a
    gauge-invariant ranking under ``f → α·f``.
    """
    return _compile_qe_core(
        g_symbolic=None, f_symbolic=f_symbolic,
        lmax=lmax, Delta=Delta, hCxx=hCxx, hCyy=hCyy,
        user_funcs=user_funcs, rlmin=rlmin, rlmax=rlmax,
        px=px, nside=nside, shape=shape, wcs=wcs,
    )


def _compile_qe_core(
    *, g_symbolic, f_symbolic, lmax, Delta, hCxx, hCyy, user_funcs,
    rlmin, rlmax, px, nside, shape, wcs,
):
    rlmax = rlmax if rlmax is not None else lmax

    # Derive whichever of (g, f) was not supplied.
    if f_symbolic is None:
        # Legacy path: user gave g; f is implicitly g·Δ·hC·hC.  A_L^{-1}
        # integrand g·f = g²·Δ·hC·hC (equal only when f is actually the
        # implied optimal response for this g).
        f_symbolic = g_symbolic * Delta * hCxx(l1) * hCyy(l2)
    if g_symbolic is None:
        # Gauge-correct path: user gave f; derive minimum-variance filter.
        g_symbolic = f_symbolic / (Delta * hCxx(l1) * hCyy(l2))

    # --- Estimator side ---
    terms = strip_gamma(compile_estimator(g_symbolic))
    estimator = compile_native(
        terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs
    )

    # --- Normalization side ---
    # Response (A_L^{-1}):  (1/(2L+1)) · sum g · conj(f)
    # conj(f) differs from f only for complex-coefficient weights
    # (e.g. W^- with ζ^- = i).  For real-coefficient weights
    # (lensing TT/EE/BB/TE) conj(f) = f and the old behavior is
    # recovered; for complex-coefficient weights, using g·f (not
    # g·conj(f)) would give a NEGATIVE integrand (I·I = -1), which
    # the A_L = 1/inv path silently zeroed to `inf`.
    integrand_AL_inv = cancel(g_symbolic * _flip_I(f_symbolic) / (2 * l + 1))

    # Noise variance per mode: (1/(2L+1)) · sum |g|² · hCxx(l1) · hCyy(l2)
    integrand_v = cancel(
        g_symbolic * _flip_I(g_symbolic) * hCxx(l1) * hCyy(l2)
        / (2 * l + 1)
    )

    compiler = NormCompiler(lmax=lmax, rlmin=rlmin, rlmax=rlmax)
    AL_inv_fn, _ = compiler.build_and_compile(integrand_AL_inv, args=[l] + list(user_funcs))
    v_fn,      _ = compiler.build_and_compile(integrand_v,      args=[l] + list(user_funcs))

    L_out = np.arange(lmax + 1, dtype=int)

    def A_L_fn(*spectra):
        inv = AL_inv_fn(L_out, *spectra)
        return np.where(inv > 0, 1.0 / inv, np.inf)

    def N_L_phi_fn(*spectra):
        A_L = A_L_fn(*spectra)
        v = v_fn(L_out, *spectra)
        return np.where(np.isfinite(A_L), A_L * A_L * v, np.inf)

    return estimator, A_L_fn, N_L_phi_fn
