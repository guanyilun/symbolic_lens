"""Unified compilation of a symbolic QE weight g into BOTH the estimator
and the Hu-Okamoto normalization A_L, from one symbolic expression.

Given a symbolic weight ``g(l1, L, l2)`` = f/(Δ · hCxx(l1) · hCyy(l2)),
this wires two existing pipelines:

  * ``compile_estimator → compile_native``  →  estimator fn(X, Y, spec)
  * ``NormCompiler``                         →  A_L^{-1} fn(L, *spectra)

The normalization integrand is derived directly from ``g``:
   A_L^{-1}(L) = (1/(2L+1)) · sum_{l1,l2} g² · Δ · hCxx(l1) · hCyy(l2)
(using the identity ``f · g = g² · Δ · hCxx · hCyy`` for real g).

For ζ^- estimators (EB, TB) the user can pass ``conjugate_g`` that
flips I → -I on one copy before multiplying.
"""
import numpy as np
from sympy import cancel, Symbol, Function

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
    rlmax = rlmax if rlmax is not None else lmax

    # --- Estimator side ---
    terms = strip_gamma(compile_estimator(g_symbolic))
    estimator = compile_native(
        terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs
    )

    # --- Normalization side ---
    # A_L^{-1}:  (1/(2L+1)) · sum f·g  where f = g · Δ · hCxx · hCyy
    # So A_L^{-1} integrand = (1/(2L+1)) · g² · Δ · hCxx(l1) · hCyy(l2)
    integrand_AL_inv = g_symbolic * g_symbolic * Delta * hCxx(l1) * hCyy(l2) / (2 * l + 1)
    integrand_AL_inv = cancel(integrand_AL_inv)

    # Noise variance per mode:  (1/(2L+1)) · sum g² · hCxx(l1) · hCyy(l2)
    integrand_v = g_symbolic * g_symbolic * hCxx(l1) * hCyy(l2) / (2 * l + 1)
    integrand_v = cancel(integrand_v)

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
