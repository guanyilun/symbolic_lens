"""Figures of merit for QE candidate ranking.

``compile_qe`` returns ``(estimator, A_L_fn, N_L_phi_fn)``.  Naively
summing ``(2L+1)/N_L^phi`` is gauge-dependent: under ``g → α·g`` the
derived ``f = g·Δ·hC·hC`` scales as α too, both ``A_L`` and the noise
``v`` scale as α², and their combination ``N_L^phi = A_L²·v`` scales
as ``1/α²`` — so the raw Fisher sum scales as α² and is maximized by
choosing an arbitrarily large α.

``fisher_fom`` removes the α² scaling by multiplying the raw Fisher
by ``A_L(L_ref)`` at a chosen reference L:

    FOM = A_L(L_ref) · Σ_{L ∈ L_range} (2L+1) / N_L^phi(L)

Under ``g → α·g`` (with f derived), ``A_L → A_L/α²`` and
``Σ 1/N → α²·Σ 1/N``; the two α² factors cancel, so FOM is
invariant under a pure rescaling of the weight.

Caveats
-------
* FOM is only a valid comparator when all candidates describe the
  SAME physical target.  For candidates with different functional
  structure in f, the derived "physical response" differs between
  candidates, and Fisher alone cannot tell "good reconstruction of
  the wrong thing" from "good reconstruction of the right thing" —
  that needs either an oracle (cross-correlation with a reference
  signal) or the ``compile_qe_from_f`` path where you provide an
  explicit target f and search only over structural variants of it.
* For reliable ranking, pick ``L_ref`` inside the SNR-rich band
  where ``A_L(L_ref)`` is well-conditioned (typically mid-range L,
  avoiding both the L→0 divergence and the noise-dominated tail).
"""
from __future__ import annotations
import numpy as np


def fisher_fom(N_L_phi, A_L, *, L_range, L_ref):
    """Gauge-fixed Fisher figure of merit for a QE candidate.

    Parameters
    ----------
    N_L_phi : 1D array
        Reconstruction noise power ``N_L^phi`` at each L.  Typically
        obtained by calling the ``N_L_phi_fn`` returned by
        ``compile_qe`` on a spectrum tuple.
    A_L : 1D array
        Hu-Okamoto normalization at each L (from ``A_L_fn``).
    L_range : iterable of int
        L modes to include in the Fisher sum.
    L_ref : int
        Reference L at which to fix the gauge.  Pick a mid-range L
        where ``A_L`` is finite and well-conditioned.

    Returns
    -------
    float
        The gauge-fixed figure of merit.  Higher = more informative.
        Returns ``-np.inf`` if ``A_L[L_ref]`` is non-finite or
        non-positive, or if no valid modes survive in ``L_range``.
    """
    aL_ref = A_L[L_ref]
    if not np.isfinite(aL_ref) or aL_ref <= 0:
        return -np.inf
    Ls = np.asarray(list(L_range), dtype=int)
    valid = np.isfinite(N_L_phi[Ls]) & (N_L_phi[Ls] > 0)
    if not np.any(valid):
        return -np.inf
    raw = np.sum((2.0 * Ls[valid] + 1.0) / N_L_phi[Ls[valid]])
    return float(aL_ref * raw)


def cross_correlation_score(phi_hat_alm, phi_true_alm, *, L_range):
    """Mode-count-weighted cross-correlation coefficient ⟨r(L)⟩.

    ``r(L) = C_L^{phi_hat × phi_true} / sqrt(C_L^{phi_hat} · C_L^{phi_true})``
    averaged over ``L_range`` with weights (2L+1).  Inherently
    scale-invariant (unaffected by phi_hat → α·phi_hat), so this is a
    valid gauge-invariant FOM when an oracle reference for phi exists
    (simulation, cross with external tracer, etc.).

    Parameters
    ----------
    phi_hat_alm : 1D complex healpy alm
        Reconstructed phi (already A_L-normalized).
    phi_true_alm : 1D complex healpy alm
        Reference phi.
    L_range : iterable of int

    Returns
    -------
    float
        ⟨r(L)⟩ in (-1, +1).  Higher = better reconstruction.  Returns
        -inf if both spectra are zero or any spectrum is non-finite.
    """
    import healpy as hp
    Ls = np.asarray(list(L_range), dtype=int)
    cl_h = hp.alm2cl(np.asarray(phi_hat_alm))
    cl_p = hp.alm2cl(np.asarray(phi_true_alm))
    cl_x = hp.alm2cl(np.asarray(phi_hat_alm), np.asarray(phi_true_alm))
    denom = np.sqrt(cl_h * cl_p)
    valid = (denom[Ls] > 0) & np.isfinite(cl_x[Ls]) & np.isfinite(denom[Ls])
    if not np.any(valid):
        return -np.inf
    r = np.zeros_like(cl_x[Ls], dtype=float)
    r[valid] = cl_x[Ls[valid]] / denom[Ls[valid]]
    w = (2.0 * Ls + 1.0)
    return float(np.sum(w[valid] * r[valid]) / np.sum(w[valid]))
