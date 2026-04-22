"""
Bit-for-bit validation of the symbolic TT estimator emitter against
falafel.qe.qe_temperature_only.

Plan:
  1. Build g^{TT} symbolically, compile to a γ-stripped EstimatorTerm list,
     hand to the pixell/falafel emitter.
  2. Generate a synthetic temperature alm (Gaussian from a toy C_ell).
  3. Run both pipelines:
       - falafel.qe.qe_temperature_only(px, filtered_X, filtered_Y, mlmax)
       - the symbolic-compiled closure(X_alm, Y_alm, spectra)
     where the filtered alms feeding falafel are what the symbolic
     recipe IMPLIES (inverse-variance and response-weighting baked in).
  4. Compare the output alms directly.
"""
import numpy as np
from sympy import sympify

from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCT, f_TT
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma, compile_tt

# --- parameters ---
LMAX  = 200
NSIDE = 256


def make_test_cl(lmax):
    """Toy C_ell^{TT} (scale of a flat-ish CMB)."""
    ell = np.arange(lmax + 1, dtype=float)
    cltt = 5e3 / (ell + 10) ** 2 + 1e-2
    nltt = 1.0 * np.ones_like(cltt)
    return cltt, nltt


def main():
    import healpy as hp
    from falafel.qe import pixelization, qe_temperature_only, filter_alms

    cltt, nltt = make_test_cl(LMAX)
    ocltt = cltt + nltt

    # --- compile the symbolic estimator ---
    g_TT = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
    terms = compile_estimator(g_TT)
    terms = strip_gamma(terms)
    sym_fn = compile_tt(terms, lmax=LMAX, nside=NSIDE)

    # --- synthetic alm ---
    np.random.seed(42)
    T_alm = hp.synalm(ocltt, lmax=LMAX, new=True)

    # --- falafel reference ---
    px = pixelization(nside=NSIDE)
    # falafel's qe_temperature_only takes (X, Y) that have already been
    # filtered: X = response-weighted, Y = inverse-variance-filtered,
    # or symmetric variants.  We replicate the conventional split:
    X_falafel = filter_alms(T_alm.copy(), cltt / ocltt)   # response-weighted
    Y_falafel = filter_alms(T_alm.copy(), 1.0 / ocltt)    # inverse-variance
    phi_falafel = qe_temperature_only(px, X_falafel, Y_falafel, LMAX)
    # qe_temperature_only returns (phi, curl) stacked; take gradient mode
    if phi_falafel.ndim == 2:
        phi_falafel = phi_falafel[0]

    # --- symbolic: pass raw T_alm and let the compiled filters do their thing ---
    # The spectra dict gives the compiled closure access to hCT, CT by name.
    spectra = {"hCT": ocltt, "CT": cltt}
    phi_sym = sym_fn(T_alm, T_alm, spectra)

    # --- compare ---
    print(f"lmax={LMAX}, nside={NSIDE}")
    print(f"falafel phi_alm:  {phi_falafel.shape}  dtype={phi_falafel.dtype}")
    print(f"symbolic phi_alm: {phi_sym.shape}  dtype={phi_sym.dtype}")
    # bring to common size / dtype if needed
    n = min(len(phi_falafel), len(phi_sym))
    a = phi_falafel[:n]
    b = phi_sym[:n]

    # show a few entries
    print("\nFirst 10 entries (falafel, symbolic, ratio):")
    for i in range(10):
        r = b[i] / a[i] if a[i] != 0 else np.nan
        print(f"  {i:3d}  {a[i]:+.4e}  {b[i]:+.4e}  ratio={r}")

    denom = np.abs(a)
    mask = denom > 0
    reldiff = np.abs(a[mask] - b[mask]) / denom[mask]
    print(f"\nMax |reldiff|:   {reldiff.max():.3e}")
    print(f"Median |reldiff|: {np.median(reldiff):.3e}")


if __name__ == "__main__":
    main()
