"""Quantify the falafel int-truncation error on the polarization ladder
factors, out to lmax=3000 (realistic CMB lensing lmax).

Two levels:
  (a) The raw per-l ratio int/float of the ladder factor.
  (b) How that propagates into the reconstructed phi_LM when the filter
      is applied to a CMB-like spectrum (EE signal).
"""
import numpy as np


def ladder_ratios(lmax):
    """Return (ells, ratio_raising, ratio_lowering) where
    ratio = int_truncated / true_float."""
    ells = np.arange(0, lmax + 1)
    raising_f = np.zeros(lmax + 1, dtype=float)
    raising_i = np.zeros(lmax + 1, dtype=float)
    m = ells >= 2
    raising_f[m] = np.sqrt((ells[m] - 2.0) * (ells[m] + 3.0))
    raising_i[m] = np.floor(raising_f[m])   # int truncation

    lowering_f = np.zeros(lmax + 1, dtype=float)
    lowering_i = np.zeros(lmax + 1, dtype=float)
    m = ells >= 1
    lowering_f[m] = np.sqrt((ells[m] - 1.0) * (ells[m] + 2.0))
    lowering_i[m] = np.floor(lowering_f[m])

    # Avoid division by zero for undefined l's
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(raising_f > 0, raising_i / raising_f, 1.0)
        rl = np.where(lowering_f > 0, lowering_i / lowering_f, 1.0)
    return ells, rr, rl


def summary_table(lmax):
    ells, rr, rl = ladder_ratios(lmax)
    print(f"\n=== Ladder factor int/float ratio (relative error = 1 - ratio) ===")
    print(f"{'l':>5s} {'raise ratio':>12s} {'rel err %':>10s}"
          f"  {'lower ratio':>12s} {'rel err %':>10s}")
    for L in [2, 3, 5, 10, 30, 100, 300, 1000, 3000]:
        if L > lmax:
            continue
        print(f"{L:>5d} {rr[L]:>12.8f} {100*(1-rr[L]):>9.4f}%"
              f"  {rl[L]:>12.8f} {100*(1-rl[L]):>9.4f}%")

    # Global stats over the valid range
    valid_r = rr[3:]  # raising is defined for l>=2, but ratio at l=2 is 0/0 → handled as 1
    valid_l = rl[2:]
    err_r = 1 - valid_r
    err_l = 1 - valid_l
    print(f"\n  raising  ladder error: max = {100*err_r.max():.3f}% (at l={3+err_r.argmax()})"
          f", rms = {100*np.sqrt((err_r**2).mean()):.5f}%")
    print(f"  lowering ladder error: max = {100*err_l.max():.3f}% (at l={2+err_l.argmax()})"
          f", rms = {100*np.sqrt((err_l**2).mean()):.5f}%")


def propagated_error(lmax):
    """Mock the error on a phi reconstruction.  The ladder factor
    multiplies the input alm at l.  Effective output error at L is
    approximately the signal-weighted mean of (1 - ratio(l))^2 over
    contributing modes.  For CMB-lensing-EE the signal peaks around
    l~1000-2000.  Give a rough CMB-weighted bound."""
    ells, rr, rl = ladder_ratios(lmax)
    # CMB-EE-like power spectrum (very rough): peak around l~1000, falls as l^-2
    ell = ells.astype(float)
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    # Weight by l*(l+1)*Clee/(2 pi) — number of modes per l is 2l+1,
    # so effective weighting on mode count is (2l+1)*Clee.
    w = (2 * ell + 1) * clee
    w[0] = 0

    # Expected fractional error squared, weighted by signal
    err_r = (1 - rr)
    err_l = (1 - rl)
    wr = (w * err_r**2).sum() / w.sum()
    wl = (w * err_l**2).sum() / w.sum()
    wr_rms = np.sqrt(wr)
    wl_rms = np.sqrt(wl)
    print(f"\n=== Signal-weighted RMS fractional error on filtered alm ===")
    print(f"  (weights = (2l+1)·C_EE, so mode-count × power)")
    print(f"  raising  (|s|=3 leg): {100*wr_rms:.4f}%")
    print(f"  lowering (|s|=1 leg): {100*wl_rms:.4f}%")
    print(f"\nThese are the *per-leg* fractional errors on the alm that goes")
    print(f"into the pixel product.  Output phi_LM scatter scales roughly with")
    print(f"this; power spectrum N_L^phi scatter is ~2x the alm scatter.")


for lmax in [200, 3000]:
    print(f"\n{'#'*60}")
    print(f"# lmax = {lmax}")
    print(f"{'#'*60}")
    summary_table(lmax)
    propagated_error(lmax)
