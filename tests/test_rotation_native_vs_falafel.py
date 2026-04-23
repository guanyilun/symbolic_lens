"""
Bit-for-bit validation: native CMB-rotation EB estimator vs
``falafel.qe.qe_rot``, on CAR geometry.

Both pipelines receive:
  - identical EE / BB spectra and dummy noise,
  - identical synthetic E/B alms from a fixed seed,
  - identical CAR shape + WCS,

and should produce the same α_LM to machine precision, up to the overall
pair_coeff that bookkeeps Δ, γ residual, signs, and the ±ζ⁻ pair fusion.
"""
import numpy as np
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, hCB, f_rot_EB
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import compile_rot_eb_native, Pixelization


LMAX = 200


def main():
    from pixell import enmap, curvedsky as cs
    from falafel.qe import qe_rot, pixelization as falafel_pix, filter_alms
    import healpy as hp

    # Full-sky CAR geometry at ~12 arcmin resolution (sufficient for lmax=200)
    res = np.deg2rad(12.0 / 60.0)
    shape, wcs = enmap.fullsky_geometry(res=res)

    ell = np.arange(LMAX + 1, dtype=float)
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb_for_synth = 5.0 / (ell + 10) ** 2 + 1e-4
    nlee = 1.5 * np.ones_like(clee)
    nlbb = 1.5 * np.ones_like(clbb_for_synth)
    oclee = clee + nlee
    oclbb = clbb_for_synth + nlbb

    # Synthetic alms
    np.random.seed(42)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True).astype(np.complex128)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True).astype(np.complex128)

    # --- Symbolic native pipeline on CAR ---
    g_rot_EB = f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g_rot_EB))
    px_native = Pixelization(shape=shape, wcs=wcs)
    emit = compile_rot_eb_native(terms, lmax=LMAX, px=px_native)
    # Rotation reconstruction assumes primordial C_B = 0
    spectra = {"hCE": oclee, "hCB": oclbb, "CE": clee, "CB": np.zeros_like(clee)}
    alpha_native = emit(E_alm, B_alm, spectra)

    # --- Falafel reference ---
    px_falafel = falafel_pix(shape=shape, wcs=wcs)
    response_cls = {"EE": clee}
    # falafel.qe_rot expects iv-filtered E and B alms
    fE = filter_alms(E_alm.copy(), 1.0 / oclee)
    fB = filter_alms(B_alm.copy(), 1.0 / oclbb)
    alpha_falafel = qe_rot(px_falafel, response_cls, LMAX, fE, fB)

    # --- Compare ---
    n = min(len(alpha_native), len(alpha_falafel))
    a = np.asarray(alpha_native[:n])
    b = np.asarray(alpha_falafel[:n])
    mask = np.abs(b) > 1e-10
    ratios = a[mask] / b[mask]
    median_abs = np.median(np.abs(ratios))
    median_complex = np.median(ratios.real) + 1j * np.median(ratios.imag)
    std_rel = np.std(np.abs(ratios)) / np.mean(np.abs(ratios))

    print(f"CAR geometry: shape={shape}, lmax={LMAX}")
    print(f"First 5 entries (falafel, native):")
    for i in range(5):
        if abs(b[i]) > 1e-10:
            print(f"  {i}  fal={b[i]:+.4e}  nat={a[i]:+.4e}")
    print(f"\nNative / Falafel ratio:")
    print(f"  median |ratio|: {median_abs:.6e}")
    print(f"  median complex: {median_complex}")
    print(f"  std/mean:       {std_rel:.3e}")


if __name__ == "__main__":
    main()
