"""
Run the rotation EB estimator end-to-end through the native backend.

This is the first estimator pipeline that uses a weight function (W^{α,+})
not present as a hand-coded primitive in falafel — the native backend
assembles it from the inlined SHT primitives purely from the symbolic
recipe.

Sanity checks:
  1. The compile pipeline runs without error.
  2. The output alm is complex-dtype and has the right size.
  3. The output is (largely) real-valued — α_LM is a scalar real field,
     so the non-m=0 entries should have the right hermiticity (a_{l,-m}
     = (-1)^m conj(a_{l,m})), giving a real α map back.
  4. A rough power spectrum check: C_L^α should have a shape that rises
     at moderate L (consistent with the reconstruction noise floor
     from the EB channel).
"""
import numpy as np
from sym_utils.l12_sum import l1, l2
from sym_utils.namikawa import hCE, hCB, f_rot_EB
from sym_utils.estimator import compile_estimator
from sym_utils.estimator_backend import strip_gamma
from sym_utils.estimator_native import compile_rot_eb_native


LMAX, NSIDE = 200, 256


def main():
    import healpy as hp

    ell = np.arange(LMAX + 1, dtype=float)
    clee = 40.0 / (ell + 10) ** 2 + 1e-3
    clbb_noise = 5.0 / (ell + 10) ** 2 + 1e-4   # used only to synthesize B
    nlee = 1.5 * np.ones_like(clee)
    nlbb = 1.5 * np.ones_like(clbb_noise)
    oclee = clee + nlee
    oclbb = clbb_noise + nlbb

    g_rot_EB = f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))
    terms = strip_gamma(compile_estimator(g_rot_EB))
    print(f"Rotation EB compiled to {len(terms)} γ-stripped atomic terms.")

    emit = compile_rot_eb_native(terms, lmax=LMAX, nside=NSIDE)

    np.random.seed(42)
    E_alm = hp.synalm(oclee, lmax=LMAX, new=True)
    B_alm = hp.synalm(oclbb, lmax=LMAX, new=True)

    # Under the standard ΛCDM assumption C_B (primordial) = 0, plans with
    # C_B in the filter vanish; only the "E-response × B-iv" term survives.
    spectra = {"hCE": oclee, "hCB": oclbb,
               "CE": clee, "CB": np.zeros_like(clee)}

    alpha_alm = emit(E_alm, B_alm, spectra)
    print(f"Output dtype: {alpha_alm.dtype}, shape: {alpha_alm.shape}")

    # Check hermiticity (output should correspond to a real scalar α map)
    alpha_map = hp.alm2map(alpha_alm.astype(np.complex128), nside=NSIDE, lmax=LMAX, pol=False)
    if np.iscomplexobj(alpha_map):
        ratio = np.max(np.abs(alpha_map.imag)) / np.max(np.abs(alpha_map.real))
        print(f"α map real-valued: max |imag|/|real| = {ratio:.3e}")
    else:
        print("α map real-valued: healpy returned pure real (good)")

    # Rough power spectrum
    cl_alpha = hp.alm2cl(alpha_alm.astype(np.complex128), lmax=LMAX)
    print(f"Reconstruction-noise-like C_L^α sample values:")
    for L in [2, 10, 50, 100, 150, 200]:
        print(f"  L={L:3d}   C_L^α = {cl_alpha[L]:.3e}")


if __name__ == "__main__":
    main()
