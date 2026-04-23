"""Compare alm_pair[0] (our extract) directly to res1[0] (qe_shear's extract)
with the SAME product map, to isolate if the divergence is in the extraction
or something else."""
import numpy as np
import healpy as hp
from symqe.engine.estimator_native import Pixelization

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1
np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
fT = hp.almxfl(T, 1.0 / ocltt)
px = Pixelization(nside=NSIDE)

from pixell import curvedsky as cs
ells = np.arange(LMAX)
shear_filt = np.sqrt((ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0))

rmapT = px.alm2map(np.stack((T, T)), spin=0, ncomp=1, mlmax=LMAX)[0]
t_alm = cs.almxfl(fT.copy(), shear_filt)
rmap = px.alm2map_spin(np.stack([t_alm, t_alm]), 0, 2, ncomp=2, mlmax=LMAX)
prod_p2 = rmap[0] * rmapT   # M+2 · T
prod_m2 = rmap[1] * rmapT   # M-2 · T

# Bypass the wrapper (which has some dtype issue here) and do map2alm_spin
# directly via (real, imag) pair decomposition — this matches both our
# compile_native's path AND qe_shear's (since (-1)^even = +1 in irot2d).
def map2alm_spin_direct(cmap, lmax, spin_tr):
    m0 = np.ascontiguousarray(-cmap.real, dtype=np.float64)
    m1 = np.ascontiguousarray(cmap.imag, dtype=np.float64)
    return hp.map2alm_spin([m0, m1], spin=spin_tr, lmax=lmax)

res1 = map2alm_spin_direct(prod_p2, LMAX, 2)
shear_alm = -2.0 * res1[0]

# our compile_native-like: same prod_p2, map2alm_spin with spin_alm=0
alm_pair_us = map2alm_spin_direct(prod_p2, LMAX, 2)

# Compare res1[0] vs alm_pair_us[0]
def ratio(gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"{'L':>4s} {'|gen|/|ref|':>12s} {'signed':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            print(f"{L:>4d} {np.sqrt(gc[L]/rc[L]):>12.5e} {xc[L]/rc[L]:>14.5e}")

print("alm_pair_us[0] (our map2alm_spin, spin_alm=0) vs res1[0] (qe_shear spin_alm=2):")
ratio(alm_pair_us[0], res1[0])

print("\nalm_pair_us[0] vs shear_alm_ref (=-2·res1[0]):")
ratio(alm_pair_us[0], shear_alm)

# Try with BOTH prod_p2 and prod_m2 combined (maybe both spin-halves contribute?)
print("\nUsing prod_p2 + prod_m2.conj() (attempt to sum both halves):")
combined = prod_p2 + prod_m2.conj()
alm_pair_both = map2alm_spin_direct(combined, LMAX, 2)
ratio(alm_pair_both[0], shear_alm)

# Try with prod_m2 alone
print("\nUsing prod_m2 alone:")
alm_pair_m2 = map2alm_spin_direct(prod_m2, LMAX, 2)
ratio(alm_pair_m2[0], shear_alm)

# What about res1 with prod_p2 (standard) — is it the same as alm_pair_us[0]?
print("\nres1[0] (qe_shear, spin_alm=2) vs alm_pair_us[0] (ours, spin_alm=0), element-wise:")
diff = res1[0] - alm_pair_us[0]
print(f"  max |diff| = {np.max(np.abs(diff)):.3e}, max |res1| = {np.max(np.abs(res1[0])):.3e}")
print(f"  ratio of rms: {np.sqrt(np.mean(np.abs(alm_pair_us[0])**2)) / np.sqrt(np.mean(np.abs(res1[0])**2)):.5f}")
