"""Verify: under imap → conj(imap), does map2alm_spin(spin=1)[0]
(the grad mode) stay the same?  Parity argument says yes (grad is
even under parity, which maps (Q,U)→(Q,-U)=conj input)."""
import numpy as np
import healpy as hp

from sym_utils.estimator_native import Pixelization, _alm_to_signed_pair

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
np.random.seed(0)
X = hp.synalm(cltt, lmax=LMAX, new=True)
Y = hp.synalm(cltt, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

# Get M_+1(X) as complex map
X_pair = np.stack([X, X])
M_plus, _ = _alm_to_signed_pair(px, X_pair, spin_alm_in=0, abs_spin_out=1, lmax=LMAX)
Y_map = px.alm2map(Y, spin=0, ncomp=1, mlmax=LMAX)[0]

# Form prod = M_+·Y_real (complex map) and its conjugate
prod = M_plus * Y_map
prod_conj = np.conj(prod)

# Run map2alm_spin(spin=1) on each
a1 = np.asarray(px.map2alm_spin(prod, lmax=LMAX, spin_alm=0, spin_transform=1))
a2 = np.asarray(px.map2alm_spin(prod_conj, lmax=LMAX, spin_alm=0, spin_transform=1))

print(f"max|grad(prod) - grad(conj(prod))| = {np.max(np.abs(a1[0]-a2[0])):.3e}")
print(f"max|curl(prod) + curl(conj(prod))| = {np.max(np.abs(a1[1]+a2[1])):.3e}")
print(f"max|grad(prod)|                   = {np.max(np.abs(a1[0])):.3e}")

# prediction: grad invariant → first diff ≈ 0; curl flips sign → second sum ≈ 0

# Also: how does map2alm_spin relate imap vs conj(imap) via the internal
# _irot2d?  Trace carefully.
dmap1 = -np.stack((np.real(prod), np.imag(prod)))
dmap2 = -np.stack((np.real(prod_conj), np.imag(prod_conj)))
print(f"\nInternal (Q,U) comparison:")
print(f"max|dmap1[0] - dmap2[0]| = {np.max(np.abs(dmap1[0]-dmap2[0])):.3e}")
print(f"max|dmap1[1] + dmap2[1]| = {np.max(np.abs(dmap1[1]+dmap2[1])):.3e}")
# These MUST be 0: (Q, U) of conj(imap) is (Q, -U).  So dmap1[0]==dmap2[0], dmap1[1]==-dmap2[1].

# Now explicitly test: hp.map2alm_spin((Q, U)) vs hp.map2alm_spin((Q, -U))
# does the grad mode stay the same?
a1_raw = np.asarray(hp.map2alm_spin(
    np.asarray([np.real(prod), np.imag(prod)], dtype=np.float64),
    lmax=LMAX, spin=1))
a2_raw = np.asarray(hp.map2alm_spin(
    np.asarray([np.real(prod), -np.imag(prod)], dtype=np.float64),
    lmax=LMAX, spin=1))
print(f"\nhp.map2alm_spin directly on (Q,U) vs (Q,-U):")
print(f"max|grad_raw_1 - grad_raw_2| = {np.max(np.abs(a1_raw[0]-a2_raw[0])):.3e}")
print(f"max|curl_raw_1 + curl_raw_2| = {np.max(np.abs(a1_raw[1]+a2_raw[1])):.3e}")
