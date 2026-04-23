"""Verify whether grad(M_-1·Y) == grad(M_+1·Y) ACROSS ALL (L, m), not
only m=0.  The original probe relied on linearity + sum/diff to
conclude this, but maybe the curl=0 and grad=0 it showed was also
only at m=0."""
import numpy as np
import healpy as hp

from symqe.engine.estimator_native import Pixelization, _alm_to_signed_pair

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
np.random.seed(0)
X = hp.synalm(cltt, lmax=LMAX, new=True)
Y = hp.synalm(cltt, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

X_pair = np.stack([X, X])
M_plus, M_minus = _alm_to_signed_pair(px, X_pair, spin_alm_in=0,
                                      abs_spin_out=1, lmax=LMAX)
Y_map = px.alm2map(Y, spin=0, ncomp=1, mlmax=LMAX)[0]

prod_plus = M_plus * Y_map
prod_minus = M_minus * Y_map

a_p = np.asarray(px.map2alm_spin(prod_plus, lmax=LMAX, spin_alm=0, spin_transform=1))
a_m = np.asarray(px.map2alm_spin(prod_minus, lmax=LMAX, spin_alm=0, spin_transform=1))

# max|grad(M_+1·Y) - grad(M_-1·Y)|
print(f"max|grad(M_+·Y) - grad(M_-·Y)| = {np.max(np.abs(a_p[0] - a_m[0])):.3e}")
print(f"max|grad(M_+·Y) + grad(M_-·Y)| = {np.max(np.abs(a_p[0] + a_m[0])):.3e}")
print(f"max|curl(M_+·Y) + curl(M_-·Y)| = {np.max(np.abs(a_p[1] + a_m[1])):.3e}")
print(f"max|curl(M_+·Y) - curl(M_-·Y)| = {np.max(np.abs(a_p[1] - a_m[1])):.3e}")

print(f"\nmax|grad(M_+·Y)| = {np.max(np.abs(a_p[0])):.3e}")
print(f"max|curl(M_+·Y)| = {np.max(np.abs(a_p[1])):.3e}")

# so: is grad even under sX flip?  is curl odd?
