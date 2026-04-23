"""Test: does hand-coded's _gradient_spin truncate ell=LMAX modes due
to `ells = np.arange(0, mlmax)` producing a length-mlmax (not mlmax+1)
filter?  If so, zeroing x_fl[LMAX] in generic should match."""
import numpy as np
import healpy as hp
from sympy import sympify

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, f_TT
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, fuse_spin_pairs, _alm_to_signed_pair, _eval_atom,
    _gradient_spin,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt = cltt + 1.0
spectra = {"hCT": ocltt, "CT": cltt}
np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)

g = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))
terms = strip_gamma(compile_estimator(g))
plans = fuse_spin_pairs(terms)
plan = plans[0]

x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
grad_l = np.sqrt(ell * (ell + 1))
x_response_fl = np.where(grad_l > 0, x_fl / grad_l, 0.0)
X_resp = hp.almxfl(T, x_response_fl)
grad_hc = _gradient_spin(px, np.stack((X_resp, X_resp)), LMAX, spin=0)

# Generic Mx: zero l<2 and l=LMAX (to test truncation hypothesis)
for description, x_fl_var in [
    ("no zeroing", x_fl.copy()),
    ("l<2 zeroed", (lambda f: (f.__setitem__(slice(0, 2), 0), f)[1])(x_fl.copy())),
    ("l<2 and l=LMAX zeroed", (lambda f: (f.__setitem__(slice(0, 2), 0),
                                           f.__setitem__(LMAX, 0), f)[2])(x_fl.copy())),
]:
    pair = _alm_to_signed_pair(
        px, np.stack([hp.almxfl(T, x_fl_var), hp.almxfl(T, x_fl_var)]),
        spin_alm_in=0, abs_spin_out=1, lmax=LMAX)
    Mx = pair[0]
    diff = np.max(np.abs(Mx - grad_hc))
    med = np.median(np.abs(Mx - grad_hc))
    rel = diff / np.max(np.abs(grad_hc))
    print(f"  {description:35s}  max|Mx-grad_hc|={diff:.3e}  med={med:.3e}  rel={rel:.3e}")
