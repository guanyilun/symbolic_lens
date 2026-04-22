"""
estimator_backend.py — pixell/falafel-backed SHT emitter for EstimatorTerm lists.

Takes the output of ``compile_estimator`` (see estimator.py), strips the
γ-normalization factors that pixell's SHT normalization reproduces
implicitly, and produces a numpy callable that executes the recipe on
input alm arrays.

Design note: rather than reinventing signed-spin bookkeeping we use
falafel's ``pixelization`` class as the SHT primitive layer, which
gives us tested conventions for (alm2map_spin / map2alm_spin / rot2d).
This means falafel is a runtime dependency of the emitter; but the
symbolic engine itself (sym_utils/{atom, sympy_bridge, normal_form,
quotient, l12_sum, estimator, namikawa}.py) is falafel-free.
"""
from __future__ import annotations
import numpy as np

from .atom import AtomicFactor, Const, Var, Pow, Mul, Add, FuncApp, ONE
from .atom import mul as atom_mul
from .estimator import EstimatorTerm


# --------------------------------------------------------------------------
# γ-stripping.

def strip_gamma(terms: list[EstimatorTerm]) -> list[EstimatorTerm]:
    """Remove sqrt(2*l+1), sqrt(2*l1+1), sqrt(2*l2+1) factors from the
    (L_factor, X_filter, Y_filter) of each term respectively.  Those
    come from splitting γ_{l1 L l2} = sqrt((2l1+1)(2L+1)(2l2+1)/(4π))
    across the three legs at W-construction time — pixell's SHT
    normalization puts them back implicitly, so they must not be
    applied as explicit filters."""
    return [
        EstimatorTerm(
            coeff    = t.coeff,
            L_factor = _strip_sqrt_2var_plus_1(t.L_factor, 'l'),
            X_filter = _strip_sqrt_2var_plus_1(t.X_filter, 'l1'),
            Y_filter = _strip_sqrt_2var_plus_1(t.Y_filter, 'l2'),
            spin_L=t.spin_L, spin_X=t.spin_X, spin_Y=t.spin_Y,
        )
        for t in terms
    ]


def _strip_sqrt_2var_plus_1(atom: AtomicFactor, var_name: str) -> AtomicFactor:
    """Remove Pow(Add(Const(1), Mul(Const(2), Var(var_name))), 1/2) from a
    Mul.  If the atom IS that target, returns ONE."""
    from sympy import Rational
    half = Rational(1, 2)

    def is_target(f):
        if not isinstance(f, Pow):    return False
        if f.exp != half:             return False
        base = f.base
        if not isinstance(base, Add): return False
        if len(base.terms) != 2:      return False
        has_one = any(isinstance(t, Const) and t.value == 1 for t in base.terms)
        def is_two_var(t):
            if not isinstance(t, Mul):        return False
            if len(t.factors) != 2:           return False
            has_two = any(isinstance(f_, Const) and f_.value == 2 for f_ in t.factors)
            has_var = any(isinstance(f_, Var)   and f_.name  == var_name for f_ in t.factors)
            return has_two and has_var
        return has_one and any(is_two_var(t) for t in base.terms)

    if isinstance(atom, Mul):
        kept = [f for f in atom.factors if not is_target(f)]
        return atom_mul(*kept) if kept else ONE
    return ONE if is_target(atom) else atom


# --------------------------------------------------------------------------
# evaluate an atomic filter to an ell-indexed numpy array.

def _atom_to_array(atom: AtomicFactor, ell: np.ndarray, spectra: dict) -> np.ndarray:
    """Evaluate an atomic factor depending on at most one of {l, l1, l2}
    to a 1-D array over ells.  Values at ell=0,1 where the expression
    diverges (sqrt of negative arg, 1/0) are set to 0."""
    fv = atom.free_vars()
    if not fv:
        val = complex(atom.evaluate({}))
        return np.full_like(ell, val, dtype=complex)

    assert len(fv) == 1, f"atom has multiple free vars: {fv}"
    (var_name,) = fv
    env = dict(spectra)
    env[var_name] = ell.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        arr = np.asarray(atom.evaluate(env))
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


# --------------------------------------------------------------------------
# TT emitter using falafel's pixelization primitives.

def compile_tt(terms: list[EstimatorTerm], lmax: int,
               *, nside: int = 2048):
    """Compile a γ-stripped list of EstimatorTerms for TT (spin-0 alms)
    into a callable ``(X_alm, Y_alm, spectra_dict) -> phi_alm``.

    Only supports spin-0 input alms (temperature).  Polarization (spin-2
    inputs with the E±iB pair) is deferred to milestone 2.
    """
    from falafel.qe import pixelization
    import healpy as hp

    px = pixelization(nside=nside)

    def compiled(X_alm, Y_alm, spectra):
        X_alm = np.asarray(X_alm, dtype=np.complex128)
        Y_alm = np.asarray(Y_alm, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)
        out = None

        for t in terms:
            x_fl = _atom_to_array(t.X_filter, ell, spectra)
            y_fl = _atom_to_array(t.Y_filter, ell, spectra)
            L_fl = _atom_to_array(t.L_factor, ell, spectra)

            x_alm_f = hp.almxfl(X_alm, x_fl.real.astype(np.float64))
            y_alm_f = hp.almxfl(Y_alm, y_fl.real.astype(np.float64))

            # alm2map for each leg; signed-spin means take +|s| or -|s| component
            x_map = _alm_to_signed_spin_map(px, x_alm_f, t.spin_X, lmax)
            y_map = _alm_to_signed_spin_map(px, y_alm_f, t.spin_Y, lmax)

            prod = x_map * y_map

            term_alm = _signed_spin_map_to_alm(px, prod, t.spin_L, lmax)
            term_alm = hp.almxfl(term_alm, L_fl.real.astype(np.float64))

            c = complex(t.coeff)
            term_alm = c * term_alm
            out = term_alm if out is None else out + term_alm

        return out

    return compiled


def _alm_to_signed_spin_map(px, alm, spin, lmax):
    """Return the signed-spin-s real-space map as a complex array.

    px.alm2map_spin returns (M_+|s|, M_-|s|) — we pick one.
    For spin=0 we use px.alm2map directly.
    """
    if spin == 0:
        return px.alm2map(alm, spin=0, ncomp=1, mlmax=lmax)[0].astype(np.complex128)
    # input alm pair for a spin-0 source field: (alm, alm) (both ±0 components)
    pair = np.stack([alm, alm])
    maps = px.alm2map_spin(pair, spin_alm=0, spin_transform=abs(spin),
                           ncomp=2, mlmax=lmax)
    # maps = (M_+|s|, M_-|s|) as complex
    return maps[0] if spin > 0 else maps[1]


def _signed_spin_map_to_alm(px, cmap, spin, lmax):
    """Inverse of the above for the output leg."""
    if spin == 0:
        m = cmap.real if np.iscomplexobj(cmap) else cmap
        return px.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)
    # px.map2alm_spin takes a complex map and outputs (a_+|s|, a_-|s|)
    # (It internally builds imap.conj() for the second component.)
    res = px.map2alm_spin(cmap, lmax=lmax, spin_alm=0, spin_transform=abs(spin))
    # res shape: (2, ...) — the ±|s| alm pair
    return res[0] if spin > 0 else res[1]
