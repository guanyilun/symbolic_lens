"""
namikawa.py — symbolic Wigner-weight and spectrum definitions following
Toshiya Namikawa's CMB QE convention (see README.org § "Quadratic Estimator
Theory").  Shared by the normalization and estimator compilers.

Weights (Namikawa Eqs. 38 / 62), brought to l-first 3j ordering via the
column-permutation identity
    w3j(l1, L, l2; ...) = P · w3j(L, l1, l2; ...')
so that they're consumable by our symbolic engine (which groups by L as
the first 3j column — see l12_sum.py). The column swap introduces an
explicit P factor that reduces under P² = 1 during compilation.

This module is deliberately free of numerical dependencies; it only
produces sympy expressions.
"""
from sympy import Function, sqrt, pi, I

from .l12_sum import wigner_3j, l, l1, l2, P


# ------------------------------------------------------------------ spectra

# Observed (signal + noise) spectra — caller passes arrays at evaluation time.
hCT  = Function("hCT")      # \hat C_\ell^{TT}
hCE  = Function("hCE")      # \hat C_\ell^{EE}
hCB  = Function("hCB")      # \hat C_\ell^{BB}
hCTE = Function("hCTE")     # \hat C_\ell^{TE}

# Signal / response spectra (unlensed or lensed, depending on use).
CT  = Function("CT")
CE  = Function("CE")
CB  = Function("CB")
CTE = Function("CTE")


# ---------------------------------------------------------- scalar helpers

def gamma_f(j1, j2, j3):
    """γ_{j1 j2 j3} = sqrt((2j1+1)(2j2+1)(2j3+1)/(4π)).  Absorbed by SHT
    normalization in the emit stage — see estimator.py docstring."""
    return sqrt((2*j1 + 1) * (2*j2 + 1) * (2*j3 + 1) / (4*pi))

def q_plus(c):  return c * (1 + P) / 2       # q^{x,+}
def q_minus(c): return c * (1 - P) / 2       # q^{x,-}

def a(ell, s):       return -sqrt((ell - s) * (ell + s + 1) / 2)   # Namikawa spin-ladder
def a_plus(ell):     return a(ell, 2)        # a^+
def a_minus(ell):    return a(ell, -2)       # a^-

# Namikawa's ζ^± (see his Sec. 2 / tempura.jl).  For lensing (φ, ϖ) we use
# c_φ = 1, so ζ^+ = 1 and ζ^- = i.  The imaginary unit on ζ^- encodes the
# E↔B mixing picked up by the backend.
ZETA_PLUS  = 1
ZETA_MINUS = I


# ---------------------------------------------------------------- weights

def W_lens_0(l1_, l_out, l2_, c=1):
    """Namikawa Eq. 38 (lensing temperature weight).

    3j column-swapped into l-first form:
        w3j(l1, L, l2; 0, 1, -1) = P · w3j(L, l1, l2; 1, 0, -1).
    """
    return -2 * a(l_out, 0) * a(l2_, 0) * q_plus(c) * gamma_f(l1_, l_out, l2_) * \
           P * wigner_3j(l_out, l1_, l2_, 1, 0, -1)


def W_lens_p(l1_, l_out, l2_, c=1):
    """Namikawa Eq. 62 — W^{x,+} (even-parity polarization weight)."""
    t1 = a_plus(l2_)       * wigner_3j(l_out, l1_, l2_,  1,  2, -3)
    t2 = c**2 * a_minus(l2_) * wigner_3j(l_out, l1_, l2_, -1,  2, -1)
    return -ZETA_PLUS * q_plus(c) * gamma_f(l1_, l_out, l2_) * a(l_out, 0) * \
           P * (t1 + t2)


def W_lens_m(l1_, l_out, l2_, c=1):
    """Namikawa Eq. 62 — W^{x,-} (odd-parity polarization weight).

    Differs from W^{x,+} by q^+ → q^-  and  ζ^+ → ζ^-.  The ζ^- = i factor
    is carried through symbolically; see estimator.py docstring."""
    t1 = a_plus(l2_)       * wigner_3j(l_out, l1_, l2_,  1,  2, -3)
    t2 = c**2 * a_minus(l2_) * wigner_3j(l_out, l1_, l2_, -1,  2, -1)
    return -ZETA_MINUS * q_minus(c) * gamma_f(l1_, l_out, l2_) * a(l_out, 0) * \
           P * (t1 + t2)


# ---------------------------------------------------------- f^{XY} builders
#
# Namikawa's f-weights for each estimator (README.org table, Sec. 3.5):
#
#    f^{x,(TT)}_{l L l'} = W^{x,0}_{l L l'} · C_T^{l'}  +  p_x · W^{x,0}_{l' L l} · C_T^l
#    f^{x,(TE)}_{l L l'} = W^{x,0}_{l L l'} · C_TE^{l'}  +  p_x · W^{x,+}_{l' L l} · C_TE^l
#    f^{x,(TB)}_{l L l'} =                                   p_x · W^{x,-}_{l' L l} · C_TE^l
#    f^{x,(EE)}_{l L l'} = W^{x,+}_{l L l'} · C_E^{l'}  +  p_x · W^{x,+}_{l' L l} · C_E^l
#    f^{x,(EB)}_{l L l'} = W^{x,-}_{l L l'} · C_B^{l'}  +  p_x · W^{x,-}_{l' L l} · C_E^l
#    f^{x,(BB)}_{l L l'} = W^{x,+}_{l L l'} · C_B^{l'}  +  p_x · W^{x,+}_{l' L l} · C_B^l
#
# p_x = +1 for even-parity fields (φ, ε); p_x = −1 for odd-parity (ϖ, α).
# Δ^{XX} = 2, Δ^{TB} = Δ^{EB} = 1  (g = f* / (Δ · \hat C^{l} · \hat C^{l'})).

def f_TT(px=+1):
    return W_lens_0(l1, l, l2) * CT(l2) + px * W_lens_0(l2, l, l1) * CT(l1)

def f_TE(px=+1):
    return W_lens_0(l1, l, l2) * CTE(l2) + px * W_lens_p(l2, l, l1) * CTE(l1)

def f_TB(px=+1):
    return px * W_lens_m(l2, l, l1) * CTE(l1)

def f_EE(px=+1):
    return W_lens_p(l1, l, l2) * CE(l2) + px * W_lens_p(l2, l, l1) * CE(l1)

def f_EB(px=+1):
    return W_lens_m(l1, l, l2) * CB(l2) + px * W_lens_m(l2, l, l1) * CE(l1)

def f_BB(px=+1):
    return W_lens_p(l1, l, l2) * CB(l2) + px * W_lens_p(l2, l, l1) * CB(l1)


# ======================================================================
# CMB rotation (anisotropic birefringence, α) distortion weights.
# Symlens.jl examples/tempura.jl reference:
#   Wₐ⁺(ℓ₁,ℓ₂,ℓ₃,c) = 2·i·ζ⁺·q⁻(c)·γ·w3j(ℓ₁,ℓ₂,ℓ₃; 2, 0, -2)
#   Wₐ⁻(ℓ₁,ℓ₂,ℓ₃,c) = 2·i·ζ⁻·q⁺(c)·γ·w3j(ℓ₁,ℓ₂,ℓ₃; 2, 0, -2)
# Rotation is an odd-parity distortion (p_α = −1).  The 2·i prefactor is
# preserved symbolically; the native backend will carry the i through the
# coefficient and the rot2d/irot2d pair will resolve it in the E/B split.
#
# 3j column-permutation identity for canonical l-first ordering:
#   w3j(l1, L, l2; 2, 0, -2) = P · w3j(L, l1, l2; 0, 2, -2)
# ======================================================================

def W_rot_p(l1_, l_out, l2_, c=1):
    """Namikawa Eq. (rotation) — W^{α,+} (even-parity polarization mixing)."""
    return 2 * I * ZETA_PLUS * q_minus(c) * gamma_f(l1_, l_out, l2_) * \
           P * wigner_3j(l_out, l1_, l2_, 0, 2, -2)


def W_rot_m(l1_, l_out, l2_, c=1):
    """Namikawa Eq. (rotation) — W^{α,-} (odd-parity polarization mixing)."""
    return 2 * I * ZETA_MINUS * q_plus(c) * gamma_f(l1_, l_out, l2_) * \
           P * wigner_3j(l_out, l1_, l2_, 0, 2, -2)


# ------------------- rotation f-weights ---------------------------
# Estimator parity for rotation is p_α = −1 (odd).  The pol-mixing
# structure flips relative to lensing: rotation EB uses W^+, not W^-.

def f_rot_EB(px=-1):
    """Rotation EB estimator weight (Namikawa-style).

    Mirrors Namikawa's lensing-EB template with the even/odd roles
    appropriate to rotation (W^{α,+} instead of W^{x,-})."""
    return W_rot_p(l1, l, l2) * CB(l2) + px * W_rot_p(l2, l, l1) * CE(l1)


def f_rot_EE(px=-1):
    """Rotation EE estimator weight — uses W^{α,-}."""
    return W_rot_m(l1, l, l2) * CE(l2) + px * W_rot_m(l2, l, l1) * CE(l1)


def f_rot_TB(px=-1):
    """Rotation TB estimator weight — uses W^{α,+} (symmetric with lensing's TB-via-W^−)."""
    return px * W_rot_p(l2, l, l1) * CTE(l1)


# ======================================================================
# Amplitude (source, patchy τ, point-source hardening) distortion weights.
# Symlens.jl examples/tempura.jl reference:
#   Wₑ⁰(ℓ₁,ℓ₂,ℓ₃,c) = γ·w3j(ℓ₁,ℓ₂,ℓ₃; 0, 0, 0)
# Column-permuted to l-first form:
#   w3j(l1, L, l2; 0,0,0) = P · w3j(L, l1, l2; 0,0,0)
# Amplitude-type distortions are even parity (p_ε = +1).
# ======================================================================

def W_ampl_0(l1_, l_out, l2_, c=1):
    """Amplitude / source / patchy-τ spin-0 weight.  No a-factors, no
    (1+P)/2 factor — the w3j(j; 0, 0, 0) already enforces j1+j2+j3 even
    automatically, making the parity structure trivial."""
    return gamma_f(l1_, l_out, l2_) * P * wigner_3j(l_out, l1_, l2_, 0, 0, 0)


def f_ampl_TT(px=+1):
    """f^{ε,(TT)} = W_ε^0(l, L, l') · C_T(l')  +  p_ε · W_ε^0(l', L, l) · C_T(l)."""
    return W_ampl_0(l1, l, l2) * CT(l2) + px * W_ampl_0(l2, l, l1) * CT(l1)
