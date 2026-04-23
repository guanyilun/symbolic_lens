"""
Dump the falafel-style SHT recipe for each of the 6 lensing quadratic
estimators (TT, TE, TB, EE, EB, BB), derived from their symbolic
g^{XY}(l, L, l') = f^{XY} / (Δ · hat_C^X_l · hat_C^Y_l').

The recipes are compared (by inspection) with falafel.qe's hand-coded
implementations to confirm the symbolic pipeline produces the same
per-leg (filter, spin, output-filter) structure.

IMPORTANT — before reading the filters:
    See estimator.py's "γ-normalization convention" docstring.  Each
    filter contains sqrt(2·var+1) pieces that come from γ_{l1 L l2} and
    are absorbed by the standard SHT library normalization.  The emitter
    (not yet written) will strip them; for this visual check, mentally
    divide them out.

Δ coefficients (Namikawa Eq. 81):
    Δ^{TT} = Δ^{EE} = Δ^{BB} = 2   (same-field estimators)
    Δ^{TE} = Δ^{TB} = Δ^{EB} = 1   (cross-field estimators)
"""
from sympy import sympify

from symqe.engine.l12_sum import l, l1, l2
from symqe.engine.namikawa import (
    hCT, hCE, hCB, hCTE,
    f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
)
from symqe.engine.estimator import compile_estimator, pretty_grouped


# (estimator name,   f builder,            Δ,    denominator spectra (hCX, hCY),  leg names)
CASES = [
    ("TT",  f_TT,  2,   (hCT(l1),  hCT(l2)),   ("T_l1m", "T_l2m")),
    ("TE",  f_TE,  1,   (hCT(l1),  hCE(l2)),   ("T_l1m", "E_l2m")),
    ("TB",  f_TB,  1,   (hCT(l1),  hCB(l2)),   ("T_l1m", "B_l2m")),
    ("EE",  f_EE,  2,   (hCE(l1),  hCE(l2)),   ("E_l1m", "E_l2m")),
    ("EB",  f_EB,  1,   (hCE(l1),  hCB(l2)),   ("E_l1m", "B_l2m")),
    ("BB",  f_BB,  2,   (hCB(l1),  hCB(l2)),   ("B_l1m", "B_l2m")),
]


def main():
    for name, f_builder, delta, (hCX_l1, hCY_l2), (xname, yname) in CASES:
        f_expr = f_builder(px=+1)               # lensing gradient, p_φ = +1
        g_expr = f_expr / (sympify(delta) * hCX_l1 * hCY_l2)

        terms = compile_estimator(g_expr)
        n_terms = len(terms)
        print("=" * 76)
        print(f"  Estimator:  {name}   (Δ = {delta},  legs:  X = {xname},  Y = {yname})")
        print(f"  Symbolic g^{{{name}}} compiles to {n_terms} atomic term(s)")
        print("=" * 76)
        print(pretty_grouped(terms, x_name=xname, y_name=yname))
        print()


if __name__ == "__main__":
    main()
