"""
quotient.py — reduce a flat-term list modulo simple polynomial relations on
one variable.

A "relation" is specified as a dict `reductions: {power_int → AtomicFactor}`
that says what to replace occurrences of ``Var(name)**power`` with. For
P² = 1, pass `{2: Const(1)}` and the reducer will rewrite every
occurrence of P-powers: P^(2k) → 1, P^(2k+1) → P.

The reducer operates at the FlatTerm level: it walks each factor, finds
Pow(Var(name), k) or the Var itself, and replaces it with the appropriate
reduced form, accumulating the coefficient contribution.

This is generic: any idempotent variable (P² = 1, x^n = 1, etc.) can be
reduced this way.
"""
from __future__ import annotations

from sympy import Rational

from .atom import AtomicFactor, Const, Var, Pow, Mul, Add, FuncApp, mul as atom_mul, const
from .normal_form import FlatTerm


def reduce_mod_power(terms: list[FlatTerm], var_name: str, period: int,
                     residues: dict[int, AtomicFactor] | None = None) -> list[FlatTerm]:
    """Reduce each term's factor list modulo the relation ``var^period = residues.get(0, 1)``.

    Default residues: ``{0: Const(1)}`` — i.e., ``var^period = 1``.

    Each Pow(Var(var_name), k) is replaced by residues[k % period]
    (with missing keys defaulting to Pow(Var, k % period))."""
    if residues is None:
        residues = {0: Const(1)}

    reduced: list[FlatTerm] = []
    for t in terms:
        new_factors: list[AtomicFactor] = []
        extra_coeff = Rational(1)
        for f in t.factors:
            base, exp = _as_power(f, var_name)
            if base is None:
                new_factors.append(f)
                continue
            k = exp % period
            if k in residues:
                rep = residues[k]
                if isinstance(rep, Const):
                    extra_coeff = extra_coeff * rep.value
                else:
                    new_factors.append(rep)
            elif k == 0:
                pass   # absorb as 1
            elif k == 1:
                new_factors.append(Var(var_name))
            else:
                new_factors.append(Pow(Var(var_name), k))
        reduced.append(FlatTerm(coeff=t.coeff * extra_coeff, factors=tuple(new_factors)))
    return reduced


def _as_power(f: AtomicFactor, var_name: str):
    """If f is Var(var_name) or Pow(Var(var_name), k), return (base, k). Else (None, None)."""
    if isinstance(f, Var) and f.name == var_name:
        return f, 1
    if isinstance(f, Pow) and isinstance(f.base, Var) and f.base.name == var_name:
        return f.base, int(f.exp)
    return None, None
