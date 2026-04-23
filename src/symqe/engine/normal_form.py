"""
normal_form.py — canonical sum-of-products for atomic trees.

A `FlatTerm` is (coeff: Rational, factors: tuple[AtomicFactor, ...]) where
factors contains no Const and no top-level Add.

`flatten(node)` distributes and collects the expression into a list of
FlatTerms. This is the atomic-tree analogue of sympy's `.expand()`
followed by `.as_ordered_terms()`, but under our control (no surprise
rewrites elsewhere in the tree).
"""
from __future__ import annotations

from dataclasses import dataclass
from sympy import Rational, Integer

from .atom import AtomicFactor, Const, Add, Mul, distribute, mul as atom_mul


@dataclass(frozen=True)
class FlatTerm:
    coeff: object                # Rational
    factors: tuple               # tuple[AtomicFactor, ...] — no Const, no Add

    def to_atom(self):
        return atom_mul(Const(self.coeff), *self.factors)


def flatten(node: AtomicFactor) -> list[FlatTerm]:
    """Distribute + collect into a list of FlatTerms.

    Numeric coefficients are pulled out. Non-numeric, non-Add factors stay
    as the `factors` tuple; Const nodes are absorbed into `coeff`."""
    expanded = distribute(node)
    if isinstance(expanded, Add):
        return [_as_flat(t) for t in expanded.terms]
    return [_as_flat(expanded)]


def _as_flat(node: AtomicFactor) -> FlatTerm:
    coeff = Integer(1)
    factors: list[AtomicFactor] = []
    items = list(node.factors) if isinstance(node, Mul) else [node]
    for f in items:
        if isinstance(f, Const):
            coeff = coeff * f.value
        else:
            factors.append(f)
    return FlatTerm(coeff=coeff, factors=tuple(factors))
