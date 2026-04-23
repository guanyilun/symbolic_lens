"""
sympy_bridge.py — convert a sympy expression into the atomic-factor tree.

Domain-agnostic: does NOT know about wigner_3j, parity, ell, or anything
else. Any sympy Function subclass (user-defined or sympy-defined like
`wigner_3j`) becomes a `FuncApp(name=func_class_name, args=(...))`; the
caller decides what to do with it.

This is the ONE place sympy is used. Downstream code operates on atoms.
"""
from __future__ import annotations

from sympy import Symbol, Integer, Rational, Float, Pow as SPow, Mul as SMul, Add as SAdd
from sympy.core.function import AppliedUndef, Function

from .atom import (
    AtomicFactor, Const, Var, FuncApp,
    const, var, power, mul, add, func,
)


def from_sympy(expr, *, known_symbols: set[Symbol] | None = None) -> AtomicFactor:
    """Convert a sympy scalar expression to an AtomicFactor.

    `known_symbols`: if given, only these sympy Symbols are allowed; a raise
    is thrown on any unknown Symbol. Pass None to accept any symbol (its
    name becomes the Var name).

    Function applications (any `Function` instance — AppliedUndef or
    concrete sympy Functions like `wigner_3j`) become FuncApp nodes carrying
    the class name and recursively converted args.
    """
    return _convert(expr, known_symbols)


def _convert(e, known):
    if isinstance(e, Symbol):
        if known is not None and e not in known:
            raise ValueError(f"unknown symbol: {e}")
        return var(e.name)
    if isinstance(e, (Integer, Rational, Float)):
        return const(e)
    if isinstance(e, SPow):
        return power(_convert(e.base, known), e.exp)
    if isinstance(e, SMul):
        return mul(*[_convert(a, known) for a in e.args])
    if isinstance(e, SAdd):
        return add(*[_convert(a, known) for a in e.args])
    if isinstance(e, Function):          # covers AppliedUndef and custom Function subclasses
        name = type(e).__name__
        return func(name, *[_convert(a, known) for a in e.args])
    # Fallback: anything else numeric-like
    if hasattr(e, 'is_number') and e.is_number:
        return const(e)
    raise ValueError(f"cannot convert {type(e).__name__}: {e}")
