"""
atom.py — domain-agnostic atomic-factor algebra.

An AtomicFactor is a pure immutable tree with strictly no auto-distribution:
``Mul(Add(a, b), c)`` stays that way until you explicitly call `.distribute()`.
This is the invariant that makes downstream pattern matching predictable.

Nothing here knows about Wigner symbols, CMB lensing, or P-parity. The tree
is a general-purpose scalar expression language intended to be reused for
any symbolic pipeline where you want full control over when simplification
happens.

Constructors `mul`, `add`, `power` do minimal normalization:
  - flatten nested Mul/Add
  - fold adjacent numeric constants
  - drop zeros / ones
  - collapse Pow(Pow(x, a), b) to Pow(x, a*b)
They do NOT distribute, expand, or reorder non-numeric factors.

Evaluation is driven by a user-provided environment dict that maps Var names
and FuncApp names to numpy arrays (or anything arithmetic-compatible).
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable

from sympy import Rational, Integer, sympify, Expr as SympyExpr


# ------------------------------------------------------------------ tree

class AtomicFactor:
    """Abstract base for the atomic tree. Instances are immutable and hashable."""
    __slots__ = ()

    # Every node implements these:
    def free_vars(self) -> frozenset[str]:
        raise NotImplementedError

    def evaluate(self, env: dict):
        """env maps Var.name → scalar/array, and FuncApp.name → callable OR array
        (if array, it is indexed by the evaluated single argument)."""
        raise NotImplementedError

    def structural_key(self):
        """Hashable structural identity — two equal-structure atoms have the
        same key, suitable for dict grouping."""
        raise NotImplementedError

    def substitute(self, mapping: dict) -> "AtomicFactor":
        """Return a new tree with Var names substituted per mapping.

        Keys may be strings (Var names) or AtomicFactor instances;
        values must be AtomicFactor. Does not descend into FuncApp.name."""
        return _substitute(self, mapping)

    def walk(self, fn: Callable[["AtomicFactor"], "AtomicFactor"]) -> "AtomicFactor":
        """Bottom-up rewrite: apply `fn` to each subtree; children are
        rewritten first, then the resulting node is passed to fn."""
        return _walk(self, fn)


@dataclass(frozen=True)
class Const(AtomicFactor):
    value: object        # Rational, int, or float

    def free_vars(self): return frozenset()
    def evaluate(self, env): return float(self.value)
    def structural_key(self): return ('C', self.value)


@dataclass(frozen=True)
class Var(AtomicFactor):
    name: str

    def free_vars(self): return frozenset({self.name})
    def evaluate(self, env): return env[self.name]
    def structural_key(self): return ('V', self.name)


@dataclass(frozen=True)
class Pow(AtomicFactor):
    base: AtomicFactor
    exp: object          # Rational or numeric

    def free_vars(self): return self.base.free_vars()
    def evaluate(self, env):
        import numpy as np
        b = self.base.evaluate(env)
        e = float(self.exp)
        with np.errstate(invalid='ignore'):
            return np.power(b, e)
    def structural_key(self): return ('P', self.base.structural_key(), self.exp)


@dataclass(frozen=True)
class Mul(AtomicFactor):
    factors: tuple

    def free_vars(self):
        out = frozenset()
        for f in self.factors: out = out | f.free_vars()
        return out
    def evaluate(self, env):
        r = 1.0
        for f in self.factors: r = r * f.evaluate(env)
        return r
    def structural_key(self):
        return ('M',) + tuple(sorted(f.structural_key() for f in self.factors))


@dataclass(frozen=True)
class Add(AtomicFactor):
    terms: tuple

    def free_vars(self):
        out = frozenset()
        for t in self.terms: out = out | t.free_vars()
        return out
    def evaluate(self, env):
        return sum(t.evaluate(env) for t in self.terms)
    def structural_key(self):
        return ('A',) + tuple(sorted(t.structural_key() for t in self.terms))


@dataclass(frozen=True)
class FuncApp(AtomicFactor):
    """Application of a named function to a tuple of AtomicFactor args.

    The engine treats this opaquely: it doesn't know what the function does.
    Evaluation looks up `env[name]` — either a callable or an array (the
    latter is indexed by the single arg's evaluation; useful for A(l1)
    patterns where A is a numpy array indexed by ell)."""
    name: str
    args: tuple  # tuple[AtomicFactor, ...]

    def free_vars(self):
        out = frozenset()
        for a in self.args: out = out | a.free_vars()
        return out

    def evaluate(self, env):
        import numpy as np
        obj = env[self.name]
        if callable(obj):
            return obj(*[a.evaluate(env) for a in self.args])
        # array lookup: needs exactly one Var arg
        assert len(self.args) == 1, f"array-backed {self.name} needs exactly one arg"
        idx = self.args[0].evaluate(env)
        return np.asarray(obj)[np.asarray(idx).astype(int)]

    def structural_key(self):
        return ('F', self.name, tuple(a.structural_key() for a in self.args))


# --------------------------------------------------------- constructors

ZERO = Const(0)
ONE  = Const(1)


def _num(x):
    """Normalize a numeric scalar. Accepts int, float, Rational, or any
    sympy numeric (incl. constants like pi). Returns a sympy Expr or Python
    int — anything you can multiply together consistently."""
    if isinstance(x, (Rational, Integer)): return x
    if isinstance(x, int):                 return x
    if isinstance(x, SympyExpr) and x.is_number:
        return x
    if isinstance(x, Real):                return Rational(x).limit_denominator(10**12)
    # last resort: try sympify
    try:
        s = sympify(x)
        if s.is_number: return s
    except Exception:
        pass
    raise TypeError(f"not a numeric scalar: {x!r}")


def const(value) -> Const:
    return Const(_num(value))

def var(name: str) -> Var:
    return Var(name)

def func(name: str, *args: AtomicFactor) -> FuncApp:
    return FuncApp(name, tuple(args))

def power(base: AtomicFactor, exp) -> AtomicFactor:
    """Pow constructor. Folds Pow(Pow(x,a), b) = Pow(x, a*b) and Pow(Const,n)."""
    if isinstance(base, Const):
        e = _num(exp)
        return Const(base.value ** e)
    if isinstance(base, Pow):
        return Pow(base.base, _num(base.exp) * _num(exp))
    e = _num(exp)
    if e == 0: return ONE
    if e == 1: return base
    return Pow(base, e)

def mul(*args: AtomicFactor) -> AtomicFactor:
    """Flatten nested Muls, fold numeric constants, drop 1, short-circuit 0.

    DOES NOT distribute over Add. To expand a product of sums, call
    `.distribute()` on the result."""
    flat: list[AtomicFactor] = []
    const_prod = Integer(1)
    for a in args:
        if isinstance(a, Mul):
            flat.extend(a.factors)
        else:
            flat.append(a)
    out: list[AtomicFactor] = []
    for f in flat:
        if isinstance(f, Const):
            if f.value == 0: return ZERO
            const_prod = _num(f.value) * const_prod
        else:
            out.append(f)
    if const_prod != 1:
        out.insert(0, Const(const_prod))
    if not out:  return ONE
    if len(out) == 1: return out[0]
    return Mul(tuple(out))

def add(*args: AtomicFactor) -> AtomicFactor:
    """Flatten nested Adds, fold numeric constants, drop 0. No distribution."""
    flat: list[AtomicFactor] = []
    const_sum = Integer(0)
    for a in args:
        if isinstance(a, Add):
            flat.extend(a.terms)
        else:
            flat.append(a)
    out: list[AtomicFactor] = []
    for t in flat:
        if isinstance(t, Const):
            const_sum = const_sum + _num(t.value)
        else:
            out.append(t)
    if const_sum != 0:
        out.insert(0, Const(const_sum))
    if not out:  return ZERO
    if len(out) == 1: return out[0]
    return Add(tuple(out))


# ----------------------------------------------------- traversal helpers

def _walk(node: AtomicFactor, fn):
    if isinstance(node, (Const, Var)):
        return fn(node)
    if isinstance(node, Pow):
        return fn(Pow(_walk(node.base, fn), node.exp))
    if isinstance(node, Mul):
        return fn(mul(*[_walk(f, fn) for f in node.factors]))
    if isinstance(node, Add):
        return fn(add(*[_walk(t, fn) for t in node.terms]))
    if isinstance(node, FuncApp):
        return fn(FuncApp(node.name, tuple(_walk(a, fn) for a in node.args)))
    raise TypeError(type(node))


def _substitute(node: AtomicFactor, mapping: dict):
    # Build a normalized mapping keyed by Var.name
    name_map: dict[str, AtomicFactor] = {}
    for k, v in mapping.items():
        if isinstance(k, str): name_map[k] = v
        elif isinstance(k, Var): name_map[k.name] = v
        else: raise TypeError(f"substitution key must be str or Var: {k!r}")
    def step(n):
        if isinstance(n, Var) and n.name in name_map:
            return name_map[n.name]
        return n
    return _walk(node, step)


# ---------- distribute: explicit request to expand products over sums ----

def distribute(node: AtomicFactor) -> AtomicFactor:
    """Push Mul across Add and expand integer powers bottom-up.

    * ``Pow(X, n)`` for any non-Const X and non-negative integer n is
      unfolded into ``Mul(X, X, ..., X)`` — this lets later stages see each
      copy as an independent factor (useful for opaque FuncApps like w3j²).
    * ``Mul(..., Add(a, b), ...)`` is pushed to ``Add(Mul(..., a, ...), ...)``.
    * Const bases of Pow are left to `power(Const, exp)` (already exact).

    No other simplification is performed."""
    def is_nonneg_int(e):
        try:
            i = int(e)
            return i == e and i >= 0
        except (TypeError, ValueError):
            return False

    def step(n):
        if isinstance(n, Pow) and not isinstance(n.base, Const) and is_nonneg_int(n.exp):
            k = int(n.exp)
            if k == 0: return ONE
            unfolded = mul(*[n.base] * k)
            return distribute(unfolded)
        if isinstance(n, Mul):
            for i, f in enumerate(n.factors):
                if isinstance(f, Add):
                    others = n.factors[:i] + n.factors[i+1:]
                    rest = mul(*others)
                    return add(*[distribute(mul(rest, t)) for t in f.terms])
        return n
    return _walk(node, step)


# --------------- partition_factors: generic by-predicate split -----------

def partition_factors(node: AtomicFactor, predicates: dict[str, Callable[[AtomicFactor], bool]]) \
        -> tuple[dict[str, list[AtomicFactor]], AtomicFactor]:
    """Split the factors of a Mul by a predicate table.

    `predicates` maps a bucket-name to a predicate on AtomicFactor. Each
    top-level factor of `node` goes into the FIRST bucket whose predicate
    returns True; factors matching none are returned as the second tuple
    element (as a Mul of the leftovers).

    If `node` is not a Mul, it is treated as a single factor.

    Returns (buckets: dict[name, list[AtomicFactor]], leftover: AtomicFactor)."""
    factors = list(node.factors) if isinstance(node, Mul) else [node]
    buckets: dict[str, list[AtomicFactor]] = {k: [] for k in predicates}
    leftover: list[AtomicFactor] = []
    for f in factors:
        placed = False
        for name, pred in predicates.items():
            if pred(f):
                buckets[name].append(f)
                placed = True
                break
        if not placed:
            leftover.append(f)
    return buckets, (mul(*leftover) if leftover else ONE)
