"""
l12_sum.py — application layer: CMB quadratic-estimator kernel compilation.

This module is specific to sums of the form::

    kernel(L) = Σ_{l1, l2} W_{l1, L, l2} · W'_{l1, L, l2} · A(l1) · B(l2)

where W, W' are scalar combinations of Wigner 3j symbols. It leans on the
generic layers (atom, sympy_bridge, normal_form, quotient) to manipulate
the expression, and adds only the Wigner-specific rules:

  * the identity
      w3j(j₁,j₂,j₃; mₐ,mₐ₁,mₐ₂) · w3j(j₁,j₂,j₃; m_b,m_b₁,m_b₂)
      = wigd(j₁, mₐ, m_b) · wigd(j₂, mₐ₁, m_b₁) · wigd(j₃, mₐ₂, m_b₂) / 2
  * absorption of a residual P-parity into wigd via ∀ leg: (m,n) → (-m,-n)
  * canonicalization of wigd convention (m ≥ 0; if m = 0, n ≥ 0), with
    sign flips accumulated into the coefficient.

Nothing in this module touches sympy except at the parse boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from sympy import symbols, Function, Rational

from .atom import (
    AtomicFactor, Const, Var, FuncApp, Mul, Add, Pow,
    mul as atom_mul, add as atom_add, const as atom_const, var as atom_var,
    power as atom_power, func as atom_func, ONE, ZERO,
)
from .sympy_bridge import from_sympy
from .normal_form import FlatTerm, flatten
from .quotient import reduce_mod_power
from glquad import GLQuad


# ------------------------- sympy-side user symbols -----------------------

l, l1, l2 = symbols("l l1 l2", integer=True, nonnegative=True)
P = symbols("P")


class wigner_3j(Function):
    """sympy Function stub — the compiler treats it as an opaque FuncApp."""
    @classmethod
    def eval(cls, *args): return None

    def _latex(self, printer, **kwargs):
        j1, j2, j3, m1, m2, m3 = [printer._print(a) for a in self.args]
        return (f"\\begin{{pmatrix}} {j1} & {j2} & {j3} \\\\ "
                f"{m1} & {m2} & {m3} \\end{{pmatrix}}")


# ---------------------- application-specific Term -----------------------

@dataclass(frozen=True)
class Term:
    """A single atomic term of the compiled kernel.

    Invariants:
      * coeff is a pure number (Rational).
      * *_factor's free variables are confined to the corresponding group.
      * wigd_* is a (m, n) tuple in canonical form (m ≥ 0; if m = 0, n ≥ 0).
    """
    coeff: object
    l_factor: AtomicFactor        # free_vars ⊆ {'l'}
    l1_factor: AtomicFactor       # free_vars ⊆ {'l1'}
    l2_factor: AtomicFactor       # free_vars ⊆ {'l2'}
    wigd_l: tuple
    wigd_l1: tuple
    wigd_l2: tuple

    def __post_init__(self):
        assert self.l_factor.free_vars()  <= {'l'}
        assert self.l1_factor.free_vars() <= {'l1'}
        assert self.l2_factor.free_vars() <= {'l2'}
        for mn in (self.wigd_l, self.wigd_l1, self.wigd_l2):
            assert isinstance(mn, tuple) and len(mn) == 2


# ------------------- wigner-specific rewrite functions ------------------

def _canonicalize_wigd(mn):
    """Canonical orientation: m ≥ 0, and if m = 0 then n ≥ 0. Returns ((m,n), sign)."""
    m, n = mn
    sign = 1
    if abs(m) < abs(n):
        sign *= (-1) ** (m - n); m, n = n, m
    if m < 0 or (m == 0 and n < 0):
        sign *= (-1) ** (m - n); m, n = -m, -n
    return (m, n), sign


def _flat_term_to_term(ft: FlatTerm) -> Term | None:
    """Turn one FlatTerm into the structured Term.

    Expects exactly 2 wigner_3j FuncApps with identical j-args, and the
    residual P power (if any) to have been reduced already (so we see
    either zero or one Var('P') remaining)."""
    w3j_apps: list[FuncApp] = []
    p_count = 0
    by_var: dict[str, list[AtomicFactor]] = {'l': [], 'l1': [], 'l2': []}

    for f in ft.factors:
        if isinstance(f, FuncApp) and f.name == 'wigner_3j':
            w3j_apps.append(f)
            continue
        if isinstance(f, Var) and f.name == 'P':
            p_count += 1
            continue
        # Scalars and user FuncApps: bucket by free variable
        free = f.free_vars()
        if 'P' in free:
            raise ValueError(f"P mixed inside a non-atomic factor: {f}")
        if free <= {'l'}:
            by_var['l'].append(f)
        elif free <= {'l1'}:
            by_var['l1'].append(f)
        elif free <= {'l2'}:
            by_var['l2'].append(f)
        else:
            raise ValueError(f"factor mixes ell variables (not atomic): free={free}, factor={f}")

    if len(w3j_apps) != 2:
        raise ValueError(f"expected 2 wigner_3j per term, got {len(w3j_apps)}")

    wigd_l, wigd_l1, wigd_l2 = _resolve_w3j_pair(w3j_apps[0], w3j_apps[1])
    coeff = ft.coeff * Rational(1, 2)   # the 3j-to-wigd identity contributes /2

    # Absorb odd P into the wigd triple.
    # The identity P·w3j(j; m1,m2,m3) = w3j(j; -m1,-m2,-m3) flips exactly ONE
    # of the two w3j factors (the "m-side") in the product w3j(ma)·w3j(mb).
    # So each wigd(l, ma, mb) has its first-slot m negated; the second-slot
    # stays the same.
    p_count %= 2
    if p_count == 1:
        wigd_l  = (-wigd_l[0],  wigd_l[1])
        wigd_l1 = (-wigd_l1[0], wigd_l1[1])
        wigd_l2 = (-wigd_l2[0], wigd_l2[1])

    (wigd_l,  s_l ) = _canonicalize_wigd(wigd_l)
    (wigd_l1, s_l1) = _canonicalize_wigd(wigd_l1)
    (wigd_l2, s_l2) = _canonicalize_wigd(wigd_l2)
    coeff = coeff * s_l * s_l1 * s_l2

    if coeff == 0:
        return None

    return Term(
        coeff=coeff,
        l_factor =atom_mul(*by_var['l'])  if by_var['l']  else ONE,
        l1_factor=atom_mul(*by_var['l1']) if by_var['l1'] else ONE,
        l2_factor=atom_mul(*by_var['l2']) if by_var['l2'] else ONE,
        wigd_l=wigd_l, wigd_l1=wigd_l1, wigd_l2=wigd_l2,
    )


def _resolve_w3j_pair(a: FuncApp, b: FuncApp):
    """Apply w3j·w3j → wigd⊗wigd⊗wigd identity, given equal j-args.

    Returns (wigd_for_l, wigd_for_l1, wigd_for_l2) where each is the (m, n)
    taken from the two 3js' m-values for the corresponding j."""
    assert len(a.args) == 6 and len(b.args) == 6
    ja, jb = a.args[:3], b.args[:3]
    if ja != jb:
        raise ValueError(f"w3j j-args differ: {ja} vs {jb}")
    ma, mb = a.args[3:], b.args[3:]
    # find the position of each of l, l1, l2 in the j-tuple
    def pos(name):
        for i, v in enumerate(ja):
            if isinstance(v, Var) and v.name == name: return i
        raise ValueError(f"w3j j-tuple missing {name}: {ja}")
    wigd_for = {}
    for name in ('l', 'l1', 'l2'):
        i = pos(name)
        wigd_for[name] = (int(_as_int(ma[i])), int(_as_int(mb[i])))
    return wigd_for['l'], wigd_for['l1'], wigd_for['l2']


def _as_int(atom: AtomicFactor) -> int:
    if isinstance(atom, Const):
        v = atom.value
        if int(v) == v: return int(v)
    raise ValueError(f"expected integer constant, got {atom}")


# ----------------------------- parse pipeline ---------------------------

def compile_to_terms(sympy_expr) -> list[Term]:
    """sympy expression → list[Term], fully canonicalized."""
    atom_expr = from_sympy(sympy_expr, known_symbols={l, l1, l2, P})
    atom_expr = _normalize_w3j_args(atom_expr)
    flat = flatten(atom_expr)
    flat = reduce_mod_power(flat, 'P', period=2)        # P² = 1
    terms: list[Term] = []
    for ft in flat:
        t = _flat_term_to_term(ft)
        if t is not None: terms.append(t)
    return terms


def _normalize_w3j_args(node: AtomicFactor) -> AtomicFactor:
    """Permute each wigner_3j FuncApp's j-args to canonical (l, l1, l2) order.

    3j column-permutation identity:
      w3j(j₁,j₂,j₃; m₁,m₂,m₃) = (-1)^{j₁+j₂+j₃} · w3j(j_{π(1)},…; m_{π(1)},…)
    for any odd permutation π. Even permutations leave the value unchanged.

    We detect non-canonical orderings of (l, l1, l2) and rewrite the w3j
    in-place, introducing a P factor (= (-1)^{l+l1+l2}) for odd permutations.
    """
    CANON = ('l', 'l1', 'l2')
    p_sym = atom_var('P')

    def is_odd_perm(perm):
        # count inversions
        n = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]: n += 1
        return n % 2 == 1

    def step(n):
        if isinstance(n, FuncApp) and n.name == 'wigner_3j':
            js = n.args[:3]; ms = n.args[3:]
            names = [j.name if isinstance(j, Var) else None for j in js]
            if any(x is None for x in names):
                return n
            if tuple(names) == CANON:
                return n
            order = [CANON.index(name) for name in names]   # where each current pos goes
            perm_from_canon = [names.index(c) for c in CANON]  # for canonical: which current idx
            new_js = tuple(js[perm_from_canon[i]] for i in range(3))
            new_ms = tuple(ms[perm_from_canon[i]] for i in range(3))
            new_w3j = atom_func('wigner_3j', *new_js, *new_ms)
            if is_odd_perm(perm_from_canon):
                return atom_mul(p_sym, new_w3j)
            return new_w3j
        return n
    return node.walk(step)


# ----------------------------- code emission ----------------------------

@dataclass
class L12SumCompiler:
    """Compile a sympy expression into a numpy callable kernel(L, A, B, ...).

    Usage:
        expr = (symbolic formula in terms of l, l1, l2, P, wigner_3j,
                and user sympy Functions A, B, ...)
        compiler = L12SumCompiler(lmax=..., rlmin=..., rlmax=...)
        func, ir = compiler.build_and_compile(expr, args=[l, A, B])
        kernel = func(ell_out_array, A_array, B_array)
    """
    lmax: int
    rlmin: int
    rlmax: int
    cl2cf: callable = field(init=False)
    cf2cl: callable = field(init=False)

    def __post_init__(self):
        glq = GLQuad(int((3 * max(self.lmax, self.rlmax) + 1) / 2))
        self.cl2cf = partial(glq.cf_from_cl, lmin=self.rlmin, lmax=self.rlmax)
        self.cf2cl = partial(glq.cl_from_cf, lmax=self.lmax)

    def build_and_compile(self, sympy_expr, args=None):
        """args = [ell_out_symbol, UserFunc1, UserFunc2, ...]; the first entry
        is the sympy Symbol for the output L (ignored — always the leading
        runtime arg), the rest are sympy Function classes whose __name__
        fixes the positional order of the user arrays."""
        terms = compile_to_terms(sympy_expr)
        func_names_ordered = None
        if args is not None and len(args) > 1:
            func_names_ordered = [a.__name__ if isinstance(a, type) else a.name
                                  for a in args[1:]]
        ir = _emit(terms, self.cl2cf, self.cf2cl, self.lmax, func_names_ordered)
        return ir, terms

    build = build_and_compile    # friendly alias


def _emit(terms: list[Term], cl2cf, cf2cl, lmax, func_names_ordered=None):
    """Group Terms by (wigd_l, l_factor_key); emit a closure that evaluates
    the full kernel."""
    import numpy as np
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for t in terms:
        key = (t.wigd_l, t.l_factor.structural_key())
        groups[key].append(t)

    # Decide the positional order of user arrays at call time. If the caller
    # supplied the explicit order, use it; otherwise scan in first-appearance
    # order.
    if func_names_ordered is not None:
        func_names = list(func_names_ordered)
    else:
        func_names = []
        for t in terms:
            for node in (t.l_factor, t.l1_factor, t.l2_factor):
                for sub in _iter_nodes(node):
                    if isinstance(sub, FuncApp) and sub.name not in func_names:
                        func_names.append(sub.name)

    def compiled(ell_out, *arrays):
        assert len(arrays) == len(func_names), \
            f"expected {len(func_names)} user arrays ({func_names}), got {len(arrays)}"
        env_base = {name: np.asarray(a) for name, a in zip(func_names, arrays)}
        ell = np.arange(0, lmax + 1)  # domain for evaluating l1/l2/l factors
        env_l1 = dict(env_base, l1=ell)
        env_l2 = dict(env_base, l2=ell)
        env_l  = dict(env_base, l=np.asarray(ell_out))

        result = np.zeros(lmax + 1)
        for (wigd_l, _lkey), group in groups.items():
            acc_cf = None
            # All terms in this group share wigd_l.m/wigd_l.n AND l_factor,
            # so they differ only in wigd_l1/wigd_l2 and their l1/l2 factors.
            for t in group:
                zeta_l1 = t.l1_factor.evaluate(env_l1)
                zeta_l2 = t.l2_factor.evaluate(env_l2)
                cf_l1 = cl2cf(t.wigd_l1[0], t.wigd_l1[1], np.asarray(zeta_l1))
                cf_l2 = cl2cf(t.wigd_l2[0], t.wigd_l2[1], np.asarray(zeta_l2))
                contrib = float(t.coeff) * cf_l1 * cf_l2
                acc_cf = contrib if acc_cf is None else acc_cf + contrib
            cl_part = cf2cl(wigd_l[0], wigd_l[1], acc_cf)
            l_pref = group[0].l_factor.evaluate(env_l)
            # Broadcast: l_pref may be scalar (when l_factor is ONE)
            result = result + np.asarray(cl_part) * np.asarray(l_pref)
        return result

    return compiled


def _iter_nodes(node: AtomicFactor):
    """Yield every subtree (pre-order)."""
    yield node
    if isinstance(node, Pow):
        yield from _iter_nodes(node.base)
    elif isinstance(node, Mul):
        for f in node.factors: yield from _iter_nodes(f)
    elif isinstance(node, Add):
        for t in node.terms: yield from _iter_nodes(t)
    elif isinstance(node, FuncApp):
        for a in node.args: yield from _iter_nodes(a)
