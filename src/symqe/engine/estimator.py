"""
estimator.py — compile a symbolic quadratic-estimator expression g(l, L, l')
into a list of EstimatorTerms, each describing the SHT recipe for one atomic
contribution:

    alm2map_spin(X * X_filter(l),     spin_transform = spin_X)    # first field
    alm2map_spin(Y * Y_filter(l'),    spin_transform = spin_Y)    # second field
    product in pixel space
    map2alm_spin(product,             spin_transform = spin_L)    # output L
    multiply result by L_factor(L)

This is the estimator analogue of the normalization compiler in l12_sum.py.
The input is expected to have exactly ONE wigner_3j per term (unlike the
normalization's W² which has two).

No SHT library is imported — this module only produces the recipe; a
separate backend (pixell / healpy) executes it.

Variable naming convention (matches l12_sum.py):
    l  = output multipole L
    l1 = first input multipole (here: the X alm)
    l2 = second input multipole (here: the Y alm)


γ-normalization convention (READ BEFORE USING THIS MODULE)
----------------------------------------------------------
Namikawa's W weights all carry a γ factor

    γ_{l1 L l2} = sqrt( (2·l1+1) · (2·L+1) · (2·l2+1) / (4π) )

The analyzer LEAVES these factors inside the per-leg filters, so you will
see ``sqrt(1 + 2·l1)`` sitting inside ``X_filter``, ``sqrt(1 + 2·l2)``
inside ``Y_filter``, and ``sqrt(1 + 2·l)`` inside ``L_factor`` — plus an
overall ``1/sqrt(4π)`` absorbed into the numeric coefficient.

**When you emit SHT calls to pixell/healpy, you MUST NOT multiply by these
sqrt(2·var+1) factors literally.** The SHT identity

    ∫ dΩ  _s Y*_{LM} · _{s1} Y_{l1 m1} · _{s2} Y_{l2 m2}
        = γ_{l1 L l2} · 3j(l1 L l2; m1 m2 M) · 3j(l1 L l2; -s1 -s2 s)

reproduces γ implicitly when you compose ``alm2map_spin + alm2map_spin +
(pixel multiply) + map2alm_spin`` with the standard library normalizations.
The displayed sqrt(2·var+1) pieces are therefore structural bookkeeping,
not filter factors to actually apply.

To mirror falafel's conventions exactly, the backend should:
  1. Strip every sqrt(2·l1+1), sqrt(2·L+1), sqrt(2·l2+1) from
     (X_filter, L_factor, Y_filter) respectively.
  2. Drop the 1/sqrt(4π) residue (it's absorbed into the 1/(2L+1) of
     Namikawa's normalization, Eq. 82, which your estimator normalizes
     by externally anyway).

Imaginary unit for W^{x,-} (TB, EB estimators)
----------------------------------------------
Namikawa writes W^{x,-} with ζ^- = i, so naive substitution into f gives
complex-valued coefficients. The imaginary i is the encoding of E↔B
mixing and is unwound by the backend via pixell/healpy's rot2d/irot2d
pair (which split a complex spin-±s field into its real and imaginary
spin-s components). The analyzer here just carries ``I`` through the
coefficient — recipes with a residual ``I`` apply to the imaginary
(B-mode) component of the output, recipes with a real coefficient to
the real (E-mode) component.
"""
from __future__ import annotations
from dataclasses import dataclass
from sympy import Rational

from .atom import (
    AtomicFactor, Const, Var, Pow, Mul, Add, FuncApp,
    mul as atom_mul, ONE,
)
from .sympy_bridge import from_sympy
from .normal_form import FlatTerm, flatten
from .quotient import reduce_mod_power
from .l12_sum import _normalize_w3j_args, l, l1, l2, P


@dataclass(frozen=True)
class EstimatorTerm:
    """One atomic term of the compiled estimator.

    Invariants:
      * coeff is a pure number.
      * *_filter / L_factor free-vars are confined to the corresponding group.
      * (spin_L, spin_X, spin_Y) are the signed integer m-values from the
        canonicalized 3j at positions (l, l1, l2).
    """
    coeff: object
    L_factor: AtomicFactor     # free_vars ⊆ {'l'}  — multiplies output at L
    X_filter: AtomicFactor     # free_vars ⊆ {'l1'} — filter on X alm before SHT
    Y_filter: AtomicFactor     # free_vars ⊆ {'l2'} — filter on Y alm before SHT
    spin_L: int
    spin_X: int
    spin_Y: int

    def __post_init__(self):
        assert self.L_factor.free_vars() <= {'l'},  f"L_factor leaks: {self.L_factor.free_vars()}"
        assert self.X_filter.free_vars() <= {'l1'}, f"X_filter leaks: {self.X_filter.free_vars()}"
        assert self.Y_filter.free_vars() <= {'l2'}, f"Y_filter leaks: {self.Y_filter.free_vars()}"


# -------------------------------------------------------- pipeline

def compile_estimator(sympy_expr, *, keep_gamma: bool = False) -> list[EstimatorTerm]:
    """Compile a symbolic estimator weight g(l, L, l') into a list of
    EstimatorTerms.

    By default the γ-factors (``sqrt(2·var+1)`` pieces that come from
    Namikawa's W-weights) are stripped, because every downstream emitter
    assumes the SHT normalization reabsorbs them.  Pass ``keep_gamma=True``
    if you want to inspect the raw, unstripped recipe.

    Pipeline (same layers as l12_sum.compile_norm, different top stage):
      1. from_sympy               sympy → atom tree
      2. _normalize_w3j_args      canonicalize each w3j to j-args (l, l1, l2)
      3. flatten                  distribute + collect to FlatTerms
      4. reduce_mod_power (P²=1)  even P^k → 1, odd → P
      5. _flat_to_estimator_term  extract (coeff, L/X/Y factors, spins)
    """
    atom_expr = from_sympy(sympy_expr, known_symbols={l, l1, l2, P})
    atom_expr = _normalize_w3j_args(atom_expr)
    flat = flatten(atom_expr)
    flat = reduce_mod_power(flat, 'P', period=2)
    terms: list[EstimatorTerm] = []
    for ft in flat:
        t = _flat_to_estimator_term(ft)
        if t is not None:
            terms.append(t)
    if keep_gamma:
        return terms
    from .estimator_backend import strip_gamma
    return strip_gamma(terms)


def _flat_to_estimator_term(ft: FlatTerm) -> EstimatorTerm | None:
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
            raise ValueError(f"factor mixes ell variables: free={free}, factor={f}")

    if len(w3j_apps) != 1:
        raise ValueError(f"estimator term expects exactly 1 wigner_3j, got {len(w3j_apps)}")
    w = w3j_apps[0]

    # After _normalize_w3j_args the j-args must be in canonical (l, l1, l2) order.
    js = w.args[:3]; ms = w.args[3:]
    expected = [('l', 0), ('l1', 1), ('l2', 2)]
    for name, idx in expected:
        v = js[idx]
        if not (isinstance(v, Var) and v.name == name):
            raise ValueError(f"w3j j-args not canonical at pos {idx}: got {v}")

    # Odd residual P flips all three m's (3j identity P·w3j(j;m) = w3j(j;-m)).
    sign_m = -1 if (p_count % 2 == 1) else 1
    spin_L = sign_m * _as_int(ms[0])
    spin_X = sign_m * _as_int(ms[1])
    spin_Y = sign_m * _as_int(ms[2])

    if ft.coeff == 0:
        return None

    return EstimatorTerm(
        coeff=ft.coeff,
        L_factor=atom_mul(*by_var['l']) if by_var['l'] else ONE,
        X_filter=atom_mul(*by_var['l1']) if by_var['l1'] else ONE,
        Y_filter=atom_mul(*by_var['l2']) if by_var['l2'] else ONE,
        spin_L=spin_L, spin_X=spin_X, spin_Y=spin_Y,
    )


def _as_int(atom: AtomicFactor) -> int:
    if isinstance(atom, Const):
        v = atom.value
        if int(v) == v:
            return int(v)
    raise ValueError(f"expected integer m, got {atom}")


# -------------------------------------------------------- grouping

def group_by_spin(terms: list[EstimatorTerm]) -> dict:
    """Group terms by (spin_L, spin_X, spin_Y, X_filter_key, Y_filter_key).
    Terms in the same group share the same SHT plan and their L_factors
    can be summed before a single map2alm_spin."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in terms:
        key = (t.spin_L, t.spin_X, t.spin_Y,
               t.X_filter.structural_key(), t.Y_filter.structural_key())
        groups[key].append(t)
    return dict(groups)


# -------------------------------------------------------- pretty-print

def _atom_to_str(a: AtomicFactor) -> str:
    if isinstance(a, Const):
        v = a.value
        return str(v)
    if isinstance(a, Var):
        return a.name
    if isinstance(a, Pow):
        return f"({_atom_to_str(a.base)})**{a.exp}"
    if isinstance(a, Mul):
        return " * ".join(_atom_to_str(f) for f in a.factors)
    if isinstance(a, Add):
        return "(" + " + ".join(_atom_to_str(t) for t in a.terms) + ")"
    if isinstance(a, FuncApp):
        return f"{a.name}({', '.join(_atom_to_str(x) for x in a.args)})"
    return str(a)


def pretty(terms: list[EstimatorTerm], x_name: str = "X", y_name: str = "Y") -> str:
    lines = [f"{len(terms)} atomic estimator term(s):\n"]
    for i, t in enumerate(terms, 1):
        lines.append(f"-- term {i} --")
        lines.append(f"   coeff    = {t.coeff}")
        lines.append(f"   X leg:   filter(l1)  = {_atom_to_str(t.X_filter)}")
        lines.append(f"            alm2map_spin({x_name} * filter,  spin_transform = {t.spin_X:+d})")
        lines.append(f"   Y leg:   filter(l2)  = {_atom_to_str(t.Y_filter)}")
        lines.append(f"            alm2map_spin({y_name} * filter,  spin_transform = {t.spin_Y:+d})")
        lines.append(f"   pixel:   product of the two maps")
        lines.append(f"   out L:   map2alm_spin(product, spin_transform = {t.spin_L:+d})")
        lines.append(f"            multiply by L_factor(l) = {_atom_to_str(t.L_factor)}")
        lines.append("")
    return "\n".join(lines)


def pretty_grouped(terms: list[EstimatorTerm], x_name: str = "X", y_name: str = "Y") -> str:
    """Like pretty() but collapse terms that share the same SHT plan into
    a single recipe, summing their L_factors."""
    groups = group_by_spin(terms)
    lines = [f"{len(terms)} term(s) collapse to {len(groups)} unique SHT plan(s):\n"]
    for i, ((sL, sX, sY, _kx, _ky), group) in enumerate(groups.items(), 1):
        lines.append(f"-- plan {i} --")
        lines.append(f"   X leg:   filter(l1)  = {_atom_to_str(group[0].X_filter)}")
        lines.append(f"            alm2map_spin({x_name} * filter,  spin_transform = {sX:+d})")
        lines.append(f"   Y leg:   filter(l2)  = {_atom_to_str(group[0].Y_filter)}")
        lines.append(f"            alm2map_spin({y_name} * filter,  spin_transform = {sY:+d})")
        lines.append(f"   pixel:   product")
        lines.append(f"   out L:   map2alm_spin(product, spin_transform = {sL:+d})")
        # Sum L-factors across grouped terms, preserving coefficient.
        L_sum_str = " + ".join(
            f"({t.coeff}) * [{_atom_to_str(t.L_factor)}]" for t in group
        )
        lines.append(f"            multiply by: {L_sum_str}")
        lines.append("")
    return "\n".join(lines)
