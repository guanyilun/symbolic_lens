#%%
"""
symbolic calculator for quadratic power of wigner 3j functions
summed over two out of the three ells, assumed to be l1, l2.

"""
from dataclasses import dataclass, field
from sympy import symbols, Function, factor, expand, prod, pi, lambdify, Symbol, Add
from sym_utils.rewrite import Rule, RuleSet, FixedPoint, Postwalk, Prewalk
from functools import partial
from collections import defaultdict

from glquad import GLQuad


# --------------
# basic symbols
# --------------

# it is assumed that l1 and l2 are dummy as they will be summed over
l, l1, l2 = symbols("l l1 l2", integer=True, nonnegative=True)

# parity factor: (-1)^(l+l1+l2)
P = symbols("P")  

# ---------------
# basic functions
# ---------------

class wigner_d(Function):
    @classmethod
    def eval(cls, *args): 
        if len(args) != 3:
            raise ValueError("wigner_d function must have exactly 3 arguments, got {}".format(args))
        return None
    def _latex(self, printer, **kwargs):
        l, m, n = [printer._print(arg) for arg in self.args]
        return f"d^{{{l}}}_{{{m},{n}}}"

class wigner_3j(Function):
    @classmethod
    def eval(cls, j1, j2, j3, m1, m2, m3): return None
    def _latex(self, printer, **kwargs):
        j1, j2, j3, m1, m2, m3 = [printer._print(arg) for arg in self.args]
        expr_str = (f"\\begin{{pmatrix}} {j1} & {j2} & {j3} \\\\ "
                    f"{m1} & {m2} & {m3} \\end{{pmatrix}}")
        if 'exp' in kwargs and kwargs['exp'] is not None:
            exp = printer._print(kwargs['exp'])
            expr_str += f"^{exp}"
        return expr_str

# ---------------------------
# basic transformation rules
# ---------------------------

w3j_sq_rule = Rule.parse(
    name="w3j_sq",
    rule_str=(
        "w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2)**2 -> "
        "w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2) * w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2)"),
    symbols_map={'w3j': wigner_3j}
)

w3j_order_rule = Rule.parse(
    name="w3j_order",
    rule_str=(
        "w3j(l, l2, l1, ?m, ?m2, ?m1) -> "
        "w3j(l, l1, l2, -?m, -?m1, -?m2)"
    ),
    symbols_map={'w3j': wigner_3j, 'l': l, 'l1': l1, 'l2': l2}
)

eliminate_P_rule = RuleSet([
    Rule.parse(
        name="resolve_P",
        rule_str=(
            "P*w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2) -> "
            "w3j(?l, ?l1, ?l2, -?m, -?m1, -?m2)"),
        symbols_map={'w3j': wigner_3j, "P": P}
    ),
    Rule.parse(
        name="resolve_P2",
        rule_str="P**2 -> 1"
    )
])

w3j_to_wigd_rule = Rule.parse(
    name="w3j_to_wigd",
    rule_str=(
        "w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2) * w3j(?l, ?l1, ?l2, ?s, ?s1, ?s2) -> "
        "wigd(?l, ?m, ?s) * wigd(?l1, ?m1, ?s1) * wigd(?l2, ?m2, ?s2) / 2"),
    where="And(?m + ?m1 + ?m2 == 0, ?s + ?s1 + ?s2 == 0)",
    symbols_map={'w3j': wigner_3j, 'wigd': wigner_d}
)

wigd_sign_rules = RuleSet([
    Rule.parse(
        name="wigd_swap",
        rule_str="wigd(?l, ?m, ?n) -> (-1)**(?m-?n)*wigd(?l, ?n, ?m)",
        where="abs(?m) < abs(?n)",
        symbols_map={'wigd': wigner_d, 'abs': abs}
    ),
    Rule.parse(
        name="wigd_flip",
        rule_str="wigd(?l, ?m, ?n) -> (-1)**(?m-?n)*wigd(?l, -?m, -?n)",
        where="?m < 0",
        symbols_map={'wigd': wigner_d}
    )
])

# -----------------------
# useful transformations
# -----------------------

def drop_P(expr, simplify=True):
    """Absorbing P into wigner 3j"""
    non_p_term = factor(expand(expr).subs(P, 0))
    p_term = factor(expand(expr).coeff(P) * P)
    p_term = FixedPoint(Postwalk(RuleSet([
        w3j_sq_rule, eliminate_P_rule, 
    ])), simplify=simplify)(p_term)
    return p_term + non_p_term

def to_wigd(expr):
    """Convert wigner 3j products into wigner d"""
    walk = FixedPoint(Postwalk(RuleSet([
        w3j_sq_rule, w3j_order_rule, w3j_to_wigd_rule
    ])))
    return walk(expr)

def flip_wigd_order(expr):
    walk = FixedPoint(Prewalk(wigd_sign_rules))
    return walk(expr)

def analyze_wigd_and_group(expr):
    """
    Analyzes a raw symbolic expression and directly builds a simplified,
    grouped Intermediate Representation (IR) in a single pass.
    """
    kernel_sums = defaultdict(int)
    non_wigd_l_map = {}
    cl2cf, cf2cl = Function("cl2cf"), Function("cf2cl")

    for term in expr.as_ordered_terms():
        # Group all multiplicative factors by l, l1, l2 dependency
        deps = defaultdict(list)
        for factor in term.as_ordered_factors():
            if not factor.free_symbols: deps['const'].append(factor)
            elif l in factor.free_symbols: deps['l'].append(factor)
            elif l1 in factor.free_symbols: deps['l1'].append(factor)
            elif l2 in factor.free_symbols: deps['l2'].append(factor)

        # Helper to extract wigner and non-wigner parts from a group of factors
        def process_group(factors):
            wigd, non_wigd = [], []
            for f in factors:
                base = getattr(f, 'base', f)
                (wigd if isinstance(base, wigner_d) else non_wigd).append(f)
            # This assumes each dependency group has exactly one wigner function
            m, n = getattr(wigd[0], 'base', wigd[0]).args[1:]
            return {'m': m, 'n': n}, prod(non_wigd)

        # Process each dependency part
        l_wigd, l_non_wigd = process_group(deps['l'])
        l1_wigd, l1_non_wigd = process_group(deps['l1'])
        l2_wigd, l2_non_wigd = process_group(deps['l2'])

        grouping_key = (l_wigd['m'], l_wigd['n'])
        
        kernel = prod(deps['const']) * \
                 cl2cf(l1_wigd['m'], l1_wigd['n'], l1_non_wigd) * \
                 cl2cf(l2_wigd['m'], l2_wigd['n'], l2_non_wigd)
        
        kernel_sums[grouping_key] += kernel

        if grouping_key not in non_wigd_l_map:
            non_wigd_l_map[grouping_key] = l_non_wigd

    final_terms = [cf2cl(m, n, non_wigd_l_map[(m, n)] * total_kernel)
                   for (m, n), total_kernel in kernel_sums.items()]

    return sum(final_terms)


@dataclass
class L12SumCompiler:
    lmax: int
    rlmin: int
    rlmax: int

    cl2cf: callable = field(init=False)
    cf2cl: callable = field(init=False)

    def __post_init__(self):
        glq = GLQuad(int((3 * max(self.lmax, self.rlmax) + 1) / 2))
        self.cl2cf = partial(glq.cf_from_cl, lmin=self.rlmin, lmax=self.rlmax)
        self.cf2cl = partial(glq.cl_from_cf, lmax=self.lmax)

    def _build_ir(self, expr):
        """intermediate representation, with cl2cf and cf2cl calls"""
        expr = expand(expr)
        if not isinstance(expr, Add):
            raise ValueError(f"Failed to expand: {expr}")
        terms = []
        for term in expr.as_ordered_terms():
            term = drop_P(term)
            term = to_wigd(term)
            term = flip_wigd_order(term)
            term = term.doit()
            terms.append(term)
        ir = analyze_wigd_and_group(sum(terms))
        return ir

    def _compile_ir(self, ir, args):
        # preprocess: l2->l, l1->l, args func -> sym
        # assuming args are Sympy.Function
        ir = ir.subs({l2: l, l1: l})
        args_func = [x for x in args if x.is_Function]
        args_sym = [Symbol(str(x)) for x in args_func]
        f2s = {f(l): s for (f, s) in zip(args_func, args_sym)}
        ir = ir.subs(f2s)
        # build function
        fmap = {
            'cl2cf': self.cl2cf,
            'cf2cl': self.cf2cl
        }
        f = lambdify(args, ir, modules=['numpy', fmap])
        return f, ir

    def build_and_compile(self, expr, args):
        ir = self._build_ir(expr)
        func, ir = self._compile_ir(ir, args)
        return func, ir
 
if __name__ == '__main__':
    ClEE = Function("ClEE")
    Claa = Function("Claa")

    # clbb
    expr = wigner_3j(l, l1, l2, 2, -2, 0)**2 * \
            ((2*l1 + 1)/(2*pi) * (2*l2 + 1) / (2*pi)) * \
            (1 + P) * Claa(l2) * ClEE(l1)

    compiler = L12SumCompiler(lmax=10, rlmin=1, rlmax=10)
    f, ir = compiler.build_and_compile(expr, args=[l, ClEE, Claa])
    print(ir)
    print(f.__doc__)

    # test
    import numpy as np
    ell = np.arange(100)
    ClEE = ell**2
    Claa = ell**2

    print(f(ell, ClEE, Claa))