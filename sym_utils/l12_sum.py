#%%
"""
symbolic calculator for quadratic power of wigner 3j functions
summed over two out of the three ells, assumed to be l1, l2.

"""
from dataclasses import dataclass, field
from sympy import symbols, Function, factor, expand, prod, pi, Pow, lambdify, Symbol
from sym_utils.rewrite import Rule, RuleSet, FixedPoint, Postwalk
from functools import partial
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

eliminate_P_rule = Rule.parse(
    name="resolve_P",
    rule_str=(
        "P*w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2) -> "
        "w3j(?l, ?l1, ?l2, -?m, -?m1, -?m2)"),
    symbols_map={'w3j': wigner_3j, "P": P}
)

w3j_to_wigd_rule = Rule.parse(
    name="w3j_to_wigd",
    rule_str=(
        "w3j(?l, ?l1, ?l2, ?m, ?m1, ?m2) * w3j(?l, ?l1, ?l2, ?s, ?s1, ?s2) -> "
        "wigd(?l, ?m, ?s) * wigd(?l1, ?m1, ?s1) * wigd(?l2, ?m2, ?s2)"),
    where="?m + ?m1 + ?m2 == 0, ?s + ?s1 + ?s2 == 0",
    symbols_map={'w3j': wigner_3j, 'wigd': wigner_d}
)

# -----------------------
# useful transformations
# -----------------------

def drop_P(expr):
    """Absorbing P into wigner 3j"""
    non_p_term = factor(expand(expr).subs(P, 0))
    p_term = factor(expand(expr).coeff(P) * P)
    p_term = FixedPoint(Postwalk(RuleSet([
        w3j_sq_rule, eliminate_P_rule 
    ])), simplify=True)(p_term)
    return p_term + non_p_term

def to_wigd(expr):
    """Convert wigner 3j products into wigner d"""
    walk = FixedPoint(Postwalk(RuleSet([
        w3j_sq_rule, w3j_to_wigd_rule
    ])))
    return walk(expr)


def analyze_wigd_by_l(expr):
    """Analyzes an expression by first grouping factors by their variable
    dependency (l, l1, l2) and then separating Wigner-d functions
    within each group."""
    analysis_results = []

    for term in expr.as_ordered_terms():
        dependency_groups = {'l': [], 'l1': [], 'l2': [], 'const': []}
        for factor in term.as_ordered_factors():
            symbols_in_factor = factor.free_symbols
            if l in symbols_in_factor:
                dependency_groups['l'].append(factor)
            elif l1 in symbols_in_factor:
                dependency_groups['l1'].append(factor)
            elif l2 in symbols_in_factor:
                dependency_groups['l2'].append(factor)
            else:
                dependency_groups['const'].append(factor)

        term_analysis = {}
        for part_name, factors_list in dependency_groups.items():
            if not factors_list:
                continue
            if part_name == 'const':
                term_analysis['const'] = prod(factors_list)
                continue
            wigner_info_list = []
            non_wigner_factors = []
            for factor in factors_list:
                base = factor.base if isinstance(factor, Pow) else factor
                if isinstance(base, wigner_d):
                    _, m, n = base.args
                    wigner_info_list.append({'m': m, 'n': n, 'expr': factor})
                else:
                    non_wigner_factors.append(factor)

            term_analysis[f"{part_name}"] = {
                'wigd': wigner_info_list,
                'non_wigd': prod(non_wigner_factors)
            }
        analysis_results.append(term_analysis)

    return analysis_results



@dataclass
class L12SumCompiler:
    lmax: int
    rlmin: int
    rlmax: int

    glq: object = field(init=False)
    cl2cf: callable = field(init=False)
    cf2cl: callable = field(init=False)

    def __post_init__(self):
        self.glq = GLQuad(int((3 * max(self.lmax, self.rlmax) + 1) / 2))
        self.cl2cf = partial(self.glq.cf_from_cl, prefactor=True, lmin=self.rlmin, lmax=self.rlmax)
        self.cf2cl = partial(self.glq.cl_from_cf, lmax=self.lmax)

    def _build_ir(self, expr):
        expr = drop_P(expr)
        expr = to_wigd(expr)
        ana = analyze_wigd_by_l(expr)

        cf2cl = Function("cf2cl")
        cl2cf = Function("cl2cf")

        terms_ = []
        for term_ in ana:
            factors_ = []
            factor_ = term_['l1']
            m, n = factor_['wigd'][0]['m'], factor_['wigd'][0]['n']
            factors_.append(cl2cf(m, n, factor_['non_wigd']))

            factor_ = term_['l2']
            m, n = factor_['wigd'][0]['m'], factor_['wigd'][0]['n']
            factors_.append(cl2cf(m, n, factor_['non_wigd']))

            factor_ = term_['const']
            factors_.append(factor_)

            factor_ = term_['l']
            m, n = factor_['wigd'][0]['m'], factor_['wigd'][0]['n']
            terms_.append(
                cf2cl(m, n, factor_['non_wigd']*prod(factors_))
            )
        return sum(terms_)

    def _compile_ir(self, ir, args):
        # preprocess: l2->l, l1->l, args func -> sym
        # assuming args are Sympy.Function
        ir = ir.subs({l2: l, l1: l})
        args_func = [x for x in args if x.is_Function]
        args_sym = [Symbol(str(x)) for x in args_func]
        f2s = {f(l): s for (f, s) in zip(args_func, args_sym)}
        print(f2s)
        ir = ir.subs(f2s)
        # build function
        fmap = {
            'cl2cf': self.cl2cf,
            'cf2cl': self.cf2cl
        }
        f = lambdify(args, ir, modules=['numpy', fmap])
        return f

    def build_and_compile(self, expr, args):
        ir = self._build_ir(expr)
        func = self._compile_ir(ir, args)
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