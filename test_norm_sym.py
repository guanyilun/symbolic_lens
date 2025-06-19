#%%
from sympy import Function, sqrt, pi, I
from sym_utils.l12_sum import *
from sym_utils.rewrite import *

import jax
jax.config.update("jax_enable_x64", True)

A, B = Function("A"), Function("B")
zeta_p, zeta_m = 1, I
c_phi, p_phi = 1, 1

def gamma(l1, l2, l3):
    return sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*pi))

def q_plus(c): return c * (1 + P) / 2
def a(ell, s): return -sqrt((ell - s) * (ell + s + 1) / 2)
def a_plus(ell): return a(ell, 2)
def a_minus(ell): return a(ell, -2)

def W_lens_0(l1, l_out, l2, c):
    return -2 * a(l1, 0) * a(l2, 0) * q_plus(c) * gamma(l1, l_out, l2) * \
           wigner_3j(l_out, l1, l2, 0, -1, 1)

def W_lens_p(l1, l_out, l2, c):
    term1 = a_plus(l2) * wigner_3j(l_out, l1, l2, -2, -1, 3)
    term2 = a_minus(l2) * wigner_3j(l_out, l1, l2, -2, 1, 1)
    return -zeta_p * q_plus(c) * gamma(l1, l_out, l2) * a(l1, 0) * (term1 + term2)

def build_sigma(W1, W2, l_out, c1, c2):
    expr = (1/(2*l_out + 1)) * W1(l1, l_out, l2, c1) * W2(l1, l_out, l2, c2)
    return expr * A(l1) * B(l2)

def build_gamma(W1, W2, l_out, c1, c2):
    expr = (1/(2*l_out + 1)) * W1(l1, l_out, l2, c1) * W2(l2, l_out, l1, c2)
    return expr * A(l1) * B(l2)

kernels_lens = {}
sigma_p_expr = build_sigma(W_lens_p, W_lens_p, l, c_phi, c_phi)
gamma_p_expr = build_gamma(W_lens_p, W_lens_p, l, c_phi, c_phi)
    
kernels_lens["S0"] = build_sigma(W_lens_0, W_lens_0, l, c_phi, c_phi)
kernels_lens["Sp"] = sigma_p_expr
kernels_lens["Sm"] = sigma_p_expr.subs(P, -P)
kernels_lens["Sx"] = build_sigma(W_lens_0, W_lens_p, l, c_phi, c_phi)
kernels_lens["G0"] = build_gamma(W_lens_0, W_lens_0, l, c_phi, c_phi)
kernels_lens["Gp"] = gamma_p_expr
kernels_lens["Gm"] = gamma_p_expr.subs(P, -P)
kernels_lens["Gx"] = build_gamma(W_lens_0, W_lens_p, l, c_phi, c_phi)

compiler = L12SumCompiler(lmax=100, rlmin=1, rlmax=100)

kernels_lens_func = {}
for name, expr in kernels_lens.items():
    print(f"Building {name}: {expr}")
    func, ir = compiler.build_and_compile(expr, args=[l, A, B])
    print(ir)
    kernels_lens_func[name] = func
print("Compilation complete.")
compiler.build_and_compile(kernels_lens["S0"], args=[l, A, B])

#%% test
import numpy as np

ell = np.arange(101)
A = ell**2
B = ell**2
for name, f in kernels_lens_func.items():
    print(f"\n{name}")
    print(f(ell, A, B))


#%% debug
# expr = expand(kernels_lens["G0"])
# terms = []
# for term in expr.as_ordered_terms():
#     terms.append(to_wigd(drop_P(term)))
# expr = sum(terms)
# expr = flip_wigd_order(expr)
# to_wigd(expr)
# expr = expr.doit()
# ana = analyze_wigd_and_group(expr)
# ana