#%%
"""
Per-triangle sanity test: each of the 8 basis kernels (S0/Sp/Sm/Sx/G0/Gp/Gm/Gx)
computed by the symbolic compiler must equal the direct double-sum over
(l1, l2) using exact sympy.physics.wigner 3j symbols.

This is a cheap but exhaustive ground-truth check: it pins down the
symbolic compiler at the atomic level before any estimator-recipe
composition. Limited to small lmax because sympy's exact 3j evaluator
is slow (the end-to-end check in check_full_norm.py runs at lmax=3000
against pytempura-validated hand-coded kernels).
"""
from sympy import Function, sqrt, pi, I
from sympy.physics.wigner import wigner_3j as w3j_exact
from symqe.engine.l12_sum import NormCompiler, wigner_3j, l, l1, l2, P
import numpy as np

# ---- Build symbolic kernels ----
A_sym, B_sym = Function("A"), Function("B")
zeta_p, zeta_m = 1, I
c_phi, p_phi = 1, 1

def gamma_f(l1, l2, l3):
    return sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*pi))

def q_plus(c): return c * (1 + P) / 2
def a(ell, s): return -sqrt((ell - s) * (ell + s + 1) / 2)
def a_plus(ell): return a(ell, 2)
def a_minus(ell): return a(ell, -2)

# Julia Symlens.jl/examples/tempura.jl defines
#   Wₓ⁰(ℓ₁,ℓ₂,ℓ₃,c) = -2*a(ℓ₂,0)*a(ℓ₃,0)*qₓ⁺(c)*γ*w3j(ℓ₁,ℓ₂,ℓ₃,0,1,-1)
# called with (ℓ₁=l1, ℓ₂=L, ℓ₃=l2). The compiler here expects the output L as the
# first 3j column, so we move it there via column swap:
#   w3j(l1,L,l2; 0,1,-1) = (-1)^(l+l1+l2) * w3j(L,l1,l2; 1,0,-1) = P * w3j(L,l1,l2; 1,0,-1)
def W_lens_0(l1, l_out, l2, c):
    return -2 * a(l_out, 0) * a(l2, 0) * q_plus(c) * gamma_f(l1, l_out, l2) * \
           P * wigner_3j(l_out, l1, l2, 1, 0, -1)

# Wₓ⁺(ℓ₁,ℓ₂,ℓ₃,c) = -qₓ⁺(c)*γ*a(ℓ₂,0)*(a⁺(ℓ₃)*w3j(ℓ₁,ℓ₂,ℓ₃,2,1,-3) + c²*a⁻(ℓ₃)*w3j(ℓ₁,ℓ₂,ℓ₃,2,-1,-1))
# After column swap each 3j gets a P factor:
#   w3j(l1,L,l2; 2, 1,-3) = P * w3j(L,l1,l2; 1, 2,-3)
#   w3j(l1,L,l2; 2,-1,-1) = P * w3j(L,l1,l2;-1, 2,-1)
def W_lens_p(l1, l_out, l2, c):
    term1 = a_plus(l2) * wigner_3j(l_out, l1, l2, 1, 2, -3)
    term2 = c**2 * a_minus(l2) * wigner_3j(l_out, l1, l2, -1, 2, -1)
    return -zeta_p * q_plus(c) * gamma_f(l1, l_out, l2) * a(l_out, 0) * P * (term1 + term2)

def build_sigma(W1, W2, l_out, c1, c2):
    expr = (1/(2*l_out + 1)) * W1(l1, l_out, l2, c1) * W2(l1, l_out, l2, c2)
    return expr * A_sym(l1) * B_sym(l2)

def build_gamma(W1, W2, l_out, c1, c2):
    expr = (1/(2*l_out + 1)) * W1(l1, l_out, l2, c1) * W2(l2, l_out, l1, c2)
    return expr * A_sym(l1) * B_sym(l2)

# ---- Brute-force evaluation of kernel expression ----
def eval_kernel_brute(kernel_name, lmax, rlmin, rlmax, A_arr, B_arr):
    """Brute-force evaluate a kernel by summing over l1, l2 with exact 3j symbols."""
    ell = np.arange(lmax + 1, dtype=float)

    def gam_num(l1v, Lv, l2v):
        return np.sqrt((2*l1v+1)*(2*Lv+1)*(2*l2v+1)/(4*np.pi))

    def a_num(lv, s):
        val = (lv - s)*(lv + s + 1)/2
        return -np.sqrt(max(0, val))

    def a_plus_num(lv): return a_num(lv, 2)
    def a_minus_num(lv): return a_num(lv, -2)

    # Mirrors the symbolic W definitions (Julia tempura.jl style) EXACTLY:
    #   W_lens_0(j1,j2,j3) = -2*a(j2,0)*a(j3,0)*(1+P)/2*γ*w3j(j1,j2,j3;0,1,-1)
    #   W_lens_p(j1,j2,j3) = -(1+P)/2*γ*a(j2,0)*[a⁺(j3)*w3j(j1,j2,j3;2,1,-3)+a⁻(j3)*w3j(j1,j2,j3;2,-1,-1)]
    # Σ⁰(L) = (1/(2L+1)) Σ_{l1,l2} W(l1,L,l2)·W(l1,L,l2) A(l1) B(l2)
    # Γ⁰(L) = (1/(2L+1)) Σ_{l1,l2} W(l1,L,l2)·W(l2,L,l1) A(l1) B(l2)
    # Mirror Julia tempura.jl exactly — the (-1)^S factor from column swap cancels
    # against P in the symbolic form, so the actual summed quantity is:
    #   W_lens_0(l1,L,l2) = -2·a(L,0)·a(l2,0)·(1+P)/2·γ·w3j(l1,L,l2; 0, 1,-1)
    #   W_lens_p(l1,L,l2) = -(1+P)/2·γ·a(L,0)·[a⁺(l2)·w3j(l1,L,l2;2,1,-3) + a⁻(l2)·w3j(l1,L,l2;2,-1,-1)]
    def W0(j1, L, j3):
        Pv = (-1)**(j1 + L + j3)
        return -2*a_num(L,0)*a_num(j3,0)*(1+Pv)/2*gam_num(j1,L,j3)*float(w3j_exact(j1,L,j3,0,1,-1))

    def Wp(j1, L, j3, sign=1):
        Pv = (-1)**(j1 + L + j3)
        qp = (1 + sign*Pv)/2
        w3a = float(w3j_exact(j1, L, j3, 2, 1, -3))
        w3b = float(w3j_exact(j1, L, j3, 2, -1, -1))
        return -qp*gam_num(j1,L,j3)*a_num(L,0)*(a_plus_num(j3)*w3a + a_minus_num(j3)*w3b)

    result = np.zeros(lmax + 1)
    for L in range(lmax + 1):
        total = 0.0
        for l1v in range(rlmin, rlmax + 1):
            for l2v in range(rlmin, rlmax + 1):
                if kernel_name == "S0":
                    Wsq = W0(l1v, L, l2v)**2
                elif kernel_name == "Sp":
                    Wsq = Wp(l1v, L, l2v, +1)**2
                elif kernel_name == "Sm":
                    Wsq = Wp(l1v, L, l2v, -1)**2
                elif kernel_name == "Sx":
                    Wsq = W0(l1v, L, l2v) * Wp(l1v, L, l2v, +1)
                elif kernel_name == "G0":
                    Wsq = W0(l1v, L, l2v) * W0(l2v, L, l1v)
                elif kernel_name == "Gp":
                    Wsq = Wp(l1v, L, l2v, +1) * Wp(l2v, L, l1v, +1)
                elif kernel_name == "Gm":
                    Wsq = Wp(l1v, L, l2v, -1) * Wp(l2v, L, l1v, -1)
                elif kernel_name == "Gx":
                    Wsq = W0(l1v, L, l2v) * Wp(l2v, L, l1v, +1)
                else:
                    raise ValueError(f"Unknown kernel: {kernel_name}")

                total += Wsq / (2*L + 1) * A_arr[l1v] * B_arr[l2v]
        result[L] = total
    return result

# Parameters (small for brute force)
# Use rlmax < lmax to avoid GL quadrature boundary errors
lmax = 10
rlmin = 1
rlmax = 8

print("Building symbolic expressions...")
kernels_sym_expr = {}
sigma_p_expr = build_sigma(W_lens_p, W_lens_p, l, c_phi, c_phi)
gamma_p_expr = build_gamma(W_lens_p, W_lens_p, l, c_phi, c_phi)

kernels_sym_expr["S0"] = build_sigma(W_lens_0, W_lens_0, l, c_phi, c_phi)
kernels_sym_expr["Sp"] = sigma_p_expr
kernels_sym_expr["Sm"] = sigma_p_expr.subs(P, -P)
kernels_sym_expr["Sx"] = build_sigma(W_lens_0, W_lens_p, l, c_phi, c_phi)
kernels_sym_expr["G0"] = build_gamma(W_lens_0, W_lens_0, l, c_phi, c_phi)
kernels_sym_expr["Gp"] = gamma_p_expr
kernels_sym_expr["Gm"] = gamma_p_expr.subs(P, -P)
kernels_sym_expr["Gx"] = build_gamma(W_lens_0, W_lens_p, l, c_phi, c_phi)

print("Compiling symbolic kernels...")
compiler = NormCompiler(lmax=lmax, rlmin=rlmin, rlmax=rlmax)
kernels_sym_func = {}
for name, expr in kernels_sym_expr.items():
    print(f"  Compiling {name}...")
    func, ir = compiler.build_and_compile(expr, args=[l, A_sym, B_sym])
    kernels_sym_func[name] = func
print("Compilation complete.\n")

# ---- Test inputs ----
ell = np.arange(lmax + 1, dtype=float)
A_test = ell * (ell + 1)
B_test = ell * (ell + 1)

# ---- Evaluate both ----
print("Evaluating symbolic kernels...")
sym_results = {}
for name, f in kernels_sym_func.items():
    sym_results[name] = np.array(f(ell, A_test, B_test))

print("Evaluating brute-force kernels (this may take a moment)...")
brute_results = {}
kernel_names = ["S0", "Sp", "Sm", "Sx", "G0", "Gp", "Gm", "Gx"]
for name in kernel_names:
    print(f"  {name}...", end=" ", flush=True)
    brute_results[name] = eval_kernel_brute(name, lmax, rlmin, rlmax, A_test, B_test)
    print("done")

# ---- Compare ----
print("\n" + "="*60)
print("Comparison: symbolic compiled vs brute-force 3j summation")
print(f"lmax={lmax}, rlmin={rlmin}, rlmax={rlmax}")
print(f"A = ell*(ell+1), B = ell*(ell+1)")
print("="*60)

all_pass = True
for name in kernel_names:
    s = sym_results[name]
    b = brute_results[name]

    mask = (np.abs(b) > 1e-20) & (ell > 1)
    if mask.sum() == 0:
        print(f"{name}: no overlapping non-zero values")
        continue

    if np.allclose(s[mask], b[mask], rtol=1e-8):
        max_reldiff = np.max(np.abs(s[mask] - b[mask]) / np.abs(b[mask]))
        print(f"{name}: PASS (max relative diff = {max_reldiff:.2e})")
    else:
        ratio = s[mask] / b[mask]
        print(f"{name}: FAIL (ratio range = [{ratio.min():.4f}, {ratio.max():.4f}])")
        all_pass = False

print()
if all_pass:
    print("ALL KERNELS PASS - symbolic compiler matches brute-force 3j.")
else:
    print("SOME KERNELS FAILED - check symbolic compiler.")
    raise SystemExit(1)
