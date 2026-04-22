"""
Combine the (validated) symbolic-compiled kernels S0/Sp/Sm/Sx/G0/Gp/Gm/Gx
into full estimator normalizations (qtt, qte, ...) using the same recipe
norm_lens.py uses, and compare against the hand-coded norm_lens.qxx.

Memory claimed both paths agree after the recipe combination. This test
settles it.
"""
import numpy as np
import jax.numpy as jnp
from sympy import Function, sqrt, pi, I

from sym_utils.l12_sum import L12SumCompiler, wigner_3j, l, l1, l2, P
import norm_lens as nl

lmax = 60
rlmin = 2
rlmax = 50

ell = np.arange(lmax + 1, dtype=float)

# --- Build & compile the 8 symbolic lensing kernels (same as test_norm_sym.py) ---
A_sym, B_sym = Function("A"), Function("B")

def gamma_f(l1, l2, l3):
    return sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*pi))

def q_plus(c): return c * (1 + P) / 2
def a(ell, s): return -sqrt((ell - s) * (ell + s + 1) / 2)
def a_plus(ell): return a(ell, 2)
def a_minus(ell): return a(ell, -2)

def W_lens_0(l1_, l_out, l2_, c):
    # Julia-exact 3j via column swap: w3j(l1,L,l2;0,1,-1) = P * w3j(L,l1,l2;1,0,-1)
    return -2 * a(l_out, 0) * a(l2_, 0) * q_plus(c) * gamma_f(l1_, l_out, l2_) * \
           P * wigner_3j(l_out, l1_, l2_, 1, 0, -1)

def W_lens_p(l1_, l_out, l2_, c):
    # w3j(l1,L,l2;2,1,-3) = P * w3j(L,l1,l2;1,2,-3); w3j(l1,L,l2;2,-1,-1) = P * w3j(L,l1,l2;-1,2,-1)
    t1 = a_plus(l2_) * wigner_3j(l_out, l1_, l2_, 1, 2, -3)
    t2 = c**2 * a_minus(l2_) * wigner_3j(l_out, l1_, l2_, -1, 2, -1)
    return -1 * q_plus(c) * gamma_f(l1_, l_out, l2_) * a(l_out, 0) * P * (t1 + t2)

def build_sigma(W1, W2):
    return (1/(2*l + 1)) * W1(l1, l, l2, 1) * W2(l1, l, l2, 1) * A_sym(l1) * B_sym(l2)

def build_gamma(W1, W2):
    return (1/(2*l + 1)) * W1(l1, l, l2, 1) * W2(l2, l, l1, 1) * A_sym(l1) * B_sym(l2)

print("Building symbolic expressions...")
exprs = {}
sp = build_sigma(W_lens_p, W_lens_p)
gp = build_gamma(W_lens_p, W_lens_p)
exprs["S0"] = build_sigma(W_lens_0, W_lens_0)
exprs["Sp"] = sp
exprs["Sm"] = sp.subs(P, -P)
exprs["Sx"] = build_sigma(W_lens_0, W_lens_p)
exprs["G0"] = build_gamma(W_lens_0, W_lens_0)
exprs["Gp"] = gp
exprs["Gm"] = gp.subs(P, -P)
exprs["Gx"] = build_gamma(W_lens_0, W_lens_p)

print(f"Compiling kernels at lmax={lmax}, rlmin={rlmin}, rlmax={rlmax}...")
compiler = L12SumCompiler(lmax=lmax, rlmin=rlmin, rlmax=rlmax)
kern = {}
for name, expr in exprs.items():
    print(f"  {name}...", flush=True)
    fn, _ = compiler.build_and_compile(expr, args=[l, A_sym, B_sym])
    kern[name] = fn
print("Compilation complete.\n")

def K(name, A, B):
    return np.asarray(kern[name](ell, np.asarray(A, float), np.asarray(B, float)))

# --- Plausible spectra ---
cltt = 100.0 / (ell + 10)**2 + 1e-6
clee = 40.0  / (ell + 10)**2 + 1e-6
clbb = 5.0   / (ell + 20)**2 + 1e-6
clte = 0.5 * np.sqrt(cltt * clee)
nltt = 1e-3 * np.ones_like(ell)
nlee = 2e-3 * np.ones_like(ell)
nlbb = 2e-3 * np.ones_like(ell)

ucl = {"TT": cltt, "TE": clte, "EE": clee, "BB": clbb}
ocl = {"TT": cltt+nltt, "TE": clte, "EE": clee+nlee, "BB": clbb+nlbb}

# --- End-to-end combinations using validated symbolic kernels ---
def qtt_sym():
    A = 1/ocl["TT"]; B = ucl["TT"]**2/ocl["TT"]
    res = K("S0", A, B)
    A = ucl["TT"]/ocl["TT"]
    res = res + K("G0", A, A)
    return 1/res

def qte_sym():
    A = 1/ocl["TT"]; B = ucl["TE"]**2/ocl["EE"]
    res = K("S0", A, B)
    A = ucl["TE"]/ocl["TT"]; B = ucl["TE"]/ocl["EE"]
    res = res + 2*K("Gx", A, B)
    A = 1/ocl["EE"]; B = ucl["TE"]**2/ocl["TT"]
    res = res + K("Sp", A, B)
    return 1/res

def qtb_sym():
    A = 1/ocl["BB"]; B = ucl["TE"]**2/ocl["TT"]
    return 1/K("Sm", A, B)

def qee_sym():
    A = 1/ocl["EE"]; B = ucl["EE"]**2/ocl["EE"]
    res = K("Sp", A, B)
    A = ucl["EE"]/ocl["EE"]
    res = res + K("Gp", A, A)
    return 1/res

def qbb_sym():
    A = 1/ocl["BB"]; B = ucl["BB"]**2/ocl["BB"]
    res = K("Sp", A, B)
    A = ucl["BB"]/ocl["BB"]
    res = res + K("Gp", A, A)
    return 1/res

def qeb_sym():
    A = 1/ocl["EE"]; B = ucl["BB"]**2/ocl["BB"]
    res = K("Sm", A, B)
    A = ucl["BB"]/ocl["BB"]; B = ucl["EE"]/ocl["EE"]
    res = res + 2*K("Gm", A, B)
    A = 1/ocl["BB"]; B = ucl["EE"]**2/ocl["EE"]
    res = res + K("Sm", A, B)
    return 1/res

sym_results = {
    "qtt": qtt_sym(), "qte": qte_sym(), "qtb": qtb_sym(),
    "qee": qee_sym(), "qbb": qbb_sym(), "qeb": qeb_sym(),
}

# --- Hand-coded results ---
hand_results = {
    "qtt": np.asarray(nl.qtt(lmax, rlmin, rlmax, ucl, ocl)),
    "qte": np.asarray(nl.qte(lmax, rlmin, rlmax, ucl, ocl)),
    "qtb": np.asarray(nl.qtb(lmax, rlmin, rlmax, ucl, ocl)),
    "qee": np.asarray(nl.qee(lmax, rlmin, rlmax, ucl, ocl)),
    "qbb": np.asarray(nl.qbb(lmax, rlmin, rlmax, ucl, ocl)),
    "qeb": np.asarray(nl.qeb(lmax, rlmin, rlmax, ucl, ocl)),
}

# --- Pytempura as third anchor when available ---
try:
    import pytempura as tp
    tp_results = {}
    uT = ucl["TT"]; uE = ucl["EE"]; uB = ucl["BB"]; uX = ucl["TE"]
    oT = ocl["TT"]; oE = ocl["EE"]; oB = ocl["BB"]; oX = ocl["TE"]
    tp_results["qtt"] = np.asarray(tp.norm_lens.qtt(lmax, rlmin, rlmax, uT, uT, oT)[0])
    tp_results["qte"] = np.asarray(tp.norm_lens.qte(lmax, rlmin, rlmax, uX, uX, oT, oE)[0])
    tp_results["qtb"] = np.asarray(tp.norm_lens.qtb(lmax, rlmin, rlmax, uX, oT, oB)[0])
    tp_results["qee"] = np.asarray(tp.norm_lens.qee(lmax, rlmin, rlmax, uE, uE, oE)[0])
    tp_results["qbb"] = np.asarray(tp.norm_lens.qbb(lmax, rlmin, rlmax, uB, uB, oB)[0])
    tp_results["qeb"] = np.asarray(tp.norm_lens.qeb(lmax, rlmin, rlmax, uE, uB, oE, oB)[0])
    have_tp = True
except Exception as e:
    print(f"(pytempura unavailable: {e})")
    tp_results = None
    have_tp = False

print()
print("="*100)
header = f"{'est':5s}  {'L':>3s}  {'hand':>15s}  {'sym(recipe)':>15s}  {'ratio h/s':>10s}"
if have_tp:
    header += f"  {'pytempura':>15s}  {'h/tp':>8s}  {'s/tp':>8s}"
print(header)
print("="*100)

for name in sym_results:
    h = hand_results[name]
    s = sym_results[name]
    t = tp_results[name] if have_tp else None
    print(f"--- {name} ---")
    for L in range(lmax + 1):
        if L < 2 or not (np.isfinite(h[L]) and np.isfinite(s[L])):
            continue
        if L % max(1, lmax // 15) and L != lmax:
            continue
        r_hs = h[L]/s[L] if abs(s[L]) > 1e-30 else np.nan
        line = f"  {name:5s} {L:3d}  {h[L]:15.6e}  {s[L]:15.6e}  {r_hs:10.4f}"
        if have_tp:
            r_ht = h[L]/t[L] if abs(t[L]) > 1e-30 else np.nan
            r_st = s[L]/t[L] if abs(t[L]) > 1e-30 else np.nan
            line += f"  {t[L]:15.6e}  {r_ht:8.4f}  {r_st:8.4f}"
        print(line)

print("\nMax |reldiff| over L in [10, lmax]:")
band = (ell >= 10) & (ell <= lmax)
summary_reldiff = {}
for name in sym_results:
    h = hand_results[name]; s = sym_results[name]
    m = band & (np.abs(s) > 1e-30) & np.isfinite(h) & np.isfinite(s)
    if m.sum() == 0:
        print(f"  {name}: no usable range"); continue
    rel = np.abs((h[m] - s[m]) / s[m]).max()
    summary_reldiff[name] = rel
    msg = f"  {name:5s}  hand vs sym(recipe): {rel:.3e}"
    if have_tp:
        t = tp_results[name]
        m2 = m & np.isfinite(t) & (np.abs(t) > 1e-30)
        if m2.any():
            rht = np.abs((h[m2]-t[m2])/t[m2]).max()
            rst = np.abs((s[m2]-t[m2])/t[m2]).max()
            msg += f"   hand vs tp: {rht:.3e}   sym(recipe) vs tp: {rst:.3e}"
    print(msg)


# ---- Plots ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

est_names = list(sym_results.keys())

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("End-to-end normalization: hand-coded vs symbolic-compiled (recipe-combined)",
             fontsize=13)
for idx, name in enumerate(est_names):
    ax = axes[idx // 3, idx % 3]
    h = hand_results[name]; s = sym_results[name]
    mask = (ell >= 2) & np.isfinite(h) & np.isfinite(s)
    ax.plot(ell[mask], np.abs(h[mask]), '-',  label='hand (norm_lens.py)', lw=2)
    ax.plot(ell[mask], np.abs(s[mask]), '--', label='symbolic (recipe)', lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel(r'$L$')
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comp_full_norm.png", dpi=150)
print("\nSaved: comp_full_norm.png")

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
for name in est_names:
    h = hand_results[name]; s = sym_results[name]
    mask = (ell >= 2) & np.isfinite(h) & np.isfinite(s) & (np.abs(s) > 1e-30)
    reldiff = np.abs((h[mask] - s[mask]) / s[mask])
    ax2.semilogy(ell[mask], reldiff, '-', label=name, alpha=0.8)
ax2.axhline(1e-13, color='k', ls=':', lw=0.8, label='1e-13 (machine precision)')
ax2.set_xlabel(r'$L$')
ax2.set_ylabel(r'$|hand - sym| / |sym|$')
ax2.set_title("Relative difference: hand-coded vs symbolic-compiled")
ax2.legend(fontsize=8, ncol=3)
ax2.set_ylim(1e-17, 1e-10)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comp_full_norm_reldiff.png", dpi=150)
print("Saved: comp_full_norm_reldiff.png")
