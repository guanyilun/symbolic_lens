"""
End-to-end validation: combine symbolic-compiled kernels via the
Namikawa recipe into full estimator normalizations (qtt, qte, ...)
and compare against the pytempura-validated hand-coded norm_lens.py
at realistic lmax with a real CMB spectrum.

Usage:
    python check_full_norm.py

Saves figures under figures/.
"""
import os
import numpy as np
import jax.numpy as jnp
from sympy import Function, sqrt, pi

from symqe.engine.l12_sum import L12SumCompiler, wigner_3j, l, l1, l2, P
from symqe.reference import norm_lens as nl

# ---- Parameters ----
LMAX  = 3000
RLMIN = 2
RLMAX = 3000
SPEC_PATH = "/home/yguan/software/Symlens.jl/data/cosmo2017_10K_acc3_lensedCls.dat"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Load real CMB spectra ----
# columns: L, TT, EE, BB, TE (all in units of \mu K^2 -- here we just need
# relative amplitudes; overall normalization cancels in qxx).
data = np.loadtxt(SPEC_PATH)
data_L = data[:, 0].astype(int)
assert data_L[0] == 2, f"expected first ell=2, got {data_L[0]}"

# Pad to ell=0..LMAX: Cl[0] = Cl[1] = 0
def pad(col):
    out = np.zeros(LMAX + 1)
    n = min(LMAX - 1, len(col))
    out[2:2 + n] = col[:n]
    return out

cltt = pad(data[:, 1])
clee = pad(data[:, 2])
clbb = pad(data[:, 3])
clte = pad(data[:, 4])

# Dummy noise model matching the Julia demo (demo3_tempura.ipynb):
ell = np.arange(LMAX + 1, dtype=float)
nltt = 10.0 * (1.0 + ell / 1000.0) ** 3                   # \mu K^2 -- T
nlpp = nltt * np.sqrt(2.0)                                #            Pol  (dummy)

ocltt = cltt + nltt
oclee = clee + nlpp
oclbb = clbb + nlpp
oclte = clte            # no noise on TE

ucl = {"TT": cltt, "TE": clte, "EE": clee, "BB": clbb}
ocl = {"TT": ocltt, "TE": oclte, "EE": oclee, "BB": oclbb}

# ---- Build symbolic kernels (once, at LMAX) ----
A_sym, B_sym = Function("A"), Function("B")

def gamma_f(j1, j2, j3):
    return sqrt((2*j1 + 1) * (2*j2 + 1) * (2*j3 + 1) / (4*pi))
def q_plus(c): return c * (1 + P) / 2
def a(e, s): return -sqrt((e - s) * (e + s + 1) / 2)
def a_plus(e): return a(e, 2)
def a_minus(e): return a(e, -2)

# Matches Symlens.jl examples/tempura.jl (Namikawa convention).
# Column-permutation identity: w3j(l1,L,l2;0,1,-1) = P * w3j(L,l1,l2;1,0,-1).
def W_lens_0(l1_, l_out, l2_, c):
    return -2 * a(l_out, 0) * a(l2_, 0) * q_plus(c) * gamma_f(l1_, l_out, l2_) * \
           P * wigner_3j(l_out, l1_, l2_, 1, 0, -1)

def W_lens_p(l1_, l_out, l2_, c):
    t1 = a_plus(l2_) * wigner_3j(l_out, l1_, l2_, 1, 2, -3)
    t2 = c**2 * a_minus(l2_) * wigner_3j(l_out, l1_, l2_, -1, 2, -1)
    return -1 * q_plus(c) * gamma_f(l1_, l_out, l2_) * a(l_out, 0) * P * (t1 + t2)

def build_sigma(W1, W2):
    return (1/(2*l + 1)) * W1(l1, l, l2, 1) * W2(l1, l, l2, 1) * A_sym(l1) * B_sym(l2)

def build_gamma(W1, W2):
    return (1/(2*l + 1)) * W1(l1, l, l2, 1) * W2(l2, l, l1, 1) * A_sym(l1) * B_sym(l2)

print(f"Compiling 8 symbolic kernels at lmax={LMAX} (one-time)...")
sp = build_sigma(W_lens_p, W_lens_p)
gp = build_gamma(W_lens_p, W_lens_p)
kernel_exprs = {
    "S0": build_sigma(W_lens_0, W_lens_0),
    "Sp": sp,
    "Sm": sp.subs(P, -P),
    "Sx": build_sigma(W_lens_0, W_lens_p),
    "G0": build_gamma(W_lens_0, W_lens_0),
    "Gp": gp,
    "Gm": gp.subs(P, -P),
    "Gx": build_gamma(W_lens_0, W_lens_p),
}
compiler = L12SumCompiler(lmax=LMAX, rlmin=RLMIN, rlmax=RLMAX)
kern_fn = {}
for name, expr in kernel_exprs.items():
    print(f"  {name}...", flush=True)
    fn, _ = compiler.build_and_compile(expr, args=[l, A_sym, B_sym])
    kern_fn[name] = fn
print("Compilation complete.\n")

def K(name, A, B):
    return np.asarray(kern_fn[name](ell, np.asarray(A, float), np.asarray(B, float)))

# ---- End-to-end recipes (identical to norm_lens.py / tempura.jl) ----

def qtt_sym():
    res  = K("S0", 1/ocl["TT"],         ucl["TT"]**2/ocl["TT"])
    res += K("G0", ucl["TT"]/ocl["TT"], ucl["TT"]/ocl["TT"])
    return 1/res

def qte_sym():
    res  = K("S0", 1/ocl["TT"],         ucl["TE"]**2/ocl["EE"])
    res += 2*K("Gx", ucl["TE"]/ocl["TT"], ucl["TE"]/ocl["EE"])
    res += K("Sp", 1/ocl["EE"],         ucl["TE"]**2/ocl["TT"])
    return 1/res

def qtb_sym():
    return 1/K("Sm", 1/ocl["BB"], ucl["TE"]**2/ocl["TT"])

def qee_sym():
    res  = K("Sp", 1/ocl["EE"],         ucl["EE"]**2/ocl["EE"])
    res += K("Gp", ucl["EE"]/ocl["EE"], ucl["EE"]/ocl["EE"])
    return 1/res

def qbb_sym():
    res  = K("Sp", 1/ocl["BB"],         ucl["BB"]**2/ocl["BB"])
    res += K("Gp", ucl["BB"]/ocl["BB"], ucl["BB"]/ocl["BB"])
    return 1/res

def qeb_sym():
    res  = K("Sm", 1/ocl["EE"],         ucl["BB"]**2/ocl["BB"])
    res += 2*K("Gm", ucl["BB"]/ocl["BB"], ucl["EE"]/ocl["EE"])
    res += K("Sm", 1/ocl["BB"],         ucl["EE"]**2/ocl["EE"])
    return 1/res

print("Evaluating symbolic estimator normalizations...")
sym = {
    "qtt": qtt_sym(),
    "qte": qte_sym(),
    "qtb": qtb_sym(),
    "qee": qee_sym(),
    "qbb": qbb_sym(),
    "qeb": qeb_sym(),
}

print("Evaluating hand-coded estimator normalizations...")
hand = {
    "qtt": np.asarray(nl.qtt(LMAX, RLMIN, RLMAX, ucl, ocl)),
    "qte": np.asarray(nl.qte(LMAX, RLMIN, RLMAX, ucl, ocl)),
    "qtb": np.asarray(nl.qtb(LMAX, RLMIN, RLMAX, ucl, ocl)),
    "qee": np.asarray(nl.qee(LMAX, RLMIN, RLMAX, ucl, ocl)),
    "qbb": np.asarray(nl.qbb(LMAX, RLMIN, RLMAX, ucl, ocl)),
    "qeb": np.asarray(nl.qeb(LMAX, RLMIN, RLMAX, ucl, ocl)),
}

# ---- Optional pytempura ----
tp = None
try:
    import pytempura as tp_mod
    print("Evaluating pytempura estimator normalizations...")
    tp = {
        "qtt": np.asarray(tp_mod.norm_lens.qtt(LMAX, RLMIN, RLMAX, cltt, cltt, ocltt)[0]),
        "qte": np.asarray(tp_mod.norm_lens.qte(LMAX, RLMIN, RLMAX, clte, clte, ocltt, oclee)[0]),
        "qtb": np.asarray(tp_mod.norm_lens.qtb(LMAX, RLMIN, RLMAX, clte, ocltt, oclbb)[0]),
        "qee": np.asarray(tp_mod.norm_lens.qee(LMAX, RLMIN, RLMAX, clee, clee, oclee)[0]),
        "qbb": np.asarray(tp_mod.norm_lens.qbb(LMAX, RLMIN, RLMAX, clbb, clbb, oclbb)[0]),
        "qeb": np.asarray(tp_mod.norm_lens.qeb(LMAX, RLMIN, RLMAX, clee, clbb, oclee, oclbb)[0]),
    }
except Exception as e:
    print(f"(pytempura unavailable: {e})")

# ---- Summary table ----
band = (ell >= 10) & (ell <= LMAX)
print("\nMax |reldiff| over L in [10, LMAX]:")
for name in sym:
    s, h = sym[name], hand[name]
    m = band & np.isfinite(s) & np.isfinite(h) & (np.abs(s) > 1e-40)
    rel_hs = np.abs((h[m] - s[m]) / s[m]).max() if m.any() else np.nan
    msg = f"  {name:4s}  hand vs sym: {rel_hs:.2e}"
    if tp is not None:
        t = tp[name]
        m2 = m & np.isfinite(t) & (np.abs(t) > 1e-40)
        rel_st = np.abs((s[m2] - t[m2]) / t[m2]).max() if m2.any() else np.nan
        msg += f"    sym vs tp: {rel_st:.2e}"
    print(msg)

# ---- Plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = list(sym.keys())
# Convert normalizations into "L^4 N_L^phi" style curves (standard convention
# when plotting lensing reconstruction noise).
pre = ell ** 4

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Lensing reconstruction noise  $L^4 N_L^\\phi$   (lmax={LMAX}, rlmax={RLMAX})",
             fontsize=14)
for idx, name in enumerate(names):
    ax = axes[idx // 3, idx % 3]
    mask = (ell >= 2)
    ax.plot(ell[mask], hand[name][mask] * pre[mask], "-",
            label="hand (norm_lens.py)", lw=2, alpha=0.8)
    ax.plot(ell[mask], sym[name][mask] * pre[mask], "--",
            label="symbolic compiler", lw=1.5, alpha=0.9)
    if tp is not None:
        ax.plot(ell[mask], tp[name][mask] * pre[mask], ":",
                label="pytempura", lw=1.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, LMAX)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$L^4\, N_L^{\phi}$")
    ax.set_title(name.upper())
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "nlphi_hand_vs_sym.png"), dpi=150)
print(f"\nSaved: {FIG_DIR}/nlphi_hand_vs_sym.png")

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5.5))
for name in names:
    s, h = sym[name], hand[name]
    mask = (ell >= 2) & np.isfinite(s) & np.isfinite(h) & (np.abs(s) > 1e-40)
    reldiff = np.abs((h[mask] - s[mask]) / s[mask])
    ax2.loglog(ell[mask], reldiff, "-", label=name, alpha=0.85)
ax2.axhline(1e-12, color="k", ls=":", lw=0.8, label=r"$10^{-12}$")
ax2.set_xlim(2, LMAX)
ax2.set_xlabel(r"$L$")
ax2.set_ylabel(r"$|N_L^{\rm hand} - N_L^{\rm sym}| / |N_L^{\rm sym}|$")
ax2.set_title("Relative difference: hand-coded vs symbolic-compiled")
ax2.legend(fontsize=9, ncol=3, loc="lower right")
ax2.grid(alpha=0.3, which="both")
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "nlphi_reldiff.png"), dpi=150)
print(f"Saved: {FIG_DIR}/nlphi_reldiff.png")
