"""Grammar-enumerator search over CMB lensing TT weight candidates,
with symmetrization + parity enumeration + reference injection.

Pipeline:
  enumerate_candidates(..., symmetrize_l1l2=True,
                            parity_factors=(1, P, 1+P, 1-P))
  → (inject reference candidates: Hu-Okamoto f_TT correct + variants)
  → compile_qe_from_f
  → fisher_fom
  → rank

The parity factors (1+P) and (1-P) are Namikawa's q_plus / q_minus
selectors (up to scale), necessary to capture the W^{x,+} / W^{x,-}
structure of physical QEs.  symmetrize_l1l2 emits ``expr + swap(expr)``
so that TT/EE/BB-style symmetric weights can be constructed in one
enumeration step.

After enumeration, we inject three hand-picked reference candidates
(Hu-Okamoto f_TT correct and two structural perturbations) so we can
report where KNOWN physics lands in the sorted ranking — a direct
test of whether the pipeline recovers the canonical weight.
"""
import time
import numpy as np
from sympy import sympify

import symqe as sq
from symqe import l, l1, l2, wigner_3j, P, hCT, CT, gamma_f, a, q_plus


LMAX  = 100
NSIDE = 64
L_LO, L_HI, L_REF = 20, 80, 40

ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt       = cltt_theory + 1.0
px = sq.Pixelization(nside=NSIDE)


# --- reference candidates (Hu-Okamoto f_TT + structural variants) -----
def swap_l12(expr):
    return expr.subs({l1: l2, l2: l1}, simultaneous=True)


def f_TT_correct():
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


def f_TT_wrong_ladder():
    W = -2 * a(l, 2) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 1, 0, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


def f_TT_wrong_m():
    W = -2 * a(l1, 0) * a(l2, 0) * q_plus(1) * gamma_f(l1, l, l2) * P \
        * wigner_3j(l, l1, l2, 0, 1, -1)
    return W * CT(l2) + swap_l12(W) * CT(l1)


REFERENCES = [
    ("f_TT (Hu-Okamoto correct)", f_TT_correct()),
    ("f_TT wrong ladder",         f_TT_wrong_ladder()),
    ("f_TT wrong 3j m-row",       f_TT_wrong_m()),
]


# --- enumerate ---------------------------------------------------------
print("=" * 78)
print("Grammar-enumerator search — CMB lensing TT (symmetrized + parity)")
print("=" * 78)

t0 = time.time()
candidates = list(sq.enumerate_candidates(
    m_max=2,
    coeffs=(1,),                                  # sign cancels in FOM
    parity_factors=(sympify(1), 1 + P),           # 1 covers P^0; 1+P is q_plus
    max_leg_factors=2,
    ladder_spins=(0, 2, -2),                      # typical lensing-T ladders
    spectra_X={"CT": CT},
    spectra_Y={"CT": CT},
    restrict_m_L=(-1, 1),                         # lensing-vector output
    symmetrize_l1l2=True,                         # f_TT-style l1↔l2 sums
))
t1 = time.time()
print(f"  enumerated {len(candidates)} distinct candidates in {t1-t0:.1f}s")

# Inject references (deduped against enumerated pool via canonical_key).
enum_keys = {sq.canonical_key(e): True for _, e in candidates}
injected = []
for label, expr in REFERENCES:
    k = sq.canonical_key(expr)
    already = enum_keys.get(k, False)
    injected.append(({"label": label, "flavor": "reference",
                      "already_in_enum": already}, expr))

print(f"  injected {len(injected)} reference candidates "
      f"({sum(1 for m,_ in injected if m['already_in_enum'])} "
      f"already present via enumeration)")
print()

all_candidates = candidates + injected


# --- score -------------------------------------------------------------
print(f"  scoring {len(all_candidates)} at lmax={LMAX}, "
      f"L ∈ [{L_LO}, {L_HI}], L_ref={L_REF}...")
t2 = time.time()
results = sq.score_candidates(
    all_candidates,
    lmax=LMAX, Delta=2,
    hCxx=hCT, hCyy=hCT,
    user_funcs=[hCT, CT],
    spectra_args=(ocltt, cltt_theory),
    L_range=range(L_LO, L_HI + 1),
    L_ref=L_REF,
    px=px,
    progress=True,
)
t3 = time.time()
print(f"  scored in {t3-t2:.1f}s ({(t3-t2)/len(all_candidates):.2f}s each)")
print()


# --- report top 20 + rank of each reference ---------------------------
def summarize_m(meta):
    if "m_triple" not in meta:
        return meta.get("label", "(ref)")[:22]
    mL, mX, mY = meta["m_triple"]
    return f"m=({mL:+d},{mX:+d},{mY:+d})"


def summarize_filter(f):
    s = str(f)
    if len(s) > 30:
        s = s[:27] + "..."
    return s


n_valid = sum(1 for r in results if np.isfinite(r[0]))
print(f"{n_valid} / {len(results)} candidates produced finite FOM.")
print()

N_SHOW = 20
print(f"Top {N_SHOW} by gauge-fixed Fisher FOM:")
print(f"  {'rk':>3s}  {'FOM':>10s}  flavor       {'m / label':>22s}  "
      f"{'X(l1)':>30s}  {'Y(l2)':>30s}  parity")
print("-" * 125)
for i, (fom, meta, expr) in enumerate(results[:N_SHOW]):
    flavor = meta.get("flavor", "?")
    parity_s = str(meta.get("parity", "--"))[:8]
    Xs = summarize_filter(meta.get("X_filter", "--"))
    Ys = summarize_filter(meta.get("Y_filter", "--"))
    print(f"  {i+1:>3d}  {fom:>10.3e}  {flavor:10s}  {summarize_m(meta):>22s}  "
          f"{Xs:>30s}  {Ys:>30s}  {parity_s}")

print()
print("Reference ranks:")
for j, (meta, expr) in enumerate(injected):
    for i, (fom, m, _) in enumerate(results):
        if m is meta:
            rel = (1 + i) / len(results)
            tag = " (structurally duplicated enum candidate)" \
                  if meta["already_in_enum"] else ""
            print(f"  {meta['label']:32s}  rank {i+1:>5d}/{len(results)}  "
                  f"(top {rel*100:4.1f}%)  FOM={fom:.3e}{tag}")
            break
