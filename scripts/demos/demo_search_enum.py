"""Grammar-enumerator search over CMB lensing TT weight candidates.

End-to-end:
  enumerate_candidates → compile_qe_from_f → fisher_fom → rank.

This is the first session actually searching the symbolic DSL (not
ranking hand-picked candidates).  The grammar is tight — restricted
to TT input (CT spectrum only), mL ∈ {-1, +1} (lensing vector output),
coeffs ±1, ±i, and ≤ 2 factors per leg from the ladder / CT vocabulary.
Still big enough to exercise dedup and to expose what the top ranker
finds.

Note on interpretation: even with the gauge fix, pure Fisher cannot
distinguish "right physics measured well" from "wrong physics measured
well".  What we're validating here is that the PIPELINE runs and that
the top candidates look structurally similar to Hu-Okamoto f_TT.
"""
import time
import numpy as np
import symqe as sq


LMAX  = 100
NSIDE = 64
L_LO, L_HI, L_REF = 20, 80, 40

ell = np.arange(LMAX + 1, dtype=float)
cltt_theory = 5e3 / (ell + 10) ** 2 + 1e-2
ocltt       = cltt_theory + 1.0
px = sq.Pixelization(nside=NSIDE)


print("=" * 78)
print("Grammar-enumerator search — CMB lensing TT")
print("=" * 78)

t0 = time.time()
candidates = list(sq.enumerate_candidates(
    m_max=2,                              # |m| ≤ 2 for a tractable first pass
    coeffs=(1, -1, 1j, -1j),
    p_powers=(0, 1),
    max_leg_factors=2,
    ladder_spins=(-2, -1, 0, 1, 2),
    spectra_X={"CT": sq.CT},              # TT input
    spectra_Y={"CT": sq.CT},
    restrict_m_L=(-1, 1),                 # lensing-like scalar/vector output
))
t1 = time.time()
print(f"  enumerated {len(candidates)} distinct candidates "
      f"(dedup+compile) in {t1-t0:.1f}s")
print()

print(f"  scoring at lmax={LMAX}, L ∈ [{L_LO}, {L_HI}], L_ref={L_REF}...")
t2 = time.time()
results = sq.score_candidates(
    candidates,
    lmax=LMAX, Delta=2,
    hCxx=sq.hCT, hCyy=sq.hCT,
    user_funcs=[sq.hCT, sq.CT],
    spectra_args=(ocltt, cltt_theory),
    L_range=range(L_LO, L_HI + 1),
    L_ref=L_REF,
    px=px,
    progress=True,
)
t3 = time.time()
print(f"  scored {len(results)} candidates in {t3-t2:.1f}s "
      f"({(t3-t2)/max(len(results),1):.2f}s per candidate)")
print()


# --- report top 20 ----------------------------------------------------
def summarize_m(meta):
    mL, mX, mY = meta["m_triple"]
    return f"m=({mL:+d},{mX:+d},{mY:+d})"


def summarize_coeff(meta):
    c = meta["coeff"]
    return {1: " +1", -1: " -1", 1j: "+1j", -1j: "-1j"}.get(c, f"{c!r}")


def summarize_filter(f):
    s = str(f)
    if len(s) > 38:
        s = s[:35] + "..."
    return s


N_SHOW = 20
print(f"Top {N_SHOW} by gauge-fixed Fisher FOM = A_L({L_REF})·Σ(2L+1)/N_L^phi:")
print(f"  {'rank':>4s}  {'FOM':>10s}  {'coeff':>4s}  {'m':>10s}  P  "
      f"{'X_filter(l1)':>38s}  {'Y_filter(l2)':>38s}")
print("-" * 130)
for i, (fom, meta, expr) in enumerate(results[:N_SHOW]):
    if not np.isfinite(fom):
        continue
    print(f"  {i+1:>4d}  {fom:>10.3e}  {summarize_coeff(meta):>4s}  "
          f"{summarize_m(meta):>10s}  {meta['P_power']}  "
          f"{summarize_filter(meta['X_filter']):>38s}  "
          f"{summarize_filter(meta['Y_filter']):>38s}")

n_valid = sum(1 for r in results if np.isfinite(r[0]))
print()
print(f"{n_valid} / {len(results)} candidates produced finite FOM.")
print()
print("Next: expand grammar to include |m|≤3, symmetrize_l1l2=True, and")
print("      add the oracle cross-correlation cut to distinguish candidates")
print("      that measure phi from those that don't.")
