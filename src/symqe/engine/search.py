"""Generic multi-case search harness.

Ties the enumerator, Fisher scoring, and oracle rerank into a single
reproducible entry point.  A ``SearchCase`` bundles the
case-specific pieces (grammar, simulation generator, compile kwargs,
L ranges, reference candidates) so that ``run_search`` can execute
the full pipeline uniformly across estimator types (lensing TT,
rotation EB, source TT, …).

Design
------
- Enumeration is deterministic and case-local: one pass.
- Fisher scoring depends only on the model spectra, not on the random
  seed — so it also runs once per case.
- Oracle cross-correlation depends on the simulation realization; it
  is the only piece that loops over seeds.
- Aggregation combines per-seed oracle scores into (mean, std) per
  candidate, plus a stable ranking based on the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from .enumerate import enumerate_candidates, score_candidates
from .scoring import oracle_rerank


@dataclass
class SearchCase:
    """Declarative description of one QE search case.

    Attributes
    ----------
    name : str
        Human-readable label (used in reports).
    lmax : int
        Max multipole for both estimator and A_L.
    grammar : dict
        kwargs passed to ``enumerate_candidates``.
    simulate : callable ``(seed: int) -> (X_input, Y_input, oracle_alm)``
        Produces input alms (pre-packaged via scalar_pair / pol_E_pair /
        …) and the oracle reference field for a given seed.
    compile_kwargs : dict
        kwargs for ``compile_qe_from_f`` via ``score_candidates``
        (``Delta``, ``hCxx``, ``hCyy``, ``user_funcs``, and one of
        ``px`` / ``nside`` / ``shape+wcs``).
    spectra_for_fisher : tuple of arrays
        Positional spectrum tuple for A_L_fn / N_L_phi_fn (same order
        as ``user_funcs`` in compile_kwargs).
    spectra_for_estimator : dict[str, array]
        Runtime spectrum dict for the estimator callable (the ``spec``
        dict ``compile_native`` expects).
    fisher_L_range : range or tuple (L_lo, L_hi)
        Modes for the Fisher sum.
    fisher_L_ref : int
        Reference L for the gauge-fix multiplier.
    oracle_L_range : range or tuple
        Modes for the cross-correlation sum.
    output_key : int
        Channel key to extract from the estimator's output dict (+1
        for lensing phi, 0 for rotation/source scalar).
    top_k_for_oracle : int
        How many Fisher-ranked candidates survive to the oracle pass.
    references : list of (label, sympy.Expr)
        Hand-coded weights injected into the candidate pool for
        ranking comparison.
    """
    name: str
    lmax: int
    grammar: dict
    simulate: Callable[[int], tuple]
    compile_kwargs: dict
    spectra_for_fisher: tuple
    spectra_for_estimator: dict
    fisher_L_range: Any
    fisher_L_ref: int
    oracle_L_range: Any
    output_key: int = +1
    top_k_for_oracle: int = 1000
    references: list = field(default_factory=list)


def _as_range(spec):
    if isinstance(spec, range):
        return spec
    lo, hi = spec
    return range(lo, hi + 1)


def run_search(case: SearchCase, *, seeds=(0,), verbose: bool = True):
    """Run the full enumerate → Fisher → oracle pipeline for one case.

    Returns
    -------
    dict with keys:
      'candidates'    : the list of (meta, expr) after enumeration + refs
      'fisher'        : Fisher pass result, list of (fom, meta, expr, bundle)
      'oracle_by_seed': dict{seed → list of (score, fisher_fom, meta, expr)}
      'summary'       : list of (rank, mean, std, count, meta, expr)
                        sorted by mean oracle score descending, aggregated
                        across seeds.
      'case'          : the input SearchCase (for plotting / reporting)
    """
    import time

    # --- 1. Enumerate candidates (+ inject references) ---
    if verbose: print(f"[{case.name}] enumerating...", flush=True)
    t0 = time.time()
    candidates = list(enumerate_candidates(**case.grammar))
    for label, expr in case.references:
        candidates.append(({"label": label, "flavor": "reference"}, expr))
    if verbose:
        print(f"[{case.name}]   {len(candidates)} candidates "
              f"({time.time()-t0:.1f}s)", flush=True)

    # --- 2. Fisher pass (once per case; spectra are seed-independent) ---
    if verbose: print(f"[{case.name}] Fisher pass...", flush=True)
    t0 = time.time()
    fisher_results = score_candidates(
        candidates,
        lmax=case.lmax,
        **case.compile_kwargs,
        spectra_args=case.spectra_for_fisher,
        L_range=_as_range(case.fisher_L_range),
        L_ref=case.fisher_L_ref,
        return_bundles=True,
        progress=False,
    )
    if verbose:
        print(f"[{case.name}]   ranked {len(fisher_results)} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # --- 3. Keep Fisher top-K ∪ references ---
    fisher_top = list(fisher_results[:case.top_k_for_oracle])
    top_ids = {id(r[2]) for r in fisher_top}
    dropped_refs = 0
    for fom, meta, expr, bundle in fisher_results:
        if (isinstance(meta, dict)
            and meta.get("flavor") == "reference"
            and id(expr) not in top_ids):
            fisher_top.append((fom, meta, expr, bundle))
            dropped_refs += 1
    if verbose and dropped_refs:
        print(f"[{case.name}]   kept {dropped_refs} refs that fell outside "
              f"top-{case.top_k_for_oracle}", flush=True)

    # --- 4. Oracle rerank per seed ---
    oracle_by_seed: dict = {}
    for seed in seeds:
        if verbose: print(f"[{case.name}] oracle seed={seed}...", flush=True)
        t0 = time.time()
        X_input, Y_input, oracle_alm = case.simulate(seed)
        res = oracle_rerank(
            fisher_top,
            X_input=X_input, Y_input=Y_input,
            spectra=case.spectra_for_estimator,
            phi_true_alm=oracle_alm,
            spec_for_A_L=case.spectra_for_fisher,
            L_range=_as_range(case.oracle_L_range),
            output_key=case.output_key,
            progress=False,
        )
        oracle_by_seed[seed] = res
        if verbose:
            print(f"[{case.name}]   seed={seed} ranked {len(res)} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # --- 5. Aggregate across seeds ---
    summary = _aggregate_across_seeds(oracle_by_seed)

    return {
        "case": case,
        "candidates": candidates,
        "fisher": fisher_results,
        "oracle_by_seed": oracle_by_seed,
        "summary": summary,
    }


def _aggregate_across_seeds(oracle_by_seed):
    """Combine per-seed oracle rankings into mean ± std per candidate.

    Candidate identity across seeds is the ``id(expr)`` — same expr
    instance was looped through every seed via the same fisher_top list.
    """
    if not oracle_by_seed:
        return []
    seeds = list(oracle_by_seed.keys())
    # Build score matrix keyed by id(expr).
    per_expr: dict = {}
    for seed in seeds:
        for score, fisher_fom_val, meta, expr in oracle_by_seed[seed]:
            eid = id(expr)
            if eid not in per_expr:
                per_expr[eid] = {
                    "scores": [],
                    "fisher": fisher_fom_val,
                    "meta": meta,
                    "expr": expr,
                }
            per_expr[eid]["scores"].append(score)

    rows = []
    for eid, entry in per_expr.items():
        scores = np.asarray(entry["scores"], dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            mean, std = -np.inf, np.inf
        else:
            mean = float(np.mean(finite))
            std  = float(np.std(finite, ddof=(1 if finite.size > 1 else 0)))
        rows.append({
            "mean": mean, "std": std, "count": int(len(scores)),
            "fisher": entry["fisher"],
            "meta": entry["meta"],
            "expr": entry["expr"],
        })
    rows.sort(key=lambda r: r["mean"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def print_summary(result, n_show: int = 20):
    """Pretty-print the top-N candidates and the reference ranks."""
    case = result["case"]
    summary = result["summary"]
    print()
    print(f"=== [{case.name}] Top {n_show} by mean oracle ⟨r(L)⟩ "
          f"(over {len(result['oracle_by_seed'])} seed(s)) ===")
    print(f"{'rk':>3s}  {'mean <r>':>10s}  {'std':>8s}  "
          f"{'fisher':>10s}  description")
    print("-" * 110)
    for row in summary[:n_show]:
        desc = _describe_meta(row["meta"])
        print(f"{row['rank']:>3d}  {row['mean']:>+10.4f}  {row['std']:>8.4f}  "
              f"{row['fisher']:>10.3e}  {desc}")
    refs = [r for r in summary if isinstance(r["meta"], dict)
            and r["meta"].get("flavor") == "reference"]
    if refs:
        print()
        print(f"=== [{case.name}] Reference candidate ranks ===")
        for row in refs:
            lbl = row["meta"]["label"]
            print(f"  {lbl:28s}  rank {row['rank']:>4d}/{len(summary)}   "
                  f"<r>={row['mean']:+.4f} ± {row['std']:.4f}   "
                  f"fisher={row['fisher']:.3e}")
    print()


def _describe_meta(meta):
    if isinstance(meta, dict) and meta.get("flavor") == "reference":
        return f"REF: {meta['label']}"
    if isinstance(meta, dict):
        mL, mX, mY = meta["m_triple"]
        Xs = str(meta.get("X_filter", "?"))[:24]
        Ys = str(meta.get("Y_filter", "?"))[:24]
        flav = meta.get("flavor", "?")
        return f"m=({mL:+d},{mX:+d},{mY:+d}) {flav:3s} X={Xs} Y={Ys}"
    return "?"
