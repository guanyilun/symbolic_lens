"""symqe — symbolic compiler for CMB quadratic estimators.

Typical use::

    import symqe as sq

    # Build a weight from the Namikawa vocabulary (f_TT / f_EE / f_EB / f_rot_EB / ...)
    g = sq.f_TT()

    # Build the estimator — one line, takes raw alms.
    estimator = sq.build_estimator(g, lmax=LMAX, nside=NSIDE, X='T', Y='T')
    phi_alm   = estimator(T_alm, T_alm, spectra)

    # Two-step form if you want to inspect the SHT recipe first:
    recipe    = sq.compile_estimator(g)
    print(sq.pretty(recipe))
    estimator = sq.build_estimator(recipe, lmax=LMAX, nside=NSIDE, X='T', Y='T')

    # Bundled: estimator + Hu-Okamoto A_L + N_L^phi from one symbolic g.
    estimator_fn, A_L_fn, N_L_fn = sq.compile_qe(g, lmax=LMAX, rlmin=100, rlmax=3000)

    # Just the normalization (A_L) pipeline.
    norm_callable = sq.compile_norm(g, lmax=LMAX, rlmin=100, rlmax=3000)

The `symqe.engine` and `symqe.reference` subpackages remain importable for
advanced or internal use.
"""

# -- DSL grammar --------------------------------------------------------
from .engine.l12_sum import l, l1, l2, P, wigner_3j

# -- Domain vocabulary: spectra and ladder helpers ----------------------
from .engine.namikawa import (
    hCT, hCE, hCB, hCTE,
    CT, CE, CB, CTE,
    a, a_plus, a_minus,
    gamma_f, q_plus, q_minus,
    ZETA_PLUS, ZETA_MINUS,
)

# -- Namikawa weight building blocks ------------------------------------
from .engine.namikawa import (
    W_lens_0, W_lens_p, W_lens_m,
    W_rot_p, W_rot_m,
    W_ampl_0,
)

# -- High-level weight builders (one per estimator) ---------------------
from .engine.namikawa import (
    f_TT, f_TE, f_TB, f_EE, f_EB, f_BB,
    f_rot_EB, f_rot_EE, f_rot_TB,
    f_ampl_TT,
)

# -- Compile entry points ----------------------------------------------
from .engine.estimator import compile_estimator, EstimatorTerm, pretty, pretty_grouped
from .engine.l12_sum import compile_norm, NormCompiler
from .engine.compile_qe import compile_qe, compile_qe_from_f
from .engine.scoring import fisher_fom, cross_correlation_score
from .engine.enumerate import (
    enumerate_candidates, score_candidates,
    m_triples, leg_filters, canonical_key,
)
from .engine.estimator_native import (
    build_estimator,
    compile_native,        # low-level: raw (alm_pair, spin_alm) interface
    # Per-field convenience wrappers (bit-for-bit with falafel/pytempura).
    compile_tt, compile_ee, compile_bb, compile_tb, compile_eb, compile_te,
    compile_rot_eb, compile_source_tt,
    Pixelization,
    scalar_pair, pol_E_pair, pol_B_pair, pol_EB_pair,
)

# -- Lower-level gamma-strip (normally auto-called; exposed for inspection) --
from .engine.estimator_backend import strip_gamma

__all__ = [
    # grammar
    "l", "l1", "l2", "P", "wigner_3j",
    # spectra
    "hCT", "hCE", "hCB", "hCTE", "CT", "CE", "CB", "CTE",
    # ladder / gamma helpers
    "a", "a_plus", "a_minus", "gamma_f", "q_plus", "q_minus",
    "ZETA_PLUS", "ZETA_MINUS",
    # W weights
    "W_lens_0", "W_lens_p", "W_lens_m",
    "W_rot_p", "W_rot_m", "W_ampl_0",
    # f builders
    "f_TT", "f_TE", "f_TB", "f_EE", "f_EB", "f_BB",
    "f_rot_EB", "f_rot_EE", "f_rot_TB", "f_ampl_TT",
    # compile (high-level + low-level)
    "compile_estimator", "compile_norm", "compile_qe", "compile_qe_from_f",
    "build_estimator", "compile_native",
    # scoring / search
    "fisher_fom", "cross_correlation_score",
    "enumerate_candidates", "score_candidates",
    "m_triples", "leg_filters", "canonical_key",
    # per-field convenience (bit-for-bit with falafel/pytempura)
    "compile_tt", "compile_ee", "compile_bb",
    "compile_tb", "compile_eb", "compile_te",
    "compile_rot_eb", "compile_source_tt",
    # types
    "EstimatorTerm", "Pixelization", "NormCompiler",
    # input packing
    "scalar_pair", "pol_E_pair", "pol_B_pair", "pol_EB_pair",
    # display / introspection
    "pretty", "pretty_grouped", "strip_gamma",
]
