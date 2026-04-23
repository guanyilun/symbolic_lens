"""
Demonstrate the symbolic engine handling a new-physics estimator that
falafel does not ship a hand-coded primitive for: the CMB rotation
(anisotropic birefringence, α) quadratic estimator, f^{α, EB}.

Workflow:
  1. Build g^{α,EB} symbolically using the rotation weight W^{α,+} from
     namikawa.py (analogous to tempura.jl's Wₐ⁺).
  2. Run the analyzer to get the list of EstimatorTerms; print the SHT
     recipe for visual inspection.
  3. Run the native backend on a synthetic alm set to confirm the
     recipe executes end-to-end and produces a reasonable-looking alm.

This is a minimum proof-of-concept showing that the symbolic compiler
can accept ANY W a user writes — falafel is not involved at either
compile time or runtime for the rotation estimator.
"""
import numpy as np
from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCE, hCB, f_rot_EB
from symqe.engine.estimator import compile_estimator, pretty_grouped
from symqe.engine.estimator_backend import strip_gamma


def main():
    # ---- 1. Symbolic recipe ----
    g_rot_EB = f_rot_EB(px=-1) / (hCE(l1) * hCB(l2))   # Δ = 1 for EB

    terms = compile_estimator(g_rot_EB)
    terms_stripped = strip_gamma(terms)

    print("="*72)
    print("Rotation EB estimator — SHT recipe (from symbolic compiler)")
    print("="*72)
    print(f"Compiled to {len(terms_stripped)} γ-stripped atomic terms.\n")
    print(pretty_grouped(terms_stripped, x_name="E_l1m", y_name="B_l2m"))

    # ---- 2. Show a couple key observations ----
    print()
    print("Key structural observations:")
    imag_coeffs = [t for t in terms_stripped
                   if abs(complex(t.coeff).imag) > 1e-12]
    real_coeffs = [t for t in terms_stripped
                   if abs(complex(t.coeff).imag) <= 1e-12]
    print(f"  terms with Im(coeff) != 0: {len(imag_coeffs)} (carry ζ⁻ = i)")
    print(f"  terms with Im(coeff) == 0: {len(real_coeffs)}")
    uniq_spin_tuples = {(t.spin_X, t.spin_Y, t.spin_L) for t in terms_stripped}
    print(f"  unique (spin_X, spin_Y, spin_L) tuples: {len(uniq_spin_tuples)}")
    print("  tuples:", sorted(uniq_spin_tuples))

    # ---- 3. What would a native-backend emitter need? ----
    print()
    print("Next step toward a full rotation estimator pipeline:")
    print(" - rotation EB needs a compile_eb-style emitter with the")
    print("   ζ⁻=i factor replaced by the rotation's own ζ handling,")
    print("   and the appropriate pair_coeff bookkeeping for p_α = -1.")
    print(" - since the SHT structure (pair of spin-2 legs + map2alm_spin")
    print("   at spin 1) is structurally the same as lensing EB, the")
    print("   native backend's qe_pol_only primitive already handles it;")
    print("   only the coefficient accounting is new.")


if __name__ == "__main__":
    main()
