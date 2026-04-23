"""Probe whether the int-truncation gotcha in falafel's _gradient_spin
explains the residual per-m scatter on EE/BB/TB.

Hypothesis: hand-coded path applies fl = int(sqrt((l-|s|)(l+|s|+1)))
to the ladder leg.  Symbolic emitter applies fl as true float.
Ratio float_fl/int_fl differs per-l, creating per-m scatter.

Test: monkey-patch compile_native to int-truncate the X_filter on the
ladder leg AT THE LADDER FACTOR ONLY.  Compare residual against
the unpatched run.
"""
import numpy as np
import healpy as hp

from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, hCE, hCB, f_EE, f_BB, f_TB, f_EB, f_TE
from symqe.engine.estimator import compile_estimator
from symqe.engine.estimator_backend import strip_gamma
from symqe.engine.estimator_native import (
    Pixelization, compile_native, scalar_pair, pol_E_pair, pol_B_pair,
    compile_ee, compile_bb, compile_tb,
    compile_eb, compile_te,
)

LMAX, NSIDE = 200, 256
ell = np.arange(LMAX + 1, dtype=float)
cltt = 5e3 / (ell + 10) ** 2 + 1e-2
clee = 40.0 / (ell + 10) ** 2 + 1e-3
clbb = 5.0 / (ell + 10) ** 2 + 1e-4
ocltt = cltt + 1; oclee = clee + 1.5; oclbb = clbb + 1.5
spec = {"hCT": ocltt, "hCE": oclee, "hCB": oclbb,
        "CT": cltt, "CE": clee, "CB": clbb,
        "CTE": 0.5 * np.sqrt(cltt * clee)}

np.random.seed(42)
T = hp.synalm(ocltt, lmax=LMAX, new=True)
E = hp.synalm(oclee, lmax=LMAX, new=True)
B = hp.synalm(oclbb, lmax=LMAX, new=True)
px = Pixelization(nside=NSIDE)


def ladder_factor_float(spin_in, lmax):
    """The exact ladder factor sqrt((l-|s|)(l+|s|+1)) for spin±2 → spin±3
    (or spin±1) — the abs-spin transition at this leg."""
    ells = np.arange(0, lmax)  # length = lmax (matches falafel mlmax convention)
    s = abs(spin_in)
    fl = np.zeros_like(ells, dtype=float)
    if s == 0:
        # spin 0 → 1 (temperature ladder): sqrt(l(l+1))
        fl[:] = np.sqrt(ells * (ells + 1.0))
    elif s == 2:
        # spin +2 → +3 (raising): sqrt((l-2)(l+3))
        # spin -2 → -1 (lowering): sqrt((l-1)(l+2))
        # NOTE: which one fires depends on |s_out|, but for now assume +
        fl[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
    return fl


def ladder_factor_int(spin_in, lmax):
    """Falafel's int-truncated version: fl = ells * 0 (int!) then float
    assignments are silently truncated to int."""
    ells = np.arange(0, lmax)
    s = abs(spin_in)
    if s == 0:
        # T case: fl = sqrt(l(l+1)) is computed as float (no truncation), so this matches float
        return np.sqrt(ells * (ells + 1.0))
    elif s == 2:
        fl = ells * 0  # INTEGER zeros
        # Try +-spin out:
        # spin=+2 → spin=+3 raising: sqrt((l-2)(l+3))
        # spin=-2 → spin=-1 lowering: sqrt((l-1)(l+2))
        fl[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
        return fl.astype(float)


def show_int_vs_float():
    """Print the actual float vs int ladder values at low l for spin=2 raising."""
    lmax = 20
    ells = np.arange(0, lmax)
    fl_f = np.zeros_like(ells, dtype=float)
    fl_f[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
    fl_i = ells * 0
    fl_i[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
    print(f"\n=== float vs int ladder factor (spin +2 → +3): sqrt((l-2)(l+3)) ===")
    print(f"{'l':>4s} {'float':>12s} {'int':>6s} {'ratio f/i':>12s}")
    for L in range(2, lmax):
        ratio = fl_f[L] / fl_i[L] if fl_i[L] != 0 else float('nan')
        print(f"{L:>4d} {fl_f[L]:>12.6f} {fl_i[L]:>6d} {ratio:>12.6f}")

    # Same for lowering: spin -2 → -1, sqrt((l-1)(l+2))
    fl_f2 = np.zeros_like(ells, dtype=float)
    fl_f2[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
    fl_i2 = ells * 0
    fl_i2[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
    print(f"\n=== float vs int ladder factor (spin -2 → -1): sqrt((l-1)(l+2)) ===")
    print(f"{'l':>4s} {'float':>12s} {'int':>6s} {'ratio f/i':>12s}")
    for L in range(1, lmax):
        ratio = fl_f2[L] / fl_i2[L] if fl_i2[L] != 0 else float('nan')
        print(f"{L:>4d} {fl_f2[L]:>12.6f} {fl_i2[L]:>6d} {ratio:>12.6f}")


show_int_vs_float()


# ---- Now monkey-patch the emitter to int-truncate the ladder portion ----
# Strategy: in compile_native, after computing x_fl on the ladder leg,
# divide out the float ladder factor and multiply in the int one.
# We need to know what ladder factor is in x_fl: depends on |s_X|.

import symqe.engine.estimator_native as en

_orig_compile_native = en.compile_native


def patched_compile_native(terms, lmax, *, px=None, nside=None,
                            shape=None, wcs=None, _truncate=True):
    base = _orig_compile_native(terms, lmax, px=px, nside=nside, shape=shape, wcs=wcs)
    if not _truncate:
        return base

    fused = base.fused_plans

    def emit(X_input, Y_input, spectra):
        from symqe.engine.estimator_backend import _eval_atom
        import healpy as hp
        from symqe.engine.estimator_native import (
            _alm_to_signed_pair, _resolve_px,
        )

        X_pair, X_spin_alm = X_input
        Y_pair, Y_spin_alm = Y_input
        X_pair = np.asarray(X_pair, dtype=np.complex128)
        Y_pair = np.asarray(Y_pair, dtype=np.complex128)
        ell = np.arange(lmax + 1, dtype=float)

        outputs = {}
        pxr = _resolve_px(px, nside, shape, wcs)

        for plan in fused:
            x_fl = _eval_atom(plan.X_filter, ell, spectra).real.astype(np.float64)
            y_fl = _eval_atom(plan.Y_filter, ell, spectra).real.astype(np.float64)
            L_fl = _eval_atom(plan.L_factor, ell, spectra).real.astype(np.float64)

            ladder_X = (plan.abs_spin_X != X_spin_alm)
            ladder_Y = (plan.abs_spin_Y != Y_spin_alm)
            if plan.abs_spin_L > 0:
                if ladder_X and plan.abs_spin_X > 0:
                    x_fl = x_fl.copy(); x_fl[lmax] = 0.0
                    x_fl[0] = 0.0; x_fl[1] = 0.0
                if ladder_Y and plan.abs_spin_Y > 0:
                    y_fl = y_fl.copy(); y_fl[lmax] = 0.0
                    y_fl[0] = 0.0; y_fl[1] = 0.0
                L_fl = L_fl.copy(); L_fl[lmax] = 0.0

                # ---------- INT TRUNCATION INJECTION ----------
                # On a ladder leg, the symbolic filter contains
                # the Namikawa a-factor a(l, s) = -sqrt((l-s)(l+s+1)/2).
                # Falafel applies sqrt((l-s)(l+s+1)) (int-truncated, no /sqrt(2)).
                # We've already verified the 1/sqrt(2) is absorbed in pair_coeff.
                # So the per-l float vs int discrepancy is in sqrt((l-s)(l+s+1)).
                #
                # Compute the int-truncated version of the ladder factor and
                # replace the float version in x_fl / y_fl by its ratio.

                def int_trunc_ladder(abs_spin_out, lmax):
                    """Build the float and int versions of the ladder factor
                    that hand-coded falafel applies.  Dispatch purely on
                    abs_spin_out (= 1 for sqrt((l-1)(l+2)) lowering ladder,
                    = 3 for sqrt((l-2)(l+3)) raising ladder).  The spin-0
                    temperature ladder sqrt(l(l+1)) is float in falafel too
                    (computed with ells*(ells+1.0)), so no int truncation."""
                    ells = np.arange(0, lmax)
                    if abs_spin_out == 3:
                        ff = np.zeros_like(ells, dtype=float)
                        ff[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
                        fi = ells * 0
                        fi[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
                        ff_full = np.zeros(lmax + 1); ff_full[:lmax] = ff
                        fi_full = np.zeros(lmax + 1); fi_full[:lmax] = fi.astype(float)
                        return ff_full, fi_full
                    elif abs_spin_out == 1:
                        # sqrt((l-1)(l+2)) — spin -2 → -1 lowering, int-truncated
                        # NOTE: for abs_spin_out=1 from spin-0 T, the ladder is
                        # sqrt(l(l+1)) which IS already float in falafel.
                        # But for abs_spin_out=1 arriving from a |s_alm|=2 context
                        # (TB's X leg), the W symbolic factor sqrt((l-1)(l+2))
                        # matches the a_minus = -sqrt((l-1)(l+2)/2) in W_lens_m
                        # which corresponds to the spin=-2 → spin=-1 ladder.
                        # We distinguish by the FILTER shape, not by spin_alm.
                        ff = np.zeros_like(ells, dtype=float)
                        ff[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
                        fi = ells * 0
                        fi[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
                        ff_full = np.zeros(lmax + 1); ff_full[:lmax] = ff
                        fi_full = np.zeros(lmax + 1); fi_full[:lmax] = fi.astype(float)
                        return ff_full, fi_full
                    return None, None

                # Detect whether the X_filter / Y_filter actually contains the
                # int-truncatable ladder factor.  For abs_spin=1 from a spin-0
                # input (T-only, temperature lensing), the ladder is sqrt(l(l+1))
                # which is float in both paths — no truncation needed.
                # For abs_spin=1 from a |s_alm|=2 input (TB X-leg, EB X-leg),
                # the filter carries sqrt((l-1)(l+2)), which IS int-truncated.
                # For abs_spin=3, always sqrt((l-2)(l+3)), int-truncated.
                def ladder_is_pol(spin_alm_in, abs_spin_out):
                    if abs_spin_out == 3:
                        return True
                    if abs_spin_out == 1 and abs(spin_alm_in) == 2:
                        return True
                    return False

                # Detect int-truncatable ladder by inspecting the plan's
                # AtomicFactor filter structure: if it contains sqrt(l-2)
                # and sqrt(l+3) as factors, the raising ladder is present;
                # if it contains sqrt(l-1) and sqrt(l+2) it's the lowering
                # ladder.  The spin-0 temperature ladder sqrt(l(l+1)) is
                # float-computed in falafel (no truncation), so we skip it.
                def filter_has_ladder_factors(atom):
                    """Return 'raising' | 'lowering' | None based on which
                    int-truncatable ladder factors the AtomicFactor contains."""
                    from symqe.engine.atom import Mul, Pow, Add, Const, Var, FuncApp
                    import sympy as sp
                    if not isinstance(atom, Mul):
                        return None
                    factors = atom.factors if hasattr(atom, 'factors') else ()
                    has_lm1 = has_lm2 = has_lp2 = has_lp3 = False
                    for f in factors:
                        if isinstance(f, Pow) and f.exp == sp.Rational(1, 2):
                            b = f.base
                            if isinstance(b, Add):
                                terms_ = b.terms if hasattr(b, 'terms') else ()
                                # Expect (const, var_name)
                                const_vals = [t.value for t in terms_ if isinstance(t, Const)]
                                if sp.Integer(-1) in const_vals: has_lm1 = True
                                if sp.Integer(-2) in const_vals: has_lm2 = True
                                if sp.Integer(2)  in const_vals: has_lp2 = True
                                if sp.Integer(3)  in const_vals: has_lp3 = True
                    if has_lm2 and has_lp3:
                        return 'raising'  # sqrt((l-2)(l+3))
                    if has_lm1 and has_lp2:
                        return 'lowering'  # sqrt((l-1)(l+2))
                    return None

                def apply_int_trunc(fl_arr, kind, lmax):
                    ells = np.arange(0, lmax)
                    if kind == 'raising':
                        ff = np.zeros_like(ells, dtype=float)
                        ff[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
                        fi = ells * 0
                        fi[ells >= 2] = np.sqrt((ells[ells >= 2] - 2) * (ells[ells >= 2] + 3.0))
                    elif kind == 'lowering':
                        ff = np.zeros_like(ells, dtype=float)
                        ff[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
                        fi = ells * 0
                        fi[ells >= 1] = np.sqrt((ells[ells >= 1] - 1) * (ells[ells >= 1] + 2.0))
                    else:
                        return fl_arr
                    ff_full = np.zeros(lmax + 1); ff_full[:lmax] = ff
                    fi_full = np.zeros(lmax + 1); fi_full[:lmax] = fi.astype(float)
                    ratio = np.where(ff_full != 0, fi_full / ff_full, 1.0)
                    return fl_arr * ratio

                if ladder_X and plan.abs_spin_X > 0:
                    kind = filter_has_ladder_factors(plan.X_filter)
                    if kind:
                        x_fl = apply_int_trunc(x_fl, kind, lmax)

                if ladder_Y and plan.abs_spin_Y > 0:
                    kind = filter_has_ladder_factors(plan.Y_filter)
                    if kind:
                        y_fl = apply_int_trunc(y_fl, kind, lmax)

            Xf = np.stack([hp.almxfl(X_pair[0], x_fl),
                           hp.almxfl(X_pair[1], x_fl)])
            Yf = np.stack([hp.almxfl(Y_pair[0], y_fl),
                           hp.almxfl(Y_pair[1], y_fl)])

            X_maps = _alm_to_signed_pair(pxr, Xf, X_spin_alm,
                                          plan.abs_spin_X, lmax)
            Y_maps = _alm_to_signed_pair(pxr, Yf, Y_spin_alm,
                                          plan.abs_spin_Y, lmax)

            flip_X = (plan.abs_spin_L > 0) and (plan.abs_spin_X > 0)
            flip_Y = (plan.abs_spin_L > 0) and (plan.abs_spin_Y > 0)
            is_W_plus = all(abs(complex(c).imag) < 1e-12 * (abs(complex(c).real) + 1e-30)
                            for c in plan.coeffs.values())
            plan_factor = 1 + 0j
            if plan.abs_spin_L > 0:
                if is_W_plus:
                    if plan.abs_spin_X == 3:
                        plan_factor = -plan_factor
                    if plan.abs_spin_Y == 3:
                        plan_factor = -plan_factor
                else:
                    plan_factor = -1j * plan_factor
                if (is_W_plus and X_spin_alm == 0 and plan.abs_spin_X > 0
                        and abs(Y_spin_alm) == 2 and plan.abs_spin_Y > 0):
                    plan_factor = -plan_factor
                if (is_W_plus and abs(X_spin_alm) == 2
                        and abs(Y_spin_alm) == 2
                        and plan.abs_spin_X > 0 and plan.abs_spin_Y > 0):
                    plan_factor = -plan_factor

            for (sX, sY, sL), coeff in plan.coeffs.items():
                sX_eff = -sX if flip_X else sX
                sY_eff = -sY if flip_Y else sY
                Mx = X_maps[0] if sX_eff >= 0 else X_maps[1]
                My = Y_maps[0] if sY_eff >= 0 else Y_maps[1]
                prod = plan_factor * complex(coeff) * Mx * My

                if plan.abs_spin_L == 0:
                    m = prod.real
                    alm = pxr.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)
                    alm = hp.almxfl(alm, L_fl)
                    outputs[0] = alm if 0 not in outputs else outputs[0] + alm
                else:
                    alm_pair = pxr.map2alm_spin(prod, lmax=lmax,
                                                spin_alm=0,
                                                spin_transform=plan.abs_spin_L)
                    pick = alm_pair[0] if sL >= 0 else alm_pair[1]
                    pick = hp.almxfl(pick, L_fl)
                    outputs[sL] = pick if sL not in outputs else outputs[sL] + pick

        return outputs

    return emit


def per_L(label, gen, ref):
    rc = hp.alm2cl(ref); gc = hp.alm2cl(gen); xc = hp.alm2cl(gen, ref)
    Ls = [5, 10, 20, 50, 100, 150, 199]
    print(f"\n-- {label} --")
    print(f"{'L':>4s} {'gen/ref':>12s} {'signed_slope':>14s}")
    for L in Ls:
        if rc[L] > 0 and gc[L] > 0:
            r = np.sqrt(gc[L] / rc[L])
            s = xc[L] / rc[L]
            print(f"{L:>4d} {r:>12.5e} {s:>14.5e}")


def stdmean(label, gen, ref):
    """Slope-and-scatter summary: median ratio + std/mean of per-L ratio."""
    gc = hp.alm2cl(gen); rc = hp.alm2cl(ref); xc = hp.alm2cl(gen, ref)
    valid = (gc > 0) & (rc > 0)
    ratios = xc[valid] / rc[valid]  # signed
    med = np.median(ratios)
    sm = np.std(ratios) / np.abs(np.mean(ratios)) if np.abs(np.mean(ratios)) > 0 else float('nan')
    print(f"  {label}: median signed_slope = {med:.4e}, std/mean = {sm:.4e}")


def run_one(name, f_fn, hcfn_args, in_x, in_y, hc_native_fn):
    g = f_fn[0]
    terms = strip_gamma(compile_estimator(g))
    # Stock symbolic
    stock = np.asarray(_orig_compile_native(terms, LMAX, px=px)(in_x, in_y, spec)[+1])
    # Patched (int-truncated ladder factor)
    patched = np.asarray(patched_compile_native(terms, LMAX, px=px)(in_x, in_y, spec)[+1])
    ref = np.asarray(hc_native_fn(terms, LMAX, px=px)(*hcfn_args))
    print(f"\n========== {name} ==========")
    stdmean(f"stock symbolic vs ref", stock, ref)
    stdmean(f"int-trunc patched vs ref", patched, ref)
    per_L("stock", stock, ref)
    per_L("int-trunc", patched, ref)


run_one("EE", (f_EE() / (hCE(l1) * hCE(l2)),),
        (E, spec), pol_E_pair(E), pol_E_pair(E), compile_ee)
run_one("BB", (f_BB() / (hCB(l1) * hCB(l2)),),
        (B, spec), pol_B_pair(B), pol_B_pair(B), compile_bb)
run_one("TB", (f_TB() / (hCT(l1) * hCB(l2)),),
        (T, B, spec), scalar_pair(T), pol_B_pair(B), compile_tb)
run_one("TE", (f_TE() / (hCT(l1) * hCE(l2)),),
        (T, E, spec), scalar_pair(T), pol_E_pair(E), compile_te)
run_one("EB", (f_EB() / (hCE(l1) * hCB(l2)),),
        (E, B, spec), pol_E_pair(E), pol_B_pair(B), compile_eb)
