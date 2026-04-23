"""TT-only estimator recipe dump — the focused version of test_estimator_all.py."""
from sympy import sympify
from symqe.engine.l12_sum import l1, l2
from symqe.engine.namikawa import hCT, f_TT
from symqe.engine.estimator import compile_estimator, pretty, pretty_grouped

g_TT = f_TT(px=+1) / (sympify(2) * hCT(l1) * hCT(l2))

terms = compile_estimator(g_TT)
print("=" * 76)
print("Per-term SHT recipes for the lensing TT estimator")
print("=" * 76)
print(pretty(terms, x_name="T_l1m", y_name="T_l2m"))

print("=" * 76)
print("Grouped by unique SHT plan (collapsing equivalent recipes)")
print("=" * 76)
print(pretty_grouped(terms, x_name="T_l1m", y_name="T_l2m"))
