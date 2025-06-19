import norm_lens
from jax import jit
from jax import grad
from falafel import utils as futils
import jax.numpy as jnp
from jax import jacobian


mlmax = 4000
lmin = 600
lmax = 3000
combos = ["TT","TE","EE","BB"]
ucls, tcls = futils.get_theory_dicts(nells = None, lmax = mlmax, grad = False)

qtt_jit = jit(norm_lens.qtt,static_argnums=(0,1,2))
qtt_deriv = jacobian(qtt_jit,allow_int=True)(mlmax,lmin,lmax,ucls,tcls)
print("Done jac qtt")

qte_jit = jit(norm_lens.qte,static_argnums=(0,1,2))
qte_deriv = jacobian(qte_jit,allow_int=True)(mlmax,lmin,lmax,ucls,tcls)
print("Done jac qte")

qee_jit = jit(norm_lens.qte,static_argnums=(0,1,2))
qee_deriv = jacobian(qee_jit,allow_int=True)(mlmax,lmin,lmax,ucls,tcls)
print("Done jac qee")

qbb_jit = jit(norm_lens.qbb,static_argnums=(0,1,2))
qbb_deriv = jacobian(qbb_jit,allow_int=True)(mlmax,lmin,lmax,ucls,tcls)
print("Done jac qee")

qeb_jit = jit(norm_lens.qeb,static_argnums=(0,1,2))
qeb_deriv = jacobian(qeb_jit,allow_int=True)(mlmax,lmin,lmax,ucls,tcls)
print("Done jac qeb")