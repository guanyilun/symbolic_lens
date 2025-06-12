import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from functools import partial

@partial(jax.jit, inline=True)
def alpha_f(l: jax.Array, s1: int, s2: int) -> jax.Array:
    return jnp.where((l <= jnp.abs(s1)) | (l <= jnp.abs(s2)), 0.0,
                     jnp.sqrt((l**2 - s1**2)*(l**2 - s2**2))/l)

def wigd_init(s1: int, s2: int) -> tuple[int, int, int, float]:
    sign = 1.0
    if abs(s1) > abs(s2):
        s1, s2 = s2, s1
        sign *= (-1) ** ((s1 + s2) % 2)
    if s2 < 0:
        s1, s2 = -s1, -s2
        sign *= (-1) ** ((s1 + s2) % 2)
    l0 = max(abs(s1), abs(s2))
    return s1, s2, l0, sign

def _calculate_norm_factor_A(s1: int, s2: int) -> jax.Array:
    abs_s1 = abs(s1)
    upper = s2 - abs_s1
    i = jnp.arange(1, upper + 1)
    terms = (s2 + abs_s1 + i) / i
    return jnp.sqrt(jnp.prod(terms, initial=1.0)).astype(jnp.float64)

@partial(jax.jit, static_argnums=(0, 1, 4, 5, 6))
def cf_from_cl(s1: int, s2: int, cl: jax.Array, cos_theta: jax.Array, 
              lmax: int | None = None, lmin: int = 0, prefactor: bool = False) -> jax.Array:
    new_s1, new_s2, l0, sign = wigd_init(s1, s2)
    lmax = len(cl)-1 if lmax is None else lmax
    if lmax < l0:
        return jnp.zeros_like(cos_theta)
    
    norm_factor_A = _calculate_norm_factor_A(new_s1, new_s2) * sign
    
    term1 = jnp.maximum(0.0, (1 + cos_theta)/2) ** ((new_s2 + new_s1)/2)
    term2 = jnp.maximum(0.0, (1 - cos_theta)/2) ** ((new_s2 - new_s1)/2)
    wigd_hi = norm_factor_A * term1 * term2
    wigd_lo = jnp.zeros_like(cos_theta)
    
    fac = (2*l0+1)/(4*jnp.pi) if prefactor else 1.0
    cond_l0 = (l0 >= lmin) & (l0 <= lmax) & (l0 < len(cl))
    cf = jnp.where(cond_l0, cl[l0] * wigd_hi * fac, 0.0)
    
    def body_fn(carry, l):
        w_hi, w_lo, cf = carry
        l_float = l.astype(w_hi.dtype)
        alpha_hi = alpha_f(l_float + 1, new_s1, new_s2)
        alpha_lo = alpha_f(l_float, new_s1, new_s2)
        beta = jnp.where((new_s1 == 0) | (new_s2 == 0), 0.0,
                         (new_s1 * new_s2) / (l_float * (l_float + 1)))
        epsilon = jnp.finfo(alpha_hi.dtype).eps
        safe_alpha_hi = jnp.where(jnp.abs(alpha_hi) < epsilon,
                                  jnp.sign(alpha_hi) * epsilon + epsilon, alpha_hi)
        w_new = ((2*l + 1) * (cos_theta - beta) * w_hi - alpha_lo * w_lo) / safe_alpha_hi
        w_new = jnp.where(jnp.abs(alpha_hi) < epsilon, 0.0, w_new)
        fac_l = (2*(l+1)+1)/(4*jnp.pi) if prefactor else 1.0
        cond_l = (l+1 >= lmin) & (l+1 <= lmax) & (l+1 < len(cl))
        cf_update = jnp.where(cond_l, cl[l+1] * w_new * fac_l, 0.0)
        return (w_new, w_hi, cf + cf_update), None
    
    if lmax > l0:
        (_, _, cf_final), _ = jax.lax.scan(body_fn, (wigd_hi, wigd_lo, cf), jnp.arange(l0, lmax))
    else:
        cf_final = cf
    return cf_final

@partial(jax.jit, static_argnums=(0, 1, 5))
def cl_from_cf(s1: int, s2: int, cf: jax.Array, cos_theta: jax.Array, 
               weights: jax.Array, lmax: int) -> jax.Array:
    new_s1, new_s2, l0, sign = wigd_init(s1, s2)
    if lmax < l0:
        return jnp.zeros(lmax + 1)
    
    norm_factor_A = _calculate_norm_factor_A(new_s1, new_s2) * sign
    
    term1 = jnp.maximum(0.0, (1 + cos_theta)/2) ** ((new_s2 + new_s1)/2)
    term2 = jnp.maximum(0.0, (1 - cos_theta)/2) ** ((new_s2 - new_s1)/2)
    wigd_hi = norm_factor_A * term1 * term2 * weights
    wigd_lo = jnp.zeros_like(cos_theta)
    
    cl = jnp.zeros(lmax + 1)
    cl = cl.at[l0].set(jnp.sum(cf * wigd_hi) if l0 <= lmax else 0.0)
    
    def body_fn(carry, l):
        w_hi, w_lo, cl = carry
        l_float = l.astype(w_hi.dtype)
        alpha_hi = alpha_f(l_float + 1, new_s1, new_s2)
        alpha_lo = alpha_f(l_float, new_s1, new_s2)
        beta = jnp.where((new_s1 == 0) | (new_s2 == 0), 0.0,
                         (new_s1 * new_s2) / (l_float * (l_float + 1)))
        epsilon = jnp.finfo(alpha_hi.dtype).eps
        safe_alpha_hi = jnp.where(jnp.abs(alpha_hi) < epsilon,
                                  jnp.sign(alpha_hi) * epsilon + epsilon, alpha_hi)
        w_new = ((2*l + 1) * (cos_theta - beta) * w_hi - alpha_lo * w_lo) / safe_alpha_hi
        w_new = jnp.where(jnp.abs(alpha_hi) < epsilon, 0.0, w_new)
        cl = cl.at[l + 1].set(jnp.where(l + 1 <= lmax, jnp.sum(cf * w_new), 0.0))
        return (w_new, w_hi, cl), None
    
    if lmax > l0:
        (_, _, cl_final), _ = jax.lax.scan(body_fn, (wigd_hi, wigd_lo, cl), jnp.arange(l0, lmax))
    else:
        cl_final = cl
    return cl_final


# Example usages
if __name__ == "__main__":
    s1, s2 = 1, 2
    assert np.allclose(
        alpha_f(np.arange(5, 10), s1, s2), 
        [4.48998886, 5.57773351, 6.63940002, 7.68521307, 8.72062972])

    cos_theta = np.array([-0.75, -0.5 ,-0.25, 0.25, 0.5, 0.75])
    
    lmax = 10
    cl = np.arange(lmax + 1) * 1.0
    cos_theta = np.linspace(-1, 1, 10)
    cf = cf_from_cl(s1, s2, cl, cos_theta, lmax)
    print(f"{cf=}")
    assert np.allclose(
        cf,
        [0.0, -1.4236208075434558, 1.3172886463419415, -0.621077274210937, -0.3104460696913065, 1.8022470909608022, -1.902863895245635, 3.139384174670357, -2.7838894965822467, 0.0])

    weights = np.ones_like(cos_theta)
    cl_recon = cl_from_cf(s1, s2, cf, cos_theta, weights, lmax)
    print(f"{cl_recon=}")
    assert np.allclose(
        cl_recon,
        [0.0, 0.0, 0.08558917001267696, -0.8918690803928095, -1.3643483186273897, -0.7240572707990893, -0.6074739860830027, 1.172257045428006, 0.5797840225510533, 2.2034783702814273, 1.1270394245456616])