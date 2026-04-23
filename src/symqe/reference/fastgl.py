#%%
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import Tuple
from dataclasses import dataclass
from functools import partial

# Modified versions of the FastGL code
# by Ignace Bogaert. The code is available at
# https://sourceforge.net/projects/fastgausslegendrequadrature/ 
# A paper describing the algorithms is available at
# https://epubs.siam.org/doi/pdf/10.1137/140954969
# This code is mostly adapted from the ducc0 implementation
# by Martin Reinecke available at
# https://gitlab.mpcdf.mpg.de/mtr/ducc/

def besseljzero(k):
    JZ = jnp.array([
        2.40482555769577276862163187933,  5.52007811028631064959660411281,
        8.65372791291101221695419871266,  11.7915344390142816137430449119,
        14.9309177084877859477625939974, 18.0710639679109225431478829756,
        21.2116366298792589590783933505, 24.3524715307493027370579447632,
        27.4934791320402547958772882346, 30.6346064684319751175495789269,
        33.7758202135735686842385463467, 36.9170983536640439797694930633
    ], dtype=jnp.float64)
    
    def compute_large_k(k):
        z = np.pi * (k - 0.25)
        r = 1.0 / z
        r2 = r * r
        
        # Coefficients in reverse order (highest degree first)
        coeffs = jnp.array([
            2092163573./82575360.,
            -6277237./3440640.,
            3779./15360.,
            -31./384.,
            0.125,
        ], dtype=jnp.float64)
        
        return z + r * jnp.polyval(coeffs, r2)
    
    return jnp.where(k <= 12, JZ[k-1], compute_large_k(k))

def besselj1squared(k):
    J1 = jnp.array([
        0.269514123941916926139021992911, 0.115780138582203695807812836182,
        0.0736863511364082151406476811985, 0.0540375731981162820417749182758,
        0.0426614290172430912655106063495, 0.0352421034909961013587473033648,
        0.0300210701030546726750888157688, 0.0261473914953080885904584675399,
        0.0231591218246913922652676382178, 0.0207838291222678576039808057297,
        0.0188504506693176678161056800214, 0.0172461575696650082995240053542,
        0.0158935181059235978027065594287
    ], dtype=jnp.float64)
    
    def compute_large_k(k):
        x = 1.0 / (k - 0.25)
        x2 = x * x
        
        # Coefficients in reverse order (highest degree first)
        coeffs = jnp.array([
            0.433710719130746277915572905025e-3,
            -0.228969902772111653038747229723e-3,
            0.198924364245969295201137972743e-3,
            -0.303380429711290253026202643516e-3,
            0.0,  # no x^2 term
            0.202642367284675542887758926420   # constant term
        ], dtype=jnp.float64)
        
        return x * jnp.polyval(coeffs, x2)
    
    return jnp.where(k <= 13, J1[k-1], compute_large_k(k))

@jit
def calc_gl_bogaert(n: int, k0: int) -> Tuple[float, float, float]:
    k = jnp.where((2*k0-1) <= n, k0, n-k0+1)
    
    # First get the Bessel zero
    w = 1.0 / (n + 0.5)
    nu = besseljzero(k)
    theta = w * nu
    x = theta * theta
    
    # Get the asymptotic BesselJ(1,nu) squared
    B = besselj1squared(k)
    
    # Coefficients for Chebyshev interpolants (highest degree first)
    SF1_coeffs = jnp.array([
        -1.29052996274280508473467968379e-12,
        2.40724685864330121825976175184e-10,
        -3.13148654635992041468855740012e-8,
        0.275573168962061235623801563453e-5,
        -0.148809523713909147898955880165e-3,
        0.416666666665193394525296923981e-2,
        -0.416666666666662959639712457549e-1
    ], dtype=jnp.float64)
    
    SF2_coeffs = jnp.array([
        2.20639421781871003734786884322e-9,
        -7.53036771373769326811030753538e-8,
        0.161969259453836261731700382098e-5,
        -0.253300326008232025914059965302e-4,
        0.282116886057560434805998583817e-3,
        -0.209022248387852902722635654229e-2,
        0.815972221772932265640401128517e-2
    ], dtype=jnp.float64)
    
    SF3_coeffs = jnp.array([
        -2.97058225375526229899781956673e-8,
        5.55845330223796209655886325712e-7,
        -0.567797841356833081642185432056e-5,
        0.418498100329504574443885193835e-4,
        -0.251395293283965914823026348764e-3,
        0.128654198542845137196151147483e-2,
        -0.416012165620204364833694266818e-2
    ], dtype=jnp.float64)
    
    WSF1_coeffs = jnp.array([
        -2.20902861044616638398573427475e-14,
        2.30365726860377376873232578871e-12,
        -1.75257700735423807659851042318e-10,
        1.03756066927916795821098009353e-8,
        -4.63968647553221331251529631098e-7,
        0.149644593625028648361395938176e-4,
        -0.326278659594412170300449074873e-3,
        0.436507936507598105249726413120e-2,
        -0.305555555555553028279487898503e-1,
        0.833333333333333302184063103900e-1
    ], dtype=jnp.float64)
    
    WSF2_coeffs = jnp.array([
        3.63117412152654783455929483029e-12,
        7.67643545069893130779501844323e-11,
        -7.12912857233642220650643150625e-9,
        2.11483880685947151466370130277e-7,
        -0.381817918680045468483009307090e-5,
        0.465969530694968391417927388162e-4,
        -0.407297185611335764191683161117e-3,
        0.268959435694729660779984493795e-2,
        -0.111111111111214923138249347172e-1
    ], dtype=jnp.float64)
    
    WSF3_coeffs = jnp.array([
        2.01826791256703301806643264922e-9,
        -4.38647122520206649251063212545e-8,
        5.08898347288671653137451093208e-7,
        -0.397933316519135275712977531366e-5,
        0.200559326396458326778521795392e-4,
        -0.422888059282921161626339411388e-4,
        -0.105646050254076140548678457002e-3,
        -0.947969308958577323145923317955e-4,
        0.656966489926484797412985260842e-2
    ], dtype=jnp.float64)
    
    # Evaluate polynomials
    SF1T = jnp.polyval(SF1_coeffs, x)
    SF2T = jnp.polyval(SF2_coeffs, x)
    SF3T = jnp.polyval(SF3_coeffs, x)
    WSF1T = jnp.polyval(WSF1_coeffs, x)
    WSF2T = jnp.polyval(WSF2_coeffs, x)
    WSF3T = jnp.polyval(WSF3_coeffs, x)
    
    # Refine with paper expansions
    NuoSin = nu / jnp.sin(theta)
    BNuoSin = B * NuoSin
    WInvSinc = w * w * NuoSin
    WIS2 = WInvSinc * WInvSinc
    
    # Compute node and weight
    theta = w * (nu + theta * WInvSinc * (SF1T + WIS2*(SF2T + WIS2*SF3T)))
    Deno = BNuoSin + BNuoSin * WIS2*(WSF1T + WIS2*(WSF2T + WIS2*WSF3T))
    weight = (2.0 * w) / Deno
    
    return (jnp.where(k == k0, jnp.cos(theta), -jnp.cos(theta)),
            weight,
            jnp.where(k == k0, theta, jnp.pi-theta))


@partial(jax.jit, static_argnums=(0,))
def compute_gl_points(n: int):
    """Compute Gauss-Legendre points and weights using vectorized operations"""
    # only need to compute half the points due to symmetry
    m = (n + 1) >> 1
    def body_fun(i):
        return calc_gl_bogaert(n, m-i)
    indices = jnp.arange(m)
    x, w, th = jax.vmap(body_fun)(indices)
    return x, w, th

@dataclass
class FastGL:
    w: jnp.ndarray
    x: jnp.ndarray
    def __init__(self, n: int):
        assert n >= 1, "number of points must be at least 1"
        x, w, _ = compute_gl_points(n)
        if n % 2 == 0:
            self.x = jnp.hstack([-x[::-1], x])
            self.w = jnp.hstack([w[::-1], w])
        else:
            self.x = jnp.hstack([-x[::-1], x[1:]])
            self.w = jnp.hstack([w[::-1], w[1:]])
            
if __name__ == '__main__':
    def test_gl(n):
        gl = FastGL(n)
        if np.allclose(gl.w, np.polynomial.legendre.leggauss(n)[1], rtol=1e-8):
            print(f"Success! GL_Integrator({n}) has correct weights.")
        else:
            print(f"Failure! GL_Integrator({n}) has incorrect weights.")
        if np.allclose(gl.x, np.polynomial.legendre.leggauss(n)[0], rtol=1e-8):
            print(f"Success! GL_Integrator({n}) has correct x.")
        else:
            print(f"Failure! GL_Integrator({n}) has incorrect x.")
            print(f"{(gl.x - np.polynomial.legendre.leggauss(n)[0])=}")
    test_gl(101)
    test_gl(200)
    test_gl(301)
    test_gl(501)
    test_gl(1000)
    test_gl(1501)
    test_gl(2000)
    test_gl(2345)

# %%
