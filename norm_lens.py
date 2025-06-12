import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from glquad import GLQuad

def qtt(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['TT']
    B = ucl['TT']**2/ocl['TT']
    res = kernel_S0(lmax, rlmin, rlmax, A, B)
    A = ucl['TT']/ocl['TT']
    res += kernel_G0(lmax, rlmin, rlmax, A, A)
    return res**-1

def qte(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['TT']
    B = ucl['TE']**2/ocl['EE']
    res = kernel_S0(lmax, rlmin, rlmax, A, B)
    A = ucl['TE']/ocl['TT']
    B = ucl['TE']/ocl['EE']
    res += 2*kernel_Gx(lmax, rlmin, rlmax, A, B)
    A = 1/ocl['EE']
    B = ucl['TE']**2/ocl['TT']
    res += kernel_Sp(lmax, rlmin, rlmax, A, B)
    return res**-1

def qtb(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['BB']
    B = ucl['TE']**2/ocl['TT']
    res = kernel_Sm(lmax, rlmin, rlmax, A, B)
    return res**-1

def qee(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['EE']
    B = ucl['EE']**2/ocl['EE']
    res = kernel_Sp(lmax, rlmin, rlmax, A, B)
    A = ucl['EE']/ocl['EE']
    res += kernel_Gp(lmax, rlmin, rlmax, A, A)
    return res**-1

def qbb(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['BB']
    B = ucl['BB']**2/ocl['BB']
    res = kernel_Sp(lmax, rlmin, rlmax, A, B)
    A = ucl['BB']/ocl['BB']
    res += kernel_Gp(lmax, rlmin, rlmax, A, A)
    return res**-1

def qeb(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['EE']
    B = ucl['BB']**2/ocl['BB']
    res = kernel_Sm(lmax, rlmin, rlmax, A, B)
    A = ucl['BB']/ocl['BB']
    B = ucl['EE']/ocl['EE']
    res += 2*kernel_Gm(lmax, rlmin, rlmax, A, B)
    A = 1/ocl['BB']
    B = ucl['EE']**2/ocl['EE']
    res += kernel_Sm(lmax, rlmin, rlmax, A, B)
    return res**-1

def qttte(lmax, rlmin, rlmax, ucl, ocl):
    A = 1/ocl['TT']
    B = ucl['TT']*ucl['TE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res = kernel_S0(lmax, rlmin, rlmax, A, B)
    A = ucl['TE']/ocl['TT']
    B = ucl['TT']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res += kernel_Gx(lmax, rlmin, rlmax, A, B)
    A = ucl['TE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['TT']/ocl['TT']
    res += kernel_G0(lmax, rlmin, rlmax, A, B)
    A = ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['TE']*ucl['TT']/ocl['TT']
    res += kernel_Sx(lmax, rlmin, rlmax, A, B)
    return res

def qttee(lmax, rlmin, rlmax, ucl, ocl):
    A = ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['TT']*ucl['EE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res = kernel_Sx(lmax, rlmin, rlmax, A, B)
    A = ocl['TE']*ucl['EE']/(ocl['TT']*ocl['EE'])
    B = ucl['TT']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res += kernel_Gx(lmax, rlmin, rlmax, A, B)
    return res

def qteee(lmax, rlmin, rlmax, ucl, ocl):
    A = ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['TE']*ucl['EE']/ocl['EE']
    res = kernel_Sx(lmax, rlmin, rlmax, A, B)
    A = ucl['TE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['EE']/ocl['EE']
    res += kernel_Gp(lmax, rlmin, rlmax, A, B)
    A = (ocl['TE']*ucl['EE'])/(ocl['TT']*ocl['EE'])
    B = (ucl['TE']*ocl['EE'])/(ocl['EE']**2)
    res += kernel_Gx(lmax, rlmin, rlmax, A, B)
    A = 1/ocl['EE']
    B = ucl['TE']*ucl['EE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res += kernel_Sp(lmax, rlmin, rlmax, A, B)
    return res

def qtbeb(lmax, rlmin, rlmax, ucl, ocl):
    A = ucl['TE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    B = ucl['BB']/ocl['BB']
    res = kernel_Gm(lmax, rlmin, rlmax, A, B)
    A = 1/ocl['BB']
    B = ucl['TE']*ucl['EE']*ocl['TE']/(ocl['TT']*ocl['EE'])
    res += kernel_Sm(lmax, rlmin, rlmax, A, B)
    return res

# A more generic method is to build several basis kernels
# and combine them to get various estimator normalizations:
# kernels: S0, Sp, Sm, Sx, G0, Gp, Gm, Gx, standing for 
# Σ⁰, Σ⁺, Σ⁻, Σˣ, Γ⁰, Γ⁺, Γ⁻, Γˣ in Toshiya's notation

def kernel_S0(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    term1 = glq.cf_from_cl(0, 0, A, lmin=rlmin, lmax=rlmax)
    term2 = glq.cf_from_cl(1, -1, B*l*(l+1), lmin=rlmin, lmax=rlmax, prefactor=True)
    A = glq.cl_from_cf(1, -1, term1*term2, lmax=lmax)
    term2 = glq.cf_from_cl(1,  1, B*l*(l+1), lmin=rlmin, lmax=rlmax, prefactor=True)
    B = glq.cl_from_cf(1,  1, term1*term2, lmax=lmax)
    return np.pi * (A + B) * l*(l+1)

def kernel_Sp(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    
    # Term for l₁
    term1 = glq.cf_from_cl(2, -2, A, lmin=rlmin, lmax=rlmax)
    term1_plus = glq.cf_from_cl(2, 2, A, lmin=rlmin, lmax=rlmax)
    
    # Terms for l₂
    # For (1, ±1)
    term2_1 = glq.cf_from_cl(1, -1, B*(l-1)*(l+2), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_1_plus = glq.cf_from_cl(1, 1, B*(l-1)*(l+2), lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # For (3, ±1)
    sqrt_term = jnp.sqrt((l-1)*(l+2)*(l-2)*(l+3))
    term2_3 = glq.cf_from_cl(3, -1, B*sqrt_term, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_3_plus = glq.cf_from_cl(3, 1, B*sqrt_term, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # For (3, ±3)
    term2_33 = glq.cf_from_cl(3, -3, B*(l-2)*(l+3), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_33_plus = glq.cf_from_cl(3, 3, B*(l-2)*(l+3), lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Combine terms
    A1 = glq.cl_from_cf(1, -1, term1*term2_1, lmax=lmax)
    A2 = glq.cl_from_cf(1, 1, term1_plus*term2_1_plus, lmax=lmax)
    A3 = glq.cl_from_cf(1, 1, term1*term2_3, lmax=lmax)
    A4 = glq.cl_from_cf(1, -1, term1_plus*term2_3_plus, lmax=lmax)
    A5 = glq.cl_from_cf(1, 1, term1_plus*term2_33_plus, lmax=lmax)
    A6 = glq.cl_from_cf(1, -1, term1*term2_33, lmax=lmax)
    
    # Final combination with prefactors
    prefactor = np.pi/4*l*(l+1)
    return (A1 + A2 + 2*A3 + 2*A4 + A5 + A6) * l*(l+1) * prefactor

def kernel_Sm(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    
    # Term for l₁ with spin-2
    term1_m2 = glq.cf_from_cl(2, -2, A, lmin=rlmin, lmax=rlmax, prefactor=True)
    term1_p2 = glq.cf_from_cl(2, 2, A, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Terms for l₂ with different spins
    # For spin-1 terms
    term2_s1_m1 = glq.cf_from_cl(1, -1, B*(l-1)*(l+2), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_s1_p1 = glq.cf_from_cl(1, 1, B*(l-1)*(l+2), lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # For spin-3 terms
    sqrt_term = jnp.sqrt((l-1)*(l+2)*(l-2)*(l+3))
    term2_s3_m3 = glq.cf_from_cl(3, -3, B*(l-2)*(l+3), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_s3_p3 = glq.cf_from_cl(3, 3, B*(l-2)*(l+3), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_s3_m1 = glq.cf_from_cl(3, -1, B*sqrt_term, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_s3_p1 = glq.cf_from_cl(3, 1, B*sqrt_term, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    A = (glq.cl_from_cf(1, -1, term1_m2*term2_s1_m1, lmax=lmax) * (-1) +
         glq.cl_from_cf(1, 1, term1_p2*term2_s1_p1, lmax=lmax) * (1) +
         glq.cl_from_cf(1, 1, term1_m2*term2_s3_m1, lmax=lmax) * (-2) +
         glq.cl_from_cf(1, -1, term1_p2*term2_s3_p1, lmax=lmax) * (2) +
         glq.cl_from_cf(1, 1, term1_p2*term2_s3_p3, lmax=lmax) * (1) +
         glq.cl_from_cf(1, -1, term1_m2*term2_s3_m3, lmax=lmax) * (-1)) / (4/np.pi)
    
    return A*l*(l+1)

def kernel_Sx(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    
    # Term for l₁
    term1 = glq.cf_from_cl(2, 0, A, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Terms for l₂ with spin-1
    factor1 = B*jnp.sqrt((l-1)*(l+2)*l*(l+1))
    term2a = glq.cf_from_cl(1, -1, factor1, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2b = glq.cf_from_cl(1,  1, factor1, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Terms for l₂ with spin-3
    factor2 = B*jnp.sqrt((l-2)*(l+3)*l*(l+1))
    term2c = glq.cf_from_cl(3, -1, factor2, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2d = glq.cf_from_cl(3,  1, factor2, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Combine terms
    A = glq.cl_from_cf(1, -1, term1*(term2a + term2c), lmax=lmax)
    B = glq.cl_from_cf(1,  1, term1*(term2b + term2d), lmax=lmax)
    
    return np.pi * (A + B) * l*(l+1) / 4

def kernel_G0(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    term1 = glq.cf_from_cl(1, 0, A * (1 + l)**0.5 * l**0.5, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2 = glq.cf_from_cl(1, 0, B * (1 + l)**0.5 * l**0.5, lmin=rlmin, lmax=rlmax, prefactor=True)
    A = glq.cl_from_cf(1, -1, term1*term2, lmax=lmax)
    B = glq.cl_from_cf(1, 1, term1*term2, lmax=lmax)
    return np.pi * (A - B)*l*(l+1)
    
def kernel_Gp(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    l1, l2 = l, l
    A1, B2 = A, B

    # Term pairs with (s1,s2) = (1,-2) for l1
    term1_l1 = glq.cf_from_cl(1, -2, A1 * jnp.sqrt((l1-1)*(l1+2)), lmin=rlmin, lmax=rlmax, prefactor=True)
    # Term pairs with (s1,s2) = (3,-2) for l1
    term2_l1 = glq.cf_from_cl(3, -2, A1 * jnp.sqrt((l1-2)*(l1+3)), lmin=rlmin, lmax=rlmax, prefactor=True)
    # Term pairs with (s1,s2) = (1,2) for l1
    term3_l1 = glq.cf_from_cl(1, 2, A1 * jnp.sqrt((l1-1)*(l1+2)), lmin=rlmin, lmax=rlmax, prefactor=True)
    # Term pairs with (s1,s2) = (3,2) for l1
    term4_l1 = glq.cf_from_cl(3, 2, A1 * jnp.sqrt((l1-2)*(l1+3)), lmin=rlmin, lmax=rlmax, prefactor=True)

    # Terms for l2
    term1_l2 = glq.cf_from_cl(3, -2, B2 * jnp.sqrt((l2-2)*(l2+3)), lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_l2 = glq.cf_from_cl(2, -1, B2 * jnp.sqrt((l2-1)*(l2+2)), lmin=rlmin, lmax=rlmax, prefactor=True)
    term3_l2 = glq.cf_from_cl(2, 1, B2 * jnp.sqrt((l2-1)*(l2+2)), lmin=rlmin, lmax=rlmax, prefactor=True)
    term4_l2 = glq.cf_from_cl(3, 2, B2 * jnp.sqrt((l2-2)*(l2+3)), lmin=rlmin, lmax=rlmax, prefactor=True)

    result1 = glq.cl_from_cf(1, -1, term1_l1*term1_l2 + term3_l1*term4_l2 - term1_l1*term2_l2 - term3_l1*term2_l2, lmax=lmax)
    result2 = glq.cl_from_cf(1, 1, -term2_l1*term1_l2 - term2_l1*term2_l2 - term4_l1*term3_l2 + term4_l1*term4_l2, lmax=lmax)
    
    # Final combination with common factors
    return np.pi * (result1 + result2) * l*(l+1) / 4

def kernel_Gm(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    
    # Terms for l1
    f1 = A * jnp.sqrt((l-1)*(l+2))  # for spin-1
    f3 = A * jnp.sqrt((l-2)*(l+3))  # for spin-3
    
    term1_p2 = glq.cf_from_cl(1, 2, f1, lmin=rlmin, lmax=rlmax, prefactor=True)
    term1_m2 = glq.cf_from_cl(1, -2, f1, lmin=rlmin, lmax=rlmax, prefactor=True)
    term1_p3 = glq.cf_from_cl(3, 2, f3, lmin=rlmin, lmax=rlmax, prefactor=True)
    term1_m3 = glq.cf_from_cl(3, -2, f3, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    # Terms for l2
    f2 = B * jnp.sqrt((l-1)*(l+2))  # for spin-2
    f3_2 = B * jnp.sqrt((l-2)*(l+3))  # for spin-3
    
    term2_p1 = glq.cf_from_cl(2, 1, f2, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_m1 = glq.cf_from_cl(2, -1, f2, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_p2 = glq.cf_from_cl(3, 2, f3_2, lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_m2 = glq.cf_from_cl(3, -2, f3_2, lmin=rlmin, lmax=rlmax, prefactor=True)
    
    A = glq.cl_from_cf(1, -1, term1_m2*term2_m2 + term1_p3*term2_m1 + 
                              term1_p2*term2_m1 + term1_m3*term2_m2, lmax=lmax)
    B = glq.cl_from_cf(1, 1, term1_m3*term2_m2 + term1_m2*term2_p1 + 
                             term1_p2*term2_p2 + term1_p3*term2_p1, lmax=lmax)

    return np.pi * (A + B) * l*(l+1) / 4

def kernel_Gx(lmax, rlmin, rlmax, A, B):
    l = jnp.arange(0, lmax+1)
    glq = GLQuad(int((3*lmax+1)/2))
    
    term1_l1 = glq.cf_from_cl(1, 0, A*jnp.sqrt((l-1)*(l+2)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_l1 = glq.cf_from_cl(3, 0, A*jnp.sqrt((l-2)*(l+3)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    
    term1_l2 = glq.cf_from_cl(1, -2, B*jnp.sqrt(l*(l+1)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    term2_l2 = glq.cf_from_cl(2, -1, B*jnp.sqrt(l*(l+1)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    term3_l2 = glq.cf_from_cl(1, 2, B*jnp.sqrt(l*(l+1)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    term4_l2 = glq.cf_from_cl(2, 1, B*jnp.sqrt(l*(l+1)), 
                              lmin=rlmin, lmax=rlmax, prefactor=True)
    
    A = glq.cl_from_cf(1, -1, term1_l1*term1_l2, lmax=lmax)
    B = glq.cl_from_cf(1, 1, term2_l1*term2_l2, lmax=lmax)
    C = glq.cl_from_cf(1, 1, term1_l1*term3_l2, lmax=lmax)
    D = glq.cl_from_cf(1, -1, term2_l1*term4_l2, lmax=lmax)
    
    return np.pi * (-A - B + C - D) * l*(l+1) / 4


def qtt_simple(lmax, rlmin, rlmax, ucl, ocl):
    glq = GLQuad(int((3*lmax + 1)/2))
    ucl, ocl = ucl['TT'], ocl['TT']
    
    ell = jnp.arange(0, len(ucl))
    llp1 = ell * (ell + 1)
    div_dl = 1/ocl
    cl_div_dl = ucl/ocl
    
    zeta_00 = glq.cf_from_cl(0, 0, div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_01_p = glq.cf_from_cl(0, 1, jnp.sqrt(llp1) * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_01_m = glq.cf_from_cl(0, -1, jnp.sqrt(llp1) * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_11_p = glq.cf_from_cl(1, 1, llp1 * ucl * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    zeta_11_m = glq.cf_from_cl(1, -1, llp1 * ucl * cl_div_dl, prefactor=True, lmin=rlmin, lmax=rlmax)
    
    nlpp_term_1 = glq.cl_from_cf(-1, -1, zeta_00*zeta_11_p - zeta_01_p**2, lmax)
    nlpp_term_2 = glq.cl_from_cf(1, -1, zeta_00*zeta_11_m - zeta_01_p*zeta_01_m, lmax)
    
    return 1/(np.pi * llp1 * (nlpp_term_1 + nlpp_term_2))

if __name__ == '__main__':
    import pytempura as tp
    from matplotlib import pyplot as plt

    cltt = np.arange(1, 102, dtype=np.float64)
    nltt = np.zeros_like(cltt)
    ucl = {'TT': cltt}
    ocl = {'TT': cltt+nltt}

    lmax_p = 100
    rtt_simple = qtt_simple(100, 1, 100, ucl, ocl)
    rtt_kernel = qtt(100, 1, 100, ucl, ocl)
    rtt_tp = tp.norm_lens.qtt(lmax_p, 1, lmax_p, ucl['TT'], ucl['TT'], ocl['TT'])[0]

    plt.figure()
    plt.plot(rtt_simple, label="simple")
    plt.plot(rtt_tp, label="tp")
    plt.plot(rtt_kernel, label="jax kernel")
    plt.legend()
    plt.yscale('log')
    plt.xlim(left=1)
    plt.show() 

    # agreement within rtol=1e-9