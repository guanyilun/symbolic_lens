import numpy as np
import sympy as sp
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.quantum.spin import WignerD
from sympy import pi, symbols
from sympy import Sum, Indexed, Function
from sympy import sin,cos,acos,asin,sqrt,integrate, lambdify, sympify, IndexedBase, Integral

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

l,m1,m2 = symbols("l,m1,m2",integer=True)
beta,mu = symbols("beta,mu")

d = WignerD(l,m1,m2,0,beta,0)
lmin,lmax = symbols("lmin,lmax")
Y = IndexedBase("Y")

xi = Sum(((2*l+1)/(4*pi)*Y[l]*d),(l,lmin,lmax))
s = symbols("s")
a_s_l = -1*sqrt((l-s)*(l+s+1)/2)
X = IndexedBase('X')
X0 = sqrt(2) * a_s_l.subs(s,0) * X[l]
X00 = sqrt(2)*a_s_l.subs(s,0)*X0
Xp = sqrt(2)*a_s_l.subs(s,2)*X[l]
Xpp = sqrt(2)*a_s_l.subs(s,2)*Xp
Xm = sqrt(2)*a_s_l.subs(s,-2)*X[l]
Xmm = sqrt(2)*a_s_l.subs(s,-2)*Xm
Xpm = sqrt(2)*a_s_l.subs(s,2)*Xm
X0m = sqrt(2)*a_s_l.subs(s,0)*Xm
X0p = sqrt(2)*a_s_l.subs(s,0)*Xp

l1=symbols("l1",integer=True)
A = IndexedBase("A")
l2=symbols("l2",integer=True)
B = IndexedBase("B")
xi_A_00 = xi.xreplace({Y[l]:A[l1],l:l1}).subs({m1:0,m2:0})
B00 = X00.subs({X:B})
xi_B00_11 = xi.xreplace({Y[l]:B00.xreplace({l:l2}),l:l2}).subs({m1:1,m2:1})
xi_B00_1m1 = xi.xreplace({Y[l]:B00.xreplace({l:l2}),l:l2}).subs({m1:1,m2:-1})

L = symbols("L", integer=True)

d = WignerD(l,m1,m2,0,beta,0)
term1 = xi_A_00*xi_B00_11*d.subs({l:L,m1:1,m2:1})
term2 = xi_A_00*xi_B00_1m1*d.subs({l:L,m1:1,m2:-1})
integrand = pi*L*(L+1)*(term1+term2)
integrand_mu = integrand.xreplace({beta:acos(mu)})

#Sigma_0 = Integral(term1+term2,beta).transform(beta,(mu,cos(beta)))
Sigma_0 = Integral(integrand_mu,(mu,-1,1))

"""
def d(l,m1,m2,beta):
    return WignerD(l,m1,m2,0,beta,0)

def xi(l,A,m1,m2,beta,lmin,lmax):
    return Sum(((2*l+1)/(4*pi)*A(l)*d(l,m1,m2,beta)),(l,lmin,lmax))

def a_s_l(l,s):
    return -1*sqrt((l-s)*(l+s+1)/2)

def X_0(l,X):
    return sqrt(2)*a_s_l(l,0)*X

def X_00(l,X):
    return X_0(l,X_0(l,X))

def X_m(l,X):
    return sqrt(2)*a_s_l(l,-2)*X

def X_mm(l,X):
    return X_m(l,X_m(l,X))

def X_p(l,X):
    return sqrt(2)*a_s_l(l,2)*X

def X_pp(l,X):
    return X_p(l,X_p(l,X))

def X_pm(l,X):
    return X_p(l,X_m(l,X))

def X_0p(l,X):
    return X_0(l,X_p(l,X))

def X_0m(l,X):
    return X_0(l,X_m(l,X)) 


def Sigma_0(l1,A,l2,B,L,beta):
    mu = cos(beta)
    xi_A_00 = xi(l1,A,0,0,beta)
    B00 = X_00(l2,B)
    xi_B00_11 = xi(l2,B00,1,1,beta)
    xi_B00_1m1 = xi(l2,B00,1,-1)
    term1 = xi_A_00*xi_B00_11*d(L,1,1,beta)
    term2 = xi_A_00*xi_B00_1m1*d(L,1,-1,beta)
    return integrate(pi*L*(L+1)*(term1+term2),(mu,1,-1))  

def Gamma_0(l1,A,l2,B,L,beta):
    mu = cos(beta)
    A0 = X_0(l1,A)
    xi_A0_01 = xi(l1,A0,0,1,beta)
    B0 = X_0(l2,B)
    xi_B0_0m1 = xi(l2,B,0,-1,beta)
    xi_B0_01 = xi(l2,B,0,1,beta)
    term1 = xi_A0_01*xi_B0_0m1*d(L,1,-1,beta)
    term2 = xi_A0_01*xi_B0_01*d(L,1,1,beta)
    return integrate(pi*L*(L+1)*(term1+term2),(mu,1,-1))

def Sigma_pm(l1,A,l2,B,L,pm,lmin,lmax):
    beta = symbols('beta')
    mu = cos(beta)
    xi_A_22 = xi(l1,A,2,2,beta,lmin,lmax)
    Bpp = lambdify([l,X],X_pp)(l2,B)
    xi_Bpp_33 = xi(l2,Bpp,3,3,beta,lmin,lmax)
    Bmm = X_mm(l2,B)
    xi_Bmm_11 = xi(l2,Bmm,1,1,beta,lmin,lmax)
    xi_A_2m2 = xi(l1,A,2,-2,beta,lmin,lmax)
    Bpm = X_pm(l2,B)
    xi_Bpm_3m1 = xi(l2,Bpm,3,-1,lmin,lmax)
    term1 = ((xi_A_22*xi_Bpp_33) + (xi_A_22*xi_Bmm_11) + 2*pm*(xi_A_2m2*xi_Bpm_3m1))*d(L,1,1,beta)
    xi_Bpp_3m3 = xi(l2,Bpp,3,-3,beta,lmin,lmax)
    xi_Bmm_1m1 = xi(l2,Bmm,1,-1,beta,lmin,lmax)
    xi_Bpm_31 = xi(l2,Bpm,3,1,beta,lmin,lmax)
    term2 = ((pm*xi_A_2m2*xi_Bpp_3m3)+(pm*xi_A_2m2*xi_Bmm_1m1)+(2*xi_A_22*xi_Bpm_31))*d(L,1,-1,beta)
    return integrate((pi*L*(L+1)/4)*(term1+term2),(mu,1,-1))
    
    
def Gamma_pm(l1,A,l2,B,L,beta,pm):
    mu = cos(beta)
    Am = X_m(l1,A)
    xi_Am_21 = xi(l1,Am,2,1,beta)
    Bp = X_p(l2,Bp)
    xi_Bp_32 = xi(l2,Bp,3,2,beta)
    Ap = X_p(l1,A)
    return 0

"""