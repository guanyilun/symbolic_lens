### test alpha
using Wignerd
using PyCall

s1, s2 = 1, 2
f = x -> Wignerd.alpha(x, s1, s2)
f.(collect(5:9))

### test wignerd_init
costheta = Array([-0.75, -0.5 ,-0.25, 0.25, 0.5, 0.75])
wigd_hi = zero(costheta)
Wignerd.wigd_init!(s1, s2, costheta, wigd_hi)
wigd_hi

### test wignerd recur
wigd_lo = zero(costheta)
Wignerd.wigd_rec!(10, s1, s2, costheta, wigd_hi, wigd_lo)
wigd_hi

### test cf_from_cl
lmax = 10
cl = collect(0:lmax)
costheta = Array(range(-1, 1, length=10))
cf = Wignerd.cf_from_cl(s1, s2, lmax, cl, costheta)

### test cl_from_cf
weights = ones(length(costheta))
cl_recon = Wignerd.cl_from_cf(s1, s2, lmax, cf, costheta, weights)

###
gl = Wignerd.glquad(10000)
# cl = Float64.(collect(0:100))
cl = Float64.(collect(1:101))
py"""
import sys
import os
sys.path.append("lens_recon")
from glquad import GLQuad
"""
py_gl = py"GLQuad"(10000)
par = x -> convert(Array{Float64}, x)

s1, s2 = -2, 2
cl_new = zero(cl)
cl_new .= cl
cl_new[1] = 0
# cf = cf_from_cl(gl, s1, s2, 100, cl)
cf = cf_from_cl(gl, s1, s2, 100, cl_new)
py_cf = par(py_gl.cf_from_cl(s1, s2, cl_new, 100))
maximum(abs.(cf - py_cf) ./ cf)

###
plt = pyimport("matplotlib.pyplot")
plt.figure()
plt.plot(abs.((cf - py_cf) ./ cf))
plt.yscale("log")
plt.savefig("test.png")
plt.clf()
###

plt.figure()
plt.plot(abs.((gl.x - py_gl.x) ./ gl.x))
plt.yscale("log")
plt.savefig("test.png")
plt.clf()

###

plt.figure()
plt.plot(abs.((gl.w - py_gl.w) ./ gl.w))
plt.yscale("log")
plt.savefig("test.png")
plt.clf()

###
cl_recon = cl_from_cf(gl, s1, s2, 100, cf)
py_cl_recon = par(py_gl.cl_from_cf(s1, s2, cf, 100))
maximum(abs.(cl_recon - py_cl_recon) / cl_recon)