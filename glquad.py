#%%
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from dataclasses import dataclass
from wignerd import cf_from_cl, cl_from_cf
from fastgl import FastGL

@dataclass
class GLQuad:
    x: np.ndarray
    w: np.ndarray

    def __init__(self, n):
        gl = FastGL(n)
        self.x = gl.x
        self.w = gl.w

    def __repr__(self):
        return f"GLQuad(n={len(self.x)})"

    def cf_from_cl(self, s1, s2, cl, lmax=None, lmin=0, prefactor=False):
        if lmax is None: lmax = len(cl) - 1
        return cf_from_cl(s1, s2, cl, self.x, lmax, prefactor, lmin)

    def cl_from_cf(self, s1, s2, cf, lmax):
        return cl_from_cf(s1, s2, cf, self.x, self.w, lmax)
    
if __name__ == '__main__':
    gl = GLQuad(15)
    cl = np.arange(1, 12)
    cf = gl.cf_from_cl(1, 2, cl)
    print(f"{cf=}")
    assert np.allclose(
        cf,
        [0.48637200593039986, 2.488677425018759, 0.5069227587296468, -1.7695498582535323, 1.2503571405011698, 0.5493888326422709, -1.5079165544889677, 1.9124057254944156, 0.5280164782369949, -1.377087728688859, 4.3518056862578165, 0.9142015051323855, -0.7385994154342077, 22.7736581725636, 28.671569539955172]
    )
    cl_new = gl.cl_from_cf(3, 4, cf, 10)
    print(f"{cl_new=}")
    assert np.allclose(
        cl_new,
        [0.0, 0.0, 0.0, 0.0, 1.3134048250068124, 1.076847756097337, 1.1038989013850293, 1.0676029242221057, 1.067067343762469, 1.054159429825423, 1.0511700259417847]
    )

# %%
