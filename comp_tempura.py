#%%
import jax
jax.config.update("jax_enable_x64", True)

import pytempura as tp
from matplotlib import pyplot as plt
import numpy as np
import norm_lens
import utils
import time

mlmax = 4000
lmin = 600
lmax = 3000


combos = ["TT","TE","EE","BB"]

ucls, tcls = utils.get_theory_dicts(nells = None, lmax = mlmax, grad = False)

Als_yilun = {}
t0 = time.time()
Als_yilun["TT"] = norm_lens.qtt(mlmax, lmin, lmax, ucls, tcls)
Als_yilun["TE"] = norm_lens.qte(mlmax, lmin, lmax, ucls, tcls)
Als_yilun["EE"] = norm_lens.qee(mlmax, lmin, lmax, ucls, tcls)
Als_yilun["BB"] = norm_lens.qbb(mlmax, lmin, lmax, ucls, tcls)
t1 = time.time()
print(f"python: {t1-t0} seconds")

Als_tempura = tp.get_norms(combos, ucls, ucls, tcls, lmin, lmax, k_ellmax=mlmax)
t2 = time.time()
print(f"tempura: {t2-t1} seconds")


for c in combos: 
        fig = plt.figure()
        plt.plot(Als_yilun[c][1:]/Als_tempura[c][0][1:], label="Yilun/tempura")
        plt.hlines(1.,0,4001,colors="r")
        plt.legend()
        plt.xlim(1,4001)
        plt.xscale("log")
        plt.ylabel(f"A_L^{c}")
        plt.tight_layout()
        plt.savefig(f"ratio_comp_yilun_A{c}.png") 
        plt.close(fig)
        
        fig = plt.figure()
        plt.plot(Als_yilun[c][1:], label="Yilun")
        plt.plot(Als_tempura[c][0][1:], label="Tempura")
        plt.legend()
        plt.xlim(1,4001)
        plt.yscale("log")
        plt.ylabel(f"A_L^{c}")
        plt.tight_layout()
        plt.savefig(f"comp_yilun_A{c}.png") 
        plt.close(fig)


##### Yilun Guan code
#assert np.all(np.nan_to_num(np.array(Als_TT)) == pytest.approx(Als["TT"][0],rel=1e-1))
#assert np.all(np.array(Als_TE) == pytest.approx(Als["TE"][0],rel=1e-3))
#assert np.all(np.array(Als_BB) == pytest.approx(Als["BB"][0],rel=1e-3))
#assert np.all(np.array(Als_EE) == pytest.approx(Als["EE"][0],rel=1e-3))

