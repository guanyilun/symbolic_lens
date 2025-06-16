import pytempura as tp
from matplotlib import pyplot as plt
from orphics import cosmology
from orphics import maps
import numpy as np
import norm_lens
import pytest
from falafel import utils as futils

mlmax = 4000
ls = np.arange(2,mlmax+3)

theory = cosmology.loadTheorySpectraFromCAMB("/global/homes/k/kaper/gitreps/falafel/data/cosmo2017_10K_acc3",get_dimensionless=False)
#ells,gt,ge,gb,gte = np.loadtxt(f"/global/homes/k/kaper/gitreps/falafel/data/cosmo2017_10K_acc3_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1,2,3,4])
ucls={}
#ucls['TT'] = maps.interp(ells,gt)(ls)
#ucls['EE'] = maps.interp(ells,ge)(ls)
#ucls['BB'] = maps.interp(ells,gb)(ls)
#ucls['TE'] = maps.interp(ells,gte)(ls)
ucls['TT'] = theory.lCl('TT',ls)
ucls['TE'] = theory.lCl('TE',ls)
ucls['EE'] = theory.lCl('EE',ls)
ucls['BB'] = theory.lCl('BB',ls)

nl = np.zeros_like(ucls['TT'])
tcls = {'TT': ucls['TT']+nl,
        'TE': ucls['TE']+nl,
        'BB': ucls['BB']+nl,
        'EE': ucls['EE']+nl}

lmin = 600
lmax = 3000
rtt_yilun_tt = norm_lens.qtt(mlmax, lmin, lmax, ucls, tcls)
rtt_yilun_te = norm_lens.qte(mlmax, lmin, lmax, ucls, tcls)
rtt_yilun_ee = norm_lens.qee(mlmax, lmin, lmax, ucls, tcls)
rtt_yilun_bb = norm_lens.qbb(mlmax, lmin, lmax, ucls, tcls)

fucls, ftcls = futils.get_theory_dicts(nells = None, lmax = mlmax, grad = False)
Als = tp.get_norms(["TT","TE","EE","BB"], ucls, ucls, tcls, lmin, lmax, k_ellmax=mlmax)


#rtt_tp = tp.norm_lens.qtt(mlmax, lmin, lmax, fucls['TT'], fucls['TT'], ftcls['TT'])[0]
#print(rtt_tp.shape)
#rtt_yilun = norm_lens.qtt(mlmax, lmin, lmax, fucls, ftcls)

fig = plt.figure()
plt.plot(Als["TT"][0], label="tempura")
plt.plot(ls,rtt_yilun_tt, label="Yilun")
plt.legend()
plt.yscale('log')
plt.ylabel("A_L^TT")
plt.xlim(left=1)
plt.savefig("comp_yilun_ATT.png") 
plt.close(fig)

fig = plt.figure()
plt.plot(Als["TT"][0], label="tempura")
plt.plot(ls,rtt_yilun_tt, label="Yilun")
plt.legend()
plt.yscale('log')
plt.ylabel("A_L^TT")
plt.xlim(left=1)
plt.savefig("comp_yilun_ATT.png") 
plt.close(fig)

fig = plt.figure()
plt.plot(Als["TE"][0], label="tempura")
plt.plot(ls,rtt_yilun_te, label="Yilun")
plt.legend()
plt.yscale('log')
plt.ylabel("A_L^TE")
plt.xlim(left=1)
plt.savefig("comp_yilun_ATE.png") 
plt.close(fig)

fig = plt.figure()
plt.plot(Als["EE"][0], label="tempura")
plt.plot(ls,rtt_yilun_ee, label="Yilun")
plt.legend()
plt.yscale('log')
plt.ylabel("A_L^EE")
plt.xlim(left=1)
plt.savefig("comp_yilun_AEE.png") 
plt.close(fig)

fig = plt.figure()
plt.plot(Als["BB"][0], label="tempura")
plt.plot(ls,rtt_yilun_bb, label="Yilun")
plt.legend()
plt.yscale('log')
plt.ylabel("A_L^BB")
plt.xlim(left=1)
plt.savefig("comp_yilun_ABB.png") 
plt.close(fig)

##### Yilun Guan code
#Als_TE = norm_lens.qte(mlmax,lmin,lmax,ucls,tcls)
#Als_TB = norm_lens.qtb(mlmax,lmin,lmax,ucls,tcls)
#Als_EB = norm_lens.qeb(mlmax,lmin,lmax,ucls,tcls)
#Als_EE = norm_lens.qee(mlmax,lmin,lmax,ucls,tcls)

#assert np.all(Als_TT == pytest.approx(np.array(Als["TT"][0]),rel=1e-3))
#assert np.all(Als_TE == pytest.approx(np.array(Als["TE"][0]),rel=1e-3))
#assert np.all(Als_TB == pytest.approx(np.array(Als["TB"][0]),rel=1e-3))
#assert np.all(Als_EB == pytest.approx(np.array(Als["EB"][0]),rel=1e-3))
#assert np.all(Als_EE == pytest.approx(np.array(Als["EE"][0]),rel=1e-3))
#Als_TT_yilun = np.array(Als_TT_yilun)
#Als_TT_yilun[~np.isfinite(Als_TT_yilun)] = 0.
#print(Als_TT_yilun)

