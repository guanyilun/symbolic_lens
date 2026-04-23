import os
import numpy as np
from orphics import cosmology


def get_theory_dicts(nells=None,lmax=9000,grad=False):
    assert grad is False, "grad=True not implemented, use falafel instead"
    ls = np.arange(lmax+1)
    ucls = {}
    tcls = {}
    theory = cosmology.default_theory()
    if nells is None: nells = {'TT':0,'EE':0,'BB':0}
    ucls['TT'] = theory.lCl('TT',ls)
    ucls['TE'] = theory.lCl('TE',ls)
    ucls['EE'] = theory.lCl('EE',ls)
    ucls['BB'] = theory.lCl('BB',ls)
    ucls['kk'] = theory.gCl('kk',ls)
    tcls['TT'] = theory.lCl('TT',ls) + nells['TT']
    tcls['TE'] = theory.lCl('TE',ls)
    tcls['EE'] = theory.lCl('EE',ls) + nells['EE']
    tcls['BB'] = theory.lCl('BB',ls) + nells['BB']
    return ucls, tcls