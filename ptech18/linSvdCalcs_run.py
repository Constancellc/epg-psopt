import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import exampleClass, linModel, calcVar, hcPdfs

WD = os.path.dirname(sys.argv[0])

fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

LM = linModel(fdr_i,WD)
LM.loadNetModel(LM.netModelNom)


pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom)
Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
Sgm = Mu0/np.sqrt(pdf.pdf['prms'])

# LM.plotNetBuses('logVar')

LM.runLinHc(30,pdf.pdf)

for i in range(5):
    plt.plot(LM.linHcRsl['Cns_pct'][0])
plt.show()


