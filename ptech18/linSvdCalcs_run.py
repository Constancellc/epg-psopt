import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import exampleClass, linModel, calcVar, hcPdfs
from scipy.stats.stats import pearsonr

WD = os.path.dirname(sys.argv[0])

fdr_i = 20
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

print('Load Linear Model feeder:',fdrs[fdr_i],'\n',time.process_time())
LM = linModel(fdr_i,WD)
LM.loadNetModel(LM.netModelNom)

pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom)
Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
Sgm = Mu0/np.sqrt(pdf.pdf['prms'])

# print('Start Svd calcs...',time.process_time())
# LM.makeSvdModel(Sgm,evSvdLim=[0.95,0.98,0.99,0.995,0.999],nMax=3500)

LM.busViolationVar(Sgm)
# LM.makeStdModel()

LM.getCovMat()

# LM.plotNetBuses('logVar')
# plt.imshow(np.log10(abs((1-LM.KtotUcorr))),cmap='viridis',vmin=-2,vmax=0)
# plt.imshow(np.log10(abs((1-LM.KtotUcorr)))<-2)

# vars = LM.KtotUvar.copy()
# varSortN = vars.argsort()[::-1]
# varSortIN = varSortN.argsort()

# LM.makeStdModel(stdLim=[0.99])
# LM.makeCorrModel(stdLim=0.99,corrLim=0.99)
# LM.makeCorrModel()

# LM.corrPlot()

LM.runLinHc(30,pdf.pdf,model='nom')

for i in range(5):
    plt.plot(LM.linHcRsl['Cns_pct'][0])
plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))
plt.show()

# LM.