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

nMc = 30
nMc = 3000
prmI=0

pdfName = 'gammaWght'; prms=np.array([0.5])
# pdfName = 'gammaFlat'; prms=np.array([0.5])
# pdfName = 'gammaFrac'; prms=np.array([0.25])
# pdfName = 'gammaXoff'; prms=np.array([[0.25,4]])

# pdfName = 'gammaWght'
# pdfName = 'gammaFlat'
# pdfName = 'gammaFrac'
# pdfName = 'gammaXoff'

fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdr_i = 17
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

vars = LM.KtotUvar.copy()
varSortN = vars.argsort()[::-1]
# varSortIN = varSortN.argsort()

LM.makeCorrModel()

pdfTest = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms)
# pdfTest = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName)
pdfMc,pdfMcU = pdfTest.genPdfMcSet(nMc,Mu0,prmI)

print(np.mean(pdfMcU))
print(np.var(pdfMcU))

plt.subplot(121)
hist = plt.hist(pdfMc.flatten(),bins=int(1e2))
plt.ylim((0,1.4*max(hist[0][1:-2])))

plt.subplot(122)
hist = plt.hist(pdfMcU.flatten(),bins=int(1e2))
plt.ylim((0,1.4*max(hist[0][1:-2])))
plt.tight_layout()
plt.show()

# plt.plot(np.mean(pdfMc,axis=1))
# plt.plot(np.std(pdfMc,axis=1))
# plt.show()


# LM.corrPlot()
# LM.plotNetBuses('logVar')

# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# for i in range(5):
    # plt.plot(LM.linHcRsl['Cns_pct'][0])
# plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))
# # plt.show()

# LM.makeStdModel(stdLim=[0.90])
# LM.runLinHc(nMc,pdf.pdf,model='std') # model options: nom / std / cor / mxt ?

# for i in range(5):
    # plt.plot(LM.linHcRsl['Cns_pct'][0],'--')
# plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))

# LM.makeCorrModel(stdLim=0.90,corrLim=[0.95])
# LM.runLinHc(nMc,pdf.pdf,model='cor') # model options: nom / std / cor / mxt ?

# for i in range(5):
    # plt.plot(LM.linHcRsl['Cns_pct'][0],'k:')
# plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))
# plt.show()

# for fdr_i in fdr_i_set:
    # print('\n==== Start Feeder:',fdrs[fdr_i])
    # LM = linModel(fdr_i,WD)
    # LM.loadNetModel(LM.netModelNom)
    
    # pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom)
    # Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
    # Sgm = Mu0/np.sqrt(pdf.pdf['prms'])
    # LM.busViolationVar(Sgm)
    
    # LM.getCovMat()
    # LM.makeCorrModel(stdLim=0.90)    
