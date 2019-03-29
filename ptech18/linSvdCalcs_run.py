import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn
from scipy.stats.stats import pearsonr


WD = os.path.dirname(sys.argv[0])

fn0 = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190325\\"

nMc = 30
# nMc = 1000
prmI = 0

pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdfName = 'gammaFlat'; prms=np.array([0.5]); prms=np.array([3.0])
# pdfName = 'gammaFrac'; prms=np.arange(0.05,1.05,0.05)
# pdfName = 'gammaXoff'; prms=(np.concatenate((0.33*np.ones((1,19)),np.array([30*np.arange(0.05,1.0,0.05)])),axis=0)).T

fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdr_i = 18
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD,QgenPf=1.0)
# LM.loadNetModel(LM.netModelNom)

# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot) # in W

# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# # Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0][0])
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0]) # in W

# # LM.busViolationVar(Sgm)
# # # LM.busViolationVar(Sgm,lim='VpMvLs') # for prettier pictures
# # LM.busViolationVar(Sgm,lim='VpLvLs',Mu=Mu0)
# # LM.makeStdModel()
# # LM.getCovMat()
# # # LM.plotNetBuses('logVar',pltType='mean')
# # LM.plotNetBuses('var',pltType='mean')

# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpMvLs')

# mu_k_first = pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0).argmax()]
# mu_k_first = mu_k_first*1.5
# LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first)
# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpLvLs')
# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpMvLs')

# LM.makeStdModel()
# LM.getCovMat()
# LM.plotNetBuses('nStd',pltType='mean')

# # LM.makeCorrModel()
# # LM.corrPlot()
# # LM.plotNetBuses('vLo',pltType='mean')
# # LM.plotNetBuses('vLo',pltType='max')
# # LM.plotNetBuses('vLo',pltType='min')
# LM.plotNetBuses('logVar',pltType='mean')


# print('Minimum HC:',np.nanmin(LM.linHcRsl['hcGenSet']))

# LM.makeStdModel(stdLim=[0.90])
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# # plt.plot(prms[:,1],LM.linHcRsl['Cns_pct'][:,0],'--')

# LM.makeCorrModel(stdLim=0.90,corrLim=[0.95])
# LM.runLinHc(nMc,pdf.pdf,model='cor') # model options: nom / std / cor / mxt ?
# plt.plot(prms[:,1],LM.linHcRsl['Cns_pct'][:,0],'k:')

# plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))
# plt.show()

# ============================ EXAMPLE: plotting the number of standard deviations for a network
fdr_i = 9
print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
LM = linModel(fdr_i,WD,QgenPf=1.0)
LM.loadNetModel(LM.netModelNom)
pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot) # in W

LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0][0])
Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0]) # in W

mu_k_first = pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0).argmax()]
mu_k_first = mu_k_first*2.0 # 2x the power at which the first violation occurs

LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first)
LM.makeStdModel()
LM.plotNetBuses('nStd',pltType='mean')
# ====================================



# # ============================ EXAMPLE: change Q for epri5
# fdr_i = 17
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD,QgenPf=1.0)
# LM.loadNetModel(LM.netModelNom)

# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0])

# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM = linModel(fdr_i,WD,QgenPf=-0.90)
# LM.loadNetModel(LM.netModelNom)
# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# plt.show()
# # ====================================



# # ============================ EXAMPLE: change Vreg for K1
# fdr_i = 9
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD)
# LM.loadNetModel(LM.netModelNom)

# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0])
# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# ax = plt.subplot(111)
# print('Time Before',time.process_time())
# LM.plotBranches(ax)
# print('Complete',time.process_time())
# plt.show()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM.updateDcpleModel(LM.regVreg0*0.99)
# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# plt.show()
# # ====================================

# # ============================ EXAMPLE: change Vreg for K1
# fdr_i = 20
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD)
# LM.loadNetModel(LM.netModelNom)
# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0])
# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM.updateDcpleModel(LM.regVreg0*0.99)
# LM.busViolationVar(Sgm)
# LM.makeStdModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(nMc,pdf.pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# plt.show()
# # ====================================


# # ==================================== PLOT all of the nice model variances
# pdfName = 'gammaXoff'; prms=(np.concatenate((0.33*np.ones((1,19)),np.array([30*np.arange(0.05,1.0,0.05)])),axis=0)).T
# fdr_i_set = [5,6,8,9,0,17,18,19,20,21,22]
# # fdr_i_set = [5,6,8]
# # fdr_i_set = [22]
# for fdr_i in fdr_i_set:
    # print('\n==== Start Feeder:',fdrs[fdr_i])
    # LM = linModel(fdr_i,WD)
    # LM.loadNetModel(LM.netModelNom)
    
    # pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
    # Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)

    # Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0][0])
    # LM.busViolationVar(Sgm)
    # LM.makeStdModel()
    # LM.getCovMat()
    
    # LM.plotNetBuses('logVar',pltShow=False,pltType='mean')
    # plt.savefig(fn0+'logVar_'+fdrs[fdr_i]+'_new.png')
    # plt.close()
    # # plt.show()
# # ====================================