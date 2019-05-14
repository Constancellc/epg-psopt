import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
# from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn
from linSvdCalcs import *
import linSvdCalcs as lsc

from scipy.stats.stats import pearsonr
from importlib import reload
WD = os.path.dirname(sys.argv[0])

prmI = 0
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy']

pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdfName = 'gammaFlat'; prms=np.array([0.5]); prms=np.array([3.0])
pdfName = 'gammaFrac'
# pdfName = 'gammaXoff'; prms=(np.concatenate((0.33*np.ones((1,19)),np.array([30*np.arange(0.05,1.0,0.05)])),axis=0)).T

# # ================== RUN through each of the networks and see if we can get the analysis running reasonably well
# feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# fdrs_i = 2
# # fdrs_i = 2
# fdrs_is = [6,8,9,19,20,21,17,18,22]
# # fdrs_is = [9,17,19,21,22]

# stdLim = np.array([0.70])
# CorLim = np.array([0.95])

# # stdLim = np.array([0.66,0.7,0.75,0.8])
# # CorLim = np.array([0.9,0.95])

# Errors = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
# Times = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
# Sizes = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )

# i=0;
# for i in range(len(fdrs_is)):
    # j=0
    
    # LM = linModel(fdrs_is[i],WD)
    # pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    # for j in range( len(stdLim) ):
        # k=0
        
        # for k in range( len(CorLim) ):
            # LM.runLinHc(pdf)
            # rslA = LM.linHcRsl
            # rslAt = rslA['runTime']
            
            # Mu,Sgm = pdf.getMuStd(LM,0)
            # LM.busViolationVar(Sgm)
            # LM.makeCorrModel(stdLim=stdLim[j],corrLim=[CorLim[k]])
            # LM.runLinHc(pdf,model='cor')
            # rslB = LM.linHcRsl
            
            # Errors[i,j,k] = np.mean(np.abs(rslB['Vp_pct']-rslA['Vp_pct']))
            # Times[i,j,k] = LM.linHcRsl['runTime']
            # Sizes[i,j,k] = len(LM.NSetCor[0])/len(LM.varKfullU)*100
            # k+=1
            
            # plt.plot(rslA['Vp_pct'])
            # plt.plot(rslB['Vp_pct'])
            # plt.show()
        # j+=1    
    # i+=1



# =============================== Getting the Linear Program version of the code up and running 
def main(linModel=6,tmax=0.1,qmax=0.2):
    reload(lsc)
    # have a go at getting the LP version of the HC calcs working
    LM = lsc.linModel(linModel,WD)
    pdf = lsc.hcPdfs( LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName,nMc=5,prms=np.arange(0.05,1.05,0.05) )
    LM.runLinHc(pdf)
    LM.runLinLp(pdf,tmax=tmax,qmax=qmax)
    # lsc.plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
    
    asd = LM.linHcRsl
    zxc = asd['Lp_pct']
    asd2 = LM.linLpRsl
    zxc2 = asd2['Lp_pct']
    # fig,[ax,ax2] = plt.subplots(2,sharey=True,sharex=True)
    
    # # fig,ax = plt.subplots()
    # i = 0
    # for asd in zxc:
        # pctls = np.percentile(asd,[5,25,50,75,95])
        # rngs = np.percentile(asd,[0,100])
        # lsc.plotBoxWhisk(ax,pdf.pdf['prms'][i],0.01,pctls,bds=rngs)
        # i+=1

    # ax.plot([-0.005,1.005],[1,1],'k--',zorder=20)
    # ax.set_xlim((-0.025,1.025))
    # # ax.set_ylim((0,3.5))
    # ax.grid(True)
    
    
    # i = 0
    # for asd in zxc2:
        # pctls = np.percentile(asd,[5,25,50,75,95])
        # rngs = np.percentile(asd,[0,100])
        # lsc.plotBoxWhisk(ax2,pdf.pdf['prms'][i],0.01,pctls,bds=rngs)
        # i+=1

    # ax2.plot([-0.005,1.005],[1,1],'k--',zorder=20)
    # ax2.set_xlim((-0.025,1.025))
    # # ax2.set_ylim((0,3.5))
    # ax2.grid(True)
    plt.show()

    return LM,pdf,zxc,zxc2

fdr_i = 21
tmax=0.1
def main2(fdr_i=6):
    reload(lsc)
    LM,pdf,zxc,zxc2 = main(fdr_i,tmax=tmax,qmax=0.2)
    LM1 = main(fdr_i,tmax=tmax,qmax=0.0)[0]
    LM2 = main(fdr_i,tmax=0.0,qmax=0.0)[0]

    clrA,clrB,clrC,clrD = cm.matlab(range(4))

    fig,ax = plt.subplots()

    pctls = LM.linHcRsl['kCdf'][[1,4,9,-5,-2]]
    rngs = LM.linHcRsl['kCdf'][[0,-1]]
    lsc.plotBoxWhisk(ax,4,0.33,pctls,bds=rngs,clr=clrA,transpose=True)

    pctls = LM2.linLpRsl['kCdf'][[1,4,9,-5,-2]]
    rngs = LM2.linLpRsl['kCdf'][[0,-1]]
    lsc.plotBoxWhisk(ax,3,0.33,pctls,bds=rngs,clr=clrB,transpose=True)

    pctls = LM1.linLpRsl['kCdf'][[1,4,9,-5,-2]]
    rngs = LM1.linLpRsl['kCdf'][[0,-1]]
    lsc.plotBoxWhisk(ax,2,0.33,pctls,bds=rngs,clr=clrC,transpose=True)

    pctls = LM.linLpRsl['kCdf'][[1,4,9,-5,-2]]
    rngs = LM.linLpRsl['kCdf'][[0,-1]]
    lsc.plotBoxWhisk(ax,1,0.33,pctls,bds=rngs,clr=clrD,transpose=True)

    plt.plot(0,0,color=clrA,label='Lin (base)')
    plt.plot(0,0,color=clrB,label='LP (no taps)')
    plt.plot(0,0,color=clrC,label='LP (taps)')
    plt.plot(0,0,color=clrD,label='LP (taps + Q)')

    plt.legend()

    plt.show()
main2(6)

# LM,pdf,zxc,zxc2 = main(6)

# fig,ax = plt.subplots()

# asd2 = LM.linLpRsl
# zxc2 = asd2['Lp_pct']

# i = 0
# for asd in zxc2:
    # pctls = np.percentile(asd,[5,25,50,75,95])
    # rngs = np.percentile(asd,[0,100])
    # lsc.plotBoxWhisk(ax,pdf.pdf['prms'][i],0.01,pctls,bds=rngs)
    # i+=1

# ax.plot([-0.005,1.005],[1,1],'k--',zorder=20)
# ax.set_xlim((-0.025,1.025))
# ax.set_ylim((0,3.5))
# ax.grid(True)
# plt.show()

# plt.plot(LM.linLpRsl['Vp_pct'])
# plt.plot(LM.linHcRsl['Vp_pct'])
# plt.show()


# # ==================== Attempting to use olkin and pratt. does not seem to work well at all. VVVVVVV
# workflow:
# - load model; run lin hc analysis; plot variance of buses model
# - then go through, calculate covariance matrix and bounds; then find chebyshev-like inequality and plot to find pseudo-optimal
# - validate the pseudo-optimal location with linear HC analysis
# fdr_i = 21
# LM = linModel(fdr_i,WD)

# pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName,prms=prms)
# Mu0,Sgm0 = pdf.getMuStd(prmI=len(prms)-1)
# # olkin and pratt: https://en.wikipedia.org/wiki/Chebyshev%27s_inequality#Multivariate_case
# # use k = 1 on all axis?
# LM.busViolationVar(Sgm0,lim='all',Mu=Mu0)

# Mtot0 = np.concatenate((LM.KtotPu,LM.KfixPu),axis=0)*Sgm0 # 
# mVars = calcVar(Mtot0)
# nIn = np.where(mVars!=1e-100)
# MtotZmZs = Mtot0*np.sqrt(1/mVars)[:,None]
# MtotCov = MtotZmZs[nIn].dot(MtotZmZs[nIn].T) # NB

# k = np.concatenate((LM.svdLim,LM.svdLimDv))*np.sqrt(1/mVars)
# k = k[nIn]

# p = len(MtotCov)
# PI = (MtotCov*(1/k)[:,None])*(1/k)
# t = np.sum(1/(k**2))
# t = sum(np.diag(PI)) # this is identical
# u = sum(sum(PI))

# # u = kSum2 + 2*rhoSum

# # Pr = 1 - ( (1/(n**2))*( ( np.sqrt(u) + (np.sqrt(n-1)*np.sqrt( (n*kSum2) - u ) ) )**2 ) )
# Pr = ( ( np.sqrt(u) + np.sqrt( (p*t - u)*(p-1) ) )**2 )/(p**2)


# # ===============================================
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0][0])
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0]) # in W

# # LM.busViolationVar(Sgm)
# # # LM.busViolationVar(Sgm,lim='VpMvLs') # for prettier pictures
# # LM.busViolationVar(Sgm,lim='VpLvLs',Mu=Mu0)
# # LM.makeVarLinModel()
# # LM.getCovMat()
# # # LM.plotNetBuses('logVar',pltType='mean')
# # LM.plotNetBuses('var',pltType='mean')

# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpMvLs')

# mu_k_first = pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0).argmax()]
# mu_k_first = mu_k_first*1.5
# LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first)
# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpLvLs')
# # LM.busViolationVar(Sgm*mu_k_first,Mu=Mu0*mu_k_first,lim='VpMvLs')

# LM.makeVarLinModel()
# LM.getCovMat()
# LM.plotNetBuses('nStd',pltType='mean')

# # LM.makeCorrModel()
# # LM.corrPlot()
# # LM.plotNetBuses('vLo',pltType='mean')
# # LM.plotNetBuses('vLo',pltType='max')
# # LM.plotNetBuses('vLo',pltType='min')
# LM.plotNetBuses('logVar',pltType='mean')


# print('Minimum HC:',np.nanmin(LM.linHcRsl['hcGenSet']))

# LM.makeVarLinModel(stdLim=[0.90])
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# # plt.plot(prms[:,1],LM.linHcRsl['Cns_pct'][:,0],'--')

# LM.makeCorrModel(stdLim=0.90,corrLim=[0.95])
# LM.runLinHc(pdf,model='cor') # model options: nom / std / cor / mxt ?
# plt.plot(prms[:,1],LM.linHcRsl['Cns_pct'][:,0],'k:')

# plt.legend(('Voltage deviation','Overvoltage (hi ld)','Undervoltage (hi ld)','Overvoltage (lo ld)','Undervoltage (lo ld)'))
# plt.show()


# # ============================ EXAMPLE: plotting the number of standard deviations for a network changing ***vregs*** uniformly
# fdr_i = 20
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())

# LM = linModel(fdr_i,WD,QgenPf=1.0)

# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])

# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0, Sgm0 = pdf.getMuStd(LM=LM) # in W
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?

# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# mu_k_set = np.linspace(0,pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0.).argmax()]*2.0,5)
# Mu_set = np.outer(mu_k_set,Mu0)
# Sgm_set = np.outer(mu_k_set,Sgm0)

# LM.busViolationVar(Sgm_set[2],Mu=Mu_set[2]) # 100% point
# LM.plotNetBuses('nStd',pltType='max')

# nOpts = 21
# opts = np.linspace(0.925,1.05,nOpts)
# for i in range(len(Mu_set)):
    # print(i)
    # N0 = []
    # for opt in opts:
        # LM.updateDcpleModel(LM.regVreg0*opt)
        # LM.busViolationVar(Sgm_set[i],Mu=Mu_set[i])
        # N0.append(np.min(LM.nStdU))
    # plt.plot(opts,N0,'x-')
# print(time.process_time())
# plt.title('Feeder: ' + fdrs[fdr_i]); plt.xlabel('$k_{\mathrm{Vreg}}$'); plt.ylabel('$N_{\sigma}$')
# plt.legend(('0%','50%','100%','150%','200%')); plt.grid(True); plt.ylim((-12,9)); plt.show()

# optVal = 0.96

# LM.updateDcpleModel(LM.regVreg0*optVal)
# LM.busViolationVar(Sgm_set[1],Mu=Mu_set[1])
# LM.plotNetBuses('nStd',pltType='mean')

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# # ==========================================


# # ============================ Choosing which to use (50, 100, 150% of californian solar)
# # NB !!!!!!!! need to make sure the change dMu to None for this to work properly
# fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# # fdr_i_set = [18]
# nMc = int(3e2)
# th_kW_mult = {}
# for fdr_i in fdr_i_set:
    # print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
    # LM = linModel(fdr_i,WD,QgenPf=1.0) # reduce power factor to try and make this work
    # pdfName = 'gammaFrac'; prms=np.array([0.05,1.00])
    # pdf = hcPdfs(LM.feeder,WD=LM.WD,dMu=np.nan,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
    # # LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
    # LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
    # print(LM.linHcRsl['Vp_pct'])
    # if LM.linHcRsl['Vp_pct'][0]>=10.:
        # th_kW_mult[fdrs[fdr_i]] = 0.5
    # elif LM.linHcRsl['Vp_pct'][1]<10.:
        # th_kW_mult[fdrs[fdr_i]] = 1.33
    # else:
        # th_kW_mult[fdrs[fdr_i]] = 1.0
# sn = os.path.join(WD,'hcResults','th_kW_mult.pkl')
# print(th_kW_mult)
# with open(sn,'wb') as handle:
    # pickle.dump(th_kW_mult,handle)

# # now check updated version.
# for fdr_i in fdr_i_set:
    # print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
    # LM = linModel(fdr_i,WD,QgenPf=1.0) # reduce power factor to try and make this work
    # pdfName = 'gammaFrac'; prms=np.array([0.05,1.00])
    # pdf = hcPdfs(LM.feeder,WD=LM.WD,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
    # LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
    # print(LM.linHcRsl['Vp_pct'])
# # ============================



# # ============================ EXAMPLE: looking at gammaFrac to try and improve the HC calculations
# fdr_i = 17
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())

# LM = linModel(fdr_i,WD,QgenPf=1.0) # reduce power factor to try and make this work

# pdfName = 'gammaFrac'; prms=np.arange(0.00,1.05,0.05)
# th_kW_mult = 1.0

# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0, Sgm0 = pdf.getMuStd(LM=LM,prmI=5) # in W

# pdf.pdf['clfnSolar'] = {'k':pdf.pdf['clfnSolar']['k'],'th_kW':th_kW_mult*pdf.pdf['clfnSolar']['th_kW']}

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# prmIs = [0,3,6,9,12]
# Mu_set = np.zeros((len(prmIs),len(Mu0)));   Sgm_set = np.zeros((len(prmIs),len(Mu0)));      i=0
# for prmI in prmIs:
    # Mu_set[i], Sgm_set[i] = pdf.getMuStd(LM=LM,prmI=prmI);     i+=1

# # mult0 = 0.98*np.ones((12))
# # LM.updateDcpleModel(LM.regVreg0*mult0)

# # LM.busViolationVar(Sgm=Sgm_set[4],Mu=Mu_set[4])
# LM.busViolationVar(Sgm_set[4],Mu=Mu_set[4])
# LM.plotNetBuses('nStd',pltType='max',varMax=10)

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# # mult0 = mult0*np.array([1.,1.,1.,0.98,0.98,0.98,1.,1.,1.,1.,1.,1.,])
# LM.updateDcpleModel(LM.regVreg0*mult0)
# LM.busViolationVar(Sgm_set[1],Mu=Mu_set[1])
# LM.plotNetBuses('nStd',pltType='max')

# # LM.DVmax=0.20
# # LM.VpLv=2.0
# nOpts = 11
# opt0 = np.linspace(0.95,1.05,nOpts)
# opts = np.ones((nOpts,len(LM.regVreg0)))
# opts[:,3] = opt0
# opts[:,4] = opt0
# opts[:,5] = opt0
# i=3 # <<<< choose prmI here
# N0 = []
# for opt in opts:
    # LM.updateDcpleModel(LM.regVreg0*mult0*opt)
    # LM.busViolationVar(Sgm_set[i],Mu=Mu_set[i])
    # N0.append(np.min(LM.nStdU))

# plt.plot(opt0,N0,'x-')
# print(time.process_time())
# plt.title('Feeder: ' + fdrs[fdr_i]); plt.xlabel('$k_{\mathrm{Vreg}}$'); plt.ylabel('$N_{\sigma}$')
# plt.legend(('0%','50%','100%','150%','200%')); plt.grid(True); plt.ylim((-12,9)); plt.show()

# LM.updateDcpleModel(LM.regVreg0*mult0)
# LM.busViolationVar(Sgm_set[1],Mu=Mu_set[1])
# LM.plotNetBuses('nStd',pltType='mean')

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# # ==========================================

# # ============================ EXAMPLE: plotting the number of standard deviations for a network changing ***Q*** uniformly
# fdr_i = 20
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD,QgenPf=1.0)

# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0, Sgm0 = pdf.getMuStd(LM=LM) # in W

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# mu_k_set = np.linspace(0,pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0.).argmax()]*2.0,5)

# Mu_set = np.outer(mu_k_set,Mu0)
# Sgm_set = np.outer(mu_k_set,Sgm0)

# LM.busViolationVar(Sgm=Sgm_set[4],Mu=Mu_set[4])
# LM.plotNetBuses('nStd',pltType='mean')

# nOpts = 20
# opts = kq2pf(np.linspace(-0.50,0.5,nOpts))
# for i in range(len(Mu_set)):
    # print(i)
    # N0 = []
    # for opt in opts:
        # LM.QgenPf = opt
        # LM.loadNetModel(LM.netModelNom)
        # LM.updateFxdModel()
        # LM.busViolationVar(Sgm=Sgm_set[i],Mu=Mu_set[i])
        # N0.append(np.min(LM.nStdU))
    # plt.plot(abs(opts),N0); 
# print(time.process_time())
# plt.title('Feeder: ' + fdrs[fdr_i]); plt.xlabel('$k_{\mathrm{Q}}$'); plt.ylabel('$N_{\sigma}$')
# plt.legend(('0%','50%','100%','150%','200%')); plt.grid(True); plt.ylim((-12,9)); plt.show()

# optVal = -0.90

# LM.QgenPf = optVal
# LM.loadNetModel(LM.netModelNom)
# LM.updateFxdModel()
# LM.busViolationVar(Sgm=Sgm_set[4],Mu=Mu_set[4])
# LM.plotNetBuses('nStd',pltType='mean')
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# # ==========================================

# # ============================ EXAMPLE: change Q for epri5
# fdr_i = 17
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD,QgenPf=1.0)

# pdfName = 'gammaWght'; prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0, Sgm0 = pdf.getMuStd(LM=LM) # in W

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# mu_k_set = np.linspace(0,pdf.pdf['mu_k'][(LM.linHcRsl['Vp_pct']>0.).argmax()]*2.0,5)

# Mu_set = np.outer(mu_k_set,Mu0)
# Sgm_set = np.outer(mu_k_set,Sgm0)

# LM.busViolationVar(Sgm=Sgm_set[4],Mu=Mu_set[4])

# LM.plotNetBuses('nStd',pltShow=True)

# LM.QgenPf = -0.95
# LM.loadNetModel(LM.netModelNom)
# LM.updateFxdModel()

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM.busViolationVar(Sgm=Sgm_set[4],Mu=Mu_set[4])
# LM.makeVarLinModel()
# LM.getCovMat()

# LM.plotNetBuses('nStd',pltShow=True)
# plt.show()
# # ====================================



# # ============================ EXAMPLE: change Vreg for K1
# fdr_i = 20
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD)

# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0,Sgm0 = 
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0])
# LM.busViolationVar(Sgm)
# LM.makeVarLinModel()
# LM.getCovMat()
 
# ax = plt.subplot(111)
# print('Time Before',time.process_time())
# LM.plotBranches(ax)
# print('Complete',time.process_time())
# plt.show()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM.updateDcpleModel(LM.regVreg0*0.99)
# LM.busViolationVar(Sgm)
# LM.makeVarLinModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
# plt.show()
# # ====================================

# # ============================ EXAMPLE: change Vreg for K1
# fdr_i = 20
# print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
# LM = linModel(fdr_i,WD)
# pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
# Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)
# Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0])
# LM.busViolationVar(Sgm)
# LM.makeVarLinModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# LM.updateDcpleModel(LM.regVreg0*0.99)
# LM.busViolationVar(Sgm)
# LM.makeVarLinModel()
# LM.getCovMat()

# LM.plotNetBuses('logVar',pltShow=True)
# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)
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
    
    # pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms )
    # Mu0 = pdf.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot)

    # Sgm = Mu0/np.sqrt(pdf.pdf['prms'][0][0])
    # LM.busViolationVar(Sgm)
    # LM.makeVarLinModel()
    # LM.getCovMat()
    
    # LM.plotNetBuses('logVar',pltShow=False,pltType='mean')
    # plt.close()
    # # plt.show()
# # ====================================