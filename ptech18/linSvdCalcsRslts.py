import pickle, os, sys, win32com.client, time, scipy.stats, getpass
import numpy as np
from dss_python_funcs import *
import matplotlib.pyplot as plt
from matplotlib import cm
import dss_stats_funcs as dsf
from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn, plotBoxWhisk
from importlib import reload
import linSvdCalcs as lsc
plt.style.use('tidySettings')
from scipy.stats import pearsonr

WD = os.path.dirname(sys.argv[0])
SD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']

pltShow = 1
# pltSave = 1
# pltCc = 1
# f_nStdBefore = 1
# f_nStdAfter = 1
# f_nStdVreg = 1
# f_nStdVregVal = 1
# f_corVars = 1
# f_hcParamSlctnCaseStudy = 1
# f_limitSensitivityV = 1
# f_limitSensitivity = 1
# f_limitSensitivityIdv = 1
# f_corVarScanCalc = 1
# f_corVarScan = 1
# f_corVarCTs = 1
# f_varCheck = 1

# # calculating setpoints for linHcCalcs:
# f_nStdVreg_8500 = 1
# f_nStdVreg_epriM1 = 1
# f_nStdVreg_epri24 = 1
# f_nStdVreg_epriJ1 = 1
# f_nStdVreg_epriK1 = 1


def main(fdr_i=6):
    reload(lsc)
    LM = lsc.linModel(fdr_i,WD)
    pdf = lsc.hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac')
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=44) # in W
    LM.busViolationVar(Sgm,calcSrsVals=True)
    return LM, pdf

# LM,pdf = main()

figsze0 = (5.2,3.0)
figsze1 = (5.2,2.7)
# ============================== 1. plotting EU LV and EPRI K1 for CC 
if 'pltCc' in locals():
    fdr_i_set = [0,20]
    for fdr_i in fdr_i_set:
        LM = linModel(fdr_i,WD,QgenPf=1.0)
        LM.loadNetModel()
        ax = LM.plotNetwork(pltShow=False)
        xlm = ax.get_xlim() 
        ylm = ax.get_ylim()
        dx = xlm[1] - xlm[0]; dy = ylm[1] - ylm[0] # these seem to be in feet for k1
        if fdr_i==0:
            # (2637175.474787638, 2653020.026654688) (2637175.474787638, 2653020.026654688)
            dist = 10
            x0 = xlm[0] + 0.8*dx
            y0 = ylm[0] + 0.05*dy
            ax.plot([x0,x0+dist],[y0,y0],'k-')
            ax.plot([x0,x0],[y0-0.005*dy,y0+0.005*dy],'k-')
            ax.plot([x0+dist,x0+dist],[y0-0.005*dy,y0+0.005*dy],'k-')
            ax.annotate('10 metres',(x0+(dist/2),y0+dy*0.02),ha='center')
            if 'pltSave' in locals():
                plt.savefig(WD+'\\hcResults\\eulvNetwork.pdf',bbox_inches='tight', pad_inches=0)
        if fdr_i==20:
            # (390860.71323843475, 391030.5357615654) (390860.71323843475, 391030.5357615654)
            dist = 5280
            x0 = xlm[0] + 0.6*dx
            y0 = ylm[0] + 0.05*dy
            ax.plot([x0,x0+dist],[y0,y0],'k-')
            ax.plot([x0,x0],[y0-0.005*dy,y0+0.005*dy],'k-')
            ax.plot([x0+dist,x0+dist],[y0-0.005*dy,y0+0.005*dy],'k-')
            ax.annotate('1 mile',(x0+(dist/2),y0+dy*0.02),ha='center')
            if 'pltSave' in locals():
                plt.savefig(WD+'\\hcResults\\epriK1Network.pdf',bbox_inches='tight', pad_inches=0)
        if 'pltShow' in locals():
            plt.show()

# ============================ FIGURE 5: plotting the number of standard deviations for a network changing ***vregs*** uniformly
if ('f_nStdBefore' in locals()) or ('f_nStdAfter' in locals()):
    fdr_i = 22
    pdfName = 'gammaFrac'
    LM = linModel(fdr_i,WD,QgenPf=1.0)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=44) # in W
    
    LM.busViolationVar(Sgm,Mu=Mu) # 100% point
    LM.legLoc = 'resPlot24'
    LM.plotNetBuses('nStd',pltType='max',minMax=[-3.,6.],cmap=plt.cm.inferno,pltShow=False)
    LM.plotSub(LM.currentAx,pltSrcReg=False)
    if 'pltSave' in locals():
        plt.savefig(SD+'nStdBefore_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdBefore_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()

    # optVal = 0.98
    # LM.updateDcpleModel(LM.regVreg0*optVal)
    LM = linModel(fdr_i,WD,QgenPf=-0.98)
    pdf.getMuStd(LM=LM,prmI=44) # in W. Required to also update LM.
    LM.busViolationVar(Sgm,Mu=Mu)
    LM.legLoc = 'resPlot24'
    LM.plotNetBuses('nStd',pltType='max',minMax=[-3.,6.],cmap=cm.inferno,pltShow=False)
    LM.plotSub(LM.currentAx,pltSrcReg=False)
    if 'pltSave' in locals():
        plt.savefig(SD+'nStdAfter_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdAfter_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

# ==================== Making a nice plot of COR, VAR for a network.
if 'f_corVars' in locals():
    fdr_i = 17
    pdfName = 'gammaFrac'
    LM = linModel(fdr_i,WD,QgenPf=1.0)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=0) # in W
    
    # LM.CovSqrt = (LM.CovSqrt[0],np.array([0]))
    # LM.CovSqrt = (np.array([1]),np.array([0]))

    LM.busViolationVar(Sgm)
    # LM.getCovMat(getFixCov=False,getTotCov=False,getFullCov=True)
    LM.getCovMat(getFixCov=True)
    # LM.makeCorrModel()
    # LM.corrPlot()

    vars = LM.varKtotU.copy()
    varSortN_new = vars.argsort()[::-1]
    corrLogAbs = np.log10(abs((1-LM.KtotUcorr)) + np.diag(np.ones(len(LM.KtotPu))) +1e-14 )

    # vars = LM.varKfullU.copy()
    # varSortN_new = vars.argsort()[::-1]
    # corrLogAbs = np.log10(abs((1-LM.KfullUcorr)) + np.diag(np.ones(len(LM.KfullUcorr))) +1e-14 )
    corrLogAbs = corrLogAbs[varSortN_new][:,varSortN_new]

    # fig,(ax1,ax0) = plt.subplots(figsize=(5.9,3.8),nrows=1,ncols=2, gridspec_kw = {'width_ratios':[1,1.8]})
    # ax0.spy(corrLogAbs<-2.0,color=cm.Blues(0.9),markersize=1,marker='.',zorder=4,label='99%') # 99%
    # ax0.spy(corrLogAbs<-1.7,color=cm.Blues(0.7),markersize=1,marker='.',zorder=3,label='98%') # 98%
    # ax0.spy(corrLogAbs<-1.3,color=cm.Blues(0.5),markersize=1,marker='.',zorder=2,label='95%') # 95%
    # ax0.spy(corrLogAbs<-1.0,color=cm.Blues(0.3),markersize=1,marker='.',zorder=1,label='90%') # 90%

    # legend = ax0.legend(loc='upper right',borderpad=0.55,fontsize='small',handletextpad=0.28,markerscale=10)
    # ax0.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    # ax0.set_yticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    # ax0.set_xticklabels([])
    # ax0.set_yticklabels([])
    # ax0.set_xlabel('Bus Index $i$')
    # ax0.set_ylabel('Bus Index $j$')

    # ax1.plot(vars[varSortN_new]/vars[varSortN_new][0])
    # ax1.set_xlim((-50,len(corrLogAbs+50)))
    # ax1.set_xlabel('Bus Index $i$')
    # ax1.set_ylabel('Var(Bus $i$), normalised')
    # ax1.set_yscale('log')
    # ax1.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    # ax1.set_xticklabels([])
    # ax1.set_yticks(ticks=[1,0.01,0.0001])

    # ax0.set_title('Correlation',pad=-5.5)
    # ax1.set_title('Variance',pad=5.5)
    # plt.tight_layout()
    # if 'pltSave' in locals():
        # plt.savefig(SD+'corVars_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0.01)
        # plt.savefig(SD+'corVars_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    # if 'pltShow' in locals():
        # plt.show()
    
    fig,ax0 = plt.subplots(figsize=(3.8,3.8))
    ax0.spy(corrLogAbs<-2.0,color=cm.Blues(0.9),markersize=1,marker='.',zorder=4,label='99%') # 99%
    ax0.spy(corrLogAbs<-1.7,color=cm.Blues(0.7),markersize=1,marker='.',zorder=3,label='98%') # 98%
    ax0.spy(corrLogAbs<-1.3,color=cm.Blues(0.5),markersize=1,marker='.',zorder=2,label='95%') # 95%
    ax0.spy(corrLogAbs<-1.0,color=cm.Blues(0.3),markersize=1,marker='.',zorder=1,label='90%') # 90%

    legend = ax0.legend(loc='upper right',borderpad=0.55,fontsize='small',handletextpad=0.28,markerscale=10)
    ax0.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax0.set_yticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_xlabel('Bus Index $i$')
    ax0.set_ylabel('Bus Index $j$')

    plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'corVarsB_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0.01)
        plt.savefig(SD+'corVarsB_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()
        
    fig,ax1 = plt.subplots(figsize=(2.1,3.8))
    ax1.plot(vars[varSortN_new]/vars[varSortN_new][0])
    ax1.set_xlim((-50,len(corrLogAbs+50)))
    ax1.set_xlabel('Bus Index $i$')
    ax1.set_ylabel('Var(Bus $i$), normalised')
    ax1.set_yscale('log')
    ax1.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax1.set_xticklabels([])
    ax1.set_yticks(ticks=[1,0.01,0.0001])
    
    plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'corVarsA_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0.01)
        plt.savefig(SD+'corVarsA_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()


# ============================ FIGURE 6: plotting our nice error
if ('f_nStdVreg' in locals()) or ('f_nStdVregVal' in locals()):
    fdr_i = 22
    pdfName = 'gammaFrac'
    LM = linModel(fdr_i,WD,QgenPf=1.0)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    # Mu, Sgm = pdf.getMuStd(LM=LM,prmI=-1) # in W. <---- UPDATED parameter here to 100% point
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=-5) # in W. <---- UPDATED parameter here to 100% point

    xTickMatch = np.linspace(0.98,1.04,7)
    xTickMatchStr = ["%.2f" % x for x in xTickMatch]
    xlims = (0.975,1.045)

    Q_set = [1.0,-0.995,-0.98]
    aFro = 1e-5
    
    nOpts = 61
    
    t = time.time()
    opts = np.linspace(0.955,1.025,nOpts)
    N0 = np.zeros((len(Q_set),len(opts)))
    for i in range(len(Q_set)):
        # LM = linModel(fdr_i,WD,QgenPf=Q_set[i]) # NB: here, the three lines below DO SEEM to do the right thing and the same as this (!!!)
        LM.QgenPf = Q_set[i]
        LM.loadNetModel(LM.netModelNom)
        LM.updateFxdModel()
        LM.updateDcpleModel(LM.regVreg0)
        LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
        j=0
        for opt in opts:
            LM.updateDcpleModel(LM.regVreg0*opt)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
        print(time.time()-t)

    fig,ax = plt.subplots(figsize=figsze1)
    # ax.plot(np.outer(opts*LM.regVreg0/(166*120),[1]*len(N0)),N0.T,'.-',markersize=4)
    ax.plot(opts*LM.regVreg0/(166*120),N0[0],'.-',markersize=4,zorder=10)
    ax.plot(opts*LM.regVreg0/(166*120),N0[1],'.-',markersize=4,zorder=5)
    ax.plot(opts*LM.regVreg0/(166*120),N0[2],'.-',markersize=4,zorder=0)

    # idxMax = np.argmax(N0,axis=1)
    # maxVals = N0[[0,1,2],idxMax]
    # maxArgs = opts[idxMax]*LM.regVreg0/(166*120)
    # ax.plot(maxArgs[0],maxVals[0],'k',marker='o',linestyle='',markerfacecolor='w',zorder=8)
    # ax.plot(maxArgs[1],maxVals[1],'k',marker='o',linestyle='',markerfacecolor='w',zorder=3)
    # ax.plot(maxArgs[2],maxVals[2],'k',marker='o',linestyle='',markerfacecolor='w',zorder=-2)

    ax.set_xlabel('Regulator setpoint, pu')
    # ax.set_ylabel('Min. no. of std. deviations, $\hat{\sigma}_{\mathrm{min}}$')
    ax.set_ylabel('Min. no. of std. deviations, $\\bar{\sigma}_{\mathrm{min}}$')
    ax.legend(('1.0','0.995','0.98'),title='PF (lag.)',loc='upper right')
    plt.xlim(xlims); plt.ylim((-10.5,8))
    plt.xticks(xTickMatch,xTickMatchStr)
    plt.grid(True); 
    plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'nStdVreg_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdVreg_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)

    if 'pltShow' in locals():
        plt.show()


if 'f_nStdVregVal' in locals():
    optVals = np.linspace(0.95,1.025,16)
    t = time.time()
    i=0
    rsltsA = {}; rsltsB = {}
    LM.QgenPf = Q_set[0]
    LM.loadNetModel(LM.netModelNom)
    LM.updateFxdModel()
    for optVal in optVals:
        print(i)
        LM.updateDcpleModel(LM.regVreg0*optVal)
        LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
        rsltsA[i] = LM.linHcRsl
        i+=1
    i=0
    LM.QgenPf = Q_set[1]
    LM.loadNetModel(LM.netModelNom)
    LM.updateFxdModel()
    for optVal in optVals:
        print(i)
        LM.updateDcpleModel(LM.regVreg0*optVal)
        LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
        rsltsB[i] = LM.linHcRsl
        i+=1
    print(time.time() - t)
    
    fig = plt.figure(figsize=figsze0)
    ax = fig.add_subplot(111)
    X = np.arange(len(optVals))
    for x in X:
        ax = plotBoxWhisk(ax,x-0.15,0.12,rsltsA[x]['kCdf'][0::5],clr=cm.tab10(0))
        ax = plotBoxWhisk(ax,x+0.15,0.12,rsltsB[x]['kCdf'][0::5],clr=cm.tab10(1))

    plt.plot(0,0,color=cm.tab10(0),label='1.00')
    plt.plot(0,0,color=cm.tab10(1),label='0.98')
    # plt.xticks(X,optVals)

    newStr = []
    for mStr in xTickMatchStr:
        newStr.append(mStr)
        newStr.append('')

    plt.xticks(np.arange(len(xTickMatch)*2),newStr)
    plt.xlabel('Regulator setpoint, pu')
    plt.ylabel('Fraction of loads with PV, %')
    plt.legend(title='PF (lag)',loc='lower center')
    plt.ylim((-1,101))
    plt.xlim((-0.8,2*len(xTickMatch)-0.5))
    plt.grid(True)
    plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'nStdVregVal_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdVregVal_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)

    if 'pltShow' in locals():
        plt.show()

# ============================ Finding regulator settings for linHcCalcs, 8500 node
if 'f_nStdVreg_8500' in locals():
    fdr_i = 9
    pdfName = 'gammaFrac'
    LM = linModel(fdr_i,WD,QgenPf=-0.95)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=30) # in W. <---- UPDATED parameter here to 100% point

    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    # LM.plotNetBuses('nStd',pltType='max',minMax=[-3.,6.],cmap=cm.inferno,pltShow=True)
    aFro = 1e-2
    
    nOpts = 5
    t = time.time()
    opts = np.linspace(0.99,1.01,nOpts)
    N0 = np.zeros((len(LM.regVreg0),len(opts)))
    LM.updateDcpleModel(LM.regVreg0)
    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    optMult0 = 0.995**np.array([3,1,2,2,1,0,1,0,0,3,3,1])
    # optMult0 = np.ones((len(LM.regVreg0)))
    # optMult0[0] = 0.995**3
    # optMult0[1] = 0.995
    # optMult0[2] = 0.995**2
    # optMult0[3] = 0.995**2
    # optMult0[4] = 0.995
    # optMult0[6] = 0.995
    # optMult0[9] = 0.995**3.0
    # optMult0[10] = 0.995**3.0
    # optMult0[11] = 0.995
    for i in range(len(LM.regVreg0)):
        optMult = optMult0.copy()
        j=0
        for opt in opts:
            optMult[i] = opt*optMult0[i]
            LM.updateDcpleModel(LM.regVreg0*optMult)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
        print(time.time()-t)
    print(np.diff(N0)[:,1])
    fig,ax = plt.subplots()
    ax.set_prop_cycle(color=cm.tab20(np.arange(20)))
    plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Preconditioning metric, $\lambda$')
    plt.legend(('0','1','2','3','4','5','6','7','8','9','10','11'),title='PF (lagging)');
    # plt.ylim((-4,4))
    plt.tight_layout(); plt.show()
    
    # LM.updateDcpleModel(LM.regVreg0*optMult0)
    # LM.runLinHc(pdf)
    # qwe = LM.linHcRsl
    
    # LM = linModel(fdr_i,WD,QgenPf=1.00)
    # LM.runLinHc(pdf)
    # qwe2 = LM.linHcRsl

    # fig,ax = plt.subplots()
    # ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe['Cns_pct'],ax=ax,pltShow=False)
    # plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe2['Cns_pct'],ax=ax,pltShow=False,lineStyle='--')
    # plt.show()

# ============================ TWEAKING regulator settings for EPRI M1
if 'f_nStdVreg_epriM1' in locals():
    fdr_i = 21
    pdfName = 'gammaFrac'
    # LM = linModel(fdr_i,WD,QgenPf=-0.95)
    LM = linModel(fdr_i,WD,QgenPf=1.00)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=15) # in W. <---- UPDATED parameter here to 100% point

    # LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    aFro = 1e-6
    
    nOpts = 30
    t = time.time()
    opts = np.linspace(0.95,1.00,nOpts)
    N0 = np.zeros((len(LM.regVreg0),len(opts)))
    LM.updateDcpleModel(LM.regVreg0)
    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    optMult0 = np.ones((len(LM.regVreg0)))
    for i in range(len(LM.regVreg0)):
        optMult = optMult0.copy()
        j=0
        for opt in opts:
            optMult[i] = opt*optMult0[i]
            LM.updateDcpleModel(LM.regVreg0*optMult)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
    print(N0)
    print(np.diff(N0))
    plt.figure(figsize=figsze0)
    plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Preconditioning metric, $\lambda$')
    plt.legend(('0','1','2','3','4','5','6','7','8','9','10','11'),title='PF (lagging)');
    plt.ylim((-8,8))
    plt.tight_layout()
    plt.show()

    chosenOpt = 0.980
    LM.updateDcpleModel(LM.regVreg0*chosenOpt)
    LM.runLinHc(pdf)
    qwe = LM.linHcRsl

    # LM.QgenPf = 1.0
    # LM.loadNetModel(LM.netModelNom)
    # LM.updateFxdModel()
    # LM.updateDcpleModel(LM.regVreg0)
    LM = linModel(fdr_i,WD,QgenPf=1.00)
    LM.runLinHc(pdf)
    qwe2 = LM.linHcRsl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe['Cns_pct'],ax=ax,pltShow=False)
    plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe2['Cns_pct'],ax=ax,pltShow=False,lineStyle='--')
    plt.show()
    
    
# ============================ TWEAKING regulator settings for EPRI M1
if 'f_nStdVreg_epriK1' in locals():
    fdr_i = 20
    pdfName = 'gammaFrac'
    # LM = linModel(fdr_i,WD,QgenPf=-0.95) # V0
    LM = linModel(fdr_i,WD,QgenPf=1.00) # V1
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=15) # in W. <---- UPDATED parameter here to 100% point

    # LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    aFro = 1e-6
    
    nOpts = 30
    t = time.time()
    opts = np.linspace(0.95,1.02,nOpts)
    N0 = np.zeros((len(LM.regVreg0),len(opts)))
    LM.updateDcpleModel(LM.regVreg0)
    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    optMult0 = np.ones((len(LM.regVreg0)))
    for i in range(len(LM.regVreg0)):
        optMult = optMult0.copy()
        j=0
        for opt in opts:
            optMult[i] = opt*optMult0[i]
            LM.updateDcpleModel(LM.regVreg0*optMult)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
    print(N0)
    print(np.diff(N0))
    plt.figure(figsize=figsze0)
    plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Preconditioning metric, $\lambda$')
    plt.legend(('0','1','2','3','4','5','6','7','8','9','10','11'),title='PF (lagging)');
    plt.ylim((-8,8))
    plt.tight_layout()
    plt.show()

    chosenOpt = 0.984
    LM.updateDcpleModel(LM.regVreg0*chosenOpt)
    LM.runLinHc(pdf)
    qwe = LM.linHcRsl

    LM.QgenPf = 1.0
    LM.loadNetModel(LM.netModelNom)
    LM.updateFxdModel()
    LM.updateDcpleModel(LM.regVreg0)
    LM.runLinHc(pdf)
    qwe2 = LM.linHcRsl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe['Cns_pct'],ax=ax,pltShow=False)
    plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe2['Cns_pct'],ax=ax,pltShow=False,lineStyle='--')
    plt.show()


# ============================ FIGURE 6: plotting our nice error AGAIN for the 8500 node network
if 'f_nStdVreg_epri24' in locals():
    fdr_i = 22
    pdfName = 'gammaFrac'
    
    LM = linModel(fdr_i,WD,QgenPf=1.00)
    # LM = linModel(fdr_i,WD,QgenPf=-0.98)
    # LM = linModel(fdr_i,WD,QgenPf=-0.95)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=15) # in W. <---- UPDATED parameter here to 100% point

    # LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    aFro = 1e-6
    nOpts = 30
    t = time.time()
    opts = np.linspace(0.95,1.00,nOpts)
    N0 = np.zeros((len(LM.regVreg0),len(opts)))
    LM.updateDcpleModel(LM.regVreg0)
    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    optMult0 = np.ones((len(LM.regVreg0)))
    for i in range(len(LM.regVreg0)):
        optMult = optMult0.copy()
        j=0
        for opt in opts:
            optMult[i] = opt*optMult0[i]
            LM.updateDcpleModel(LM.regVreg0*optMult)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
    print(N0)
    print(np.diff(N0))
    plt.figure(figsize=figsze0)
    plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Preconditioning metric, $\lambda$')
    plt.legend(('0','1','2','3','4','5','6','7','8','9','10','11'),title='PF (lagging)');
    plt.ylim((-8,8))
    plt.tight_layout()

    plt.show()

    chosenOpt = 0.981
    LM.updateDcpleModel(LM.regVreg0*chosenOpt)
    LM.runLinHc(pdf)
    qwe = LM.linHcRsl

    LM.QgenPf = 1.0
    LM.loadNetModel(LM.netModelNom)
    LM.updateFxdModel()
    LM.updateDcpleModel(LM.regVreg0)
    qwe2 = LM.runLinHc(pdf)
    qwe2 = LM.linHcRsl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe['Cns_pct'],ax=ax,pltShow=False)
    plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],qwe2['Cns_pct'],ax=ax,pltShow=False,lineStyle='--')
    plt.show()


# ============================ FIGURE 6: plotting our nice error AGAIN, J1 Network
if 'f_nStdVreg_epriJ1' in locals():
    fdr_i = 19
    pdfName = 'gammaFrac'
    LM = linModel(fdr_i,WD,QgenPf=1.00)
    # LM = linModel(fdr_i,WD,QgenPf=-0.98)
    # LM = linModel(fdr_i,WD,QgenPf=-0.95)
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    Mu, Sgm = pdf.getMuStd(LM=LM,prmI=40) # in W. <---- UPDATED parameter here to 100% point

    # LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    aFro = 1e-6
    
    # nOpts = 2
    nOpts = 3
    t = time.time()
    opts = np.linspace(0.995,1.005,nOpts)
    N0 = np.zeros((len(LM.regVreg0),len(opts)))
    LM.updateDcpleModel(LM.regVreg0)
    LM.busViolationVar(Sgm,Mu=Mu,calcSrsVals=True)
    optMult0 = np.ones((len(LM.regVreg0)))
    optMult0 = 0.995**np.array([5,4,0,-1,2,4,0,5.5,4])
    # optMult0[0] = 0.995**5
    # optMult0[1] = 0.995**4
    # optMult0[3] = 0.995
    # optMult0[4] = 0.995**2
    # optMult0[5] = 0.995**4
    # optMult0[7] = 0.995**5.5
    # optMult0[8] = 0.995**4
    for i in range(len(LM.regVreg0)):
        optMult = optMult0.copy()
        j=0
        for opt in opts:
            optMult[i] = opt*optMult0[i]
            LM.updateDcpleModel(LM.regVreg0*optMult)
            Kfro,Knstd = LM.updateNormCalc(Mu=Mu,inclSmallMu=True)
            N0[i,j] = Knstd - aFro*Kfro
            j+=1
    print(N0)
    print(np.diff(N0))
    plt.figure(figsize=figsze0)
    plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Preconditioning metric, $\lambda$')
    plt.legend(('0','1','2','3','4','5','6','7','8','9','10','11'),title='PF (lagging)');
    # plt.ylim((-8,8))
    plt.tight_layout()
    plt.show()

if 'f_hcParamSlctnCaseStudy' in locals():
    fdr_i = 19

    LM = linModel(fdr_i,WD,QgenPf=-0.95)
    # LM = linModel(fdr_i,WD,QgenPf=-0.98)
    optMult0 = np.ones((len(LM.regVreg0)))
    # optMult0 = 0.995**np.array([5,4,0,-1,2,4,0,5.5,4])
    LM.updateDcpleModel(LM.regVreg0*optMult0)
    
    pdfName = 'gammaFrac'
    pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
    
    LM.runLinHc(pdf)
    rsltAft = LM.linHcRsl

    LM = linModel(fdr_i,WD,QgenPf=1.00)
    # LM.QgenPf = 1.0 # NB: these do not seem to be doing well ATM !!!!
    # LM.loadNetModel(LM.netModelNom)
    # LM.updateFxdModel()
    # LM.updateDcpleModel(LM.regVreg0)
    LM.runLinHc(pdf)
    rsltBef = LM.linHcRsl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],rsltAft['Cns_pct'],ax=ax,pltShow=False)
    plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],rsltBef['Cns_pct'],ax=ax,pltShow=False,lineStyle='--')
    plt.show()

    rslts = {'pdf':pdf,'rsltBef':rsltBef,'rsltAft':rsltAft}
    SN = os.path.join(WD,'hcResults','hcParamSlctnCaseStudy.pkl')
    with open(SN,'wb') as file:
        pickle.dump(rslts,file)
        

if 'f_limitSensitivityV' in locals():
    # NB this hasn't been used/tested fully
    I = [9,19,29,39,49] # 20-80%
    VpSet = [1.055-0.002,1.055+0.002]
    fdr_i_set = [6,8,9,17,18,19,20,21,22]
    fdr_i_set=[6,8,20]
    fdr_i_set=[17]
    mnsV = [];    std = [];    rslts = [];    rsltsDiff = []
    for fdr_i in fdr_i_set:
        LM = linModel(fdr_i,WD)
        pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac' )
        rslt = np.empty((len(I),0))
        for Vp in VpSet:
            LM.VpLv=Vp
            LM.VpMv=Vp
            LM.runLinHc(pdf,fast=True)
            rslt = np.hstack((rslt,LM.linHcRsl['Vp_pct'][I]))
        
        rslts.append(rslt)
        rsltsDiff.append(np.diff(rslt))
        mnsV.append(np.mean(-np.diff(rslt)))
        std.append(np.std(np.diff(rslt)))

    plt.plot(VpSet,rslt.T,'-'); plt.xlabel('Voltage upper bound (%)'); plt.ylabel('Prob. overvoltage');
    plt.legend(('20','40','60','80','100'),title='% PV'); plt.title(LM.feeder); plt.show()


if 'f_limitSensitivity' in locals():
    I = range(50) # 20-80%
    ddMu = 0.05
    dMuSet = [1-ddMu,1+ddMu]
    # nMc = 300
    nMc = 100
    # nMc = 30

    fdr_i_set = [6,8,9,17,18,19,20,21,22]
    # fdr_i_set = [6,8,20]
    # fdr_i_set = [6]
    
    mnsS = []
    std = []
    rslts = []
    rsltsDiff = []
    for fdr_i in fdr_i_set:
        LM = linModel(fdr_i,WD)
        
        pdf0 = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac',nMc=nMc )
        
        Mu,Sgm = pdf0.getMuStd(LM,0)
        LM.busViolationVar(Sgm)
        LM.makeCorrModel()
        
        SD = os.path.join(WD,'hcResults','th_kW_mult.pkl')
        with open(SD,'rb') as handle:
            circuitK = pickle.load(handle)
        
        dMu0 = circuitK[LM.feeder]
        rslt = np.empty((len(I),0))
        for dMu in dMuSet:
            print(dMu)
            pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac',dMu=dMu*dMu0,nMc=nMc )
            # LM.runLinHc(pdf,fast=True)
            LM.runLinHc(pdf,model='cor',fast=False)
            rslt = np.hstack((rslt,LM.linHcRsl['Vp_pct'][I]))
        rslts.append(rslt)
        rsltsDiff.append(np.diff(rslt))
        mnsS.append(np.mean(np.diff(rslt)))
        std.append(np.std(np.diff(rslt)))

        kHeur = 1 - 0.003*mnsS[-1]
        
        pdf = hcPdfs( LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac',dMu=kHeur*circuitK[LM.feeder],nMc=nMc )
        LM.runLinHc(pdf,fast=True)
        rslt = np.hstack((rslt,LM.linHcRsl['Vp_pct'][I]))
        # plt.plot(rslt[:,-2:]); plt.show()
    # errors = [2.18,0.00,5.12, 8.82,9.02, 1.98,0.00,15.04, 3.44]
    # pearsonr(errors,mnsS)
    # # pearsonOutR = \(0.869855163294139, 0.002300177033527779)
    # plt.plot(mnsS,errors,'x',label='Change in P with S vs. Error')
    # plt.xlabel('Error indicator (D Prob/D Sgen)')
    # plt.ylabel('Actual error')
    # plt.legend()
    # plt.show()
    
    feederErrors = dict( zip(fdr_i_set,mnsS) )
    SDerrors = os.path.join(WD,'hcResults','feederErrors_corr.pkl')
    with open(SDerrors,'wb') as saveFile:
        pickle.dump(feederErrors,saveFile)
    
    # # PLOT the new stuff
    # i=0
    # for fdr_i in fdr_i_set:
        # plt.plot(np.arange(2,102,2),rsltsDiff[i],label=fdrs[fdr_i])
        # i+=1
    # plt.xlabel('No. loads with PV (%)'); plt.ylabel('Change in Prob, 10% change in Sgen (%)'); plt.legend(); plt.show()


if 'f_limitSensitivityIdv' in locals():
    # an individual PLOT:
    SD = os.path.join(WD,'hcResults','th_kW_mult.pkl')
    with open(SD,'rb') as handle:
        circuitK = pickle.load(handle)

    I = [9,19,29,39,49] # 20-80%
    dMuSet = np.linspace(0.7,1.3,11)
    fdr_i = 21
    LM = linModel(fdr_i,WD)
    dMu0 = circuitK[LM.feeder]

    rslt = np.empty((len(I),0))
    for dMu in dMuSet:
        print(dMu)
        pdf = hcPdfs( LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName='gammaFrac',dMu=dMu*dMu0 )
        LM.runLinHc(pdf,fast=True)
        rslt = np.hstack((rslt,LM.linHcRsl['Vp_pct'][I]))

    plt.plot(dMuSet,rslt.T,'-'); plt.xlabel('Scaled generation (%)'); plt.ylabel('Constraint Violations (%)');
    plt.legend(('20','40','60','80','100'),title='% PV'); plt.title(LM.feeder); plt.show()


if 'f_varCheck' in locals():
    fdrs_is = [6,8,9,19,20,21,17,18,22]
    fdrs_is = [6,20,21,17,18]
    # fdrs_is = [6] # for exploring other ideas
    muIdxs = [-1]*len(fdrs_is)
    # for 9, correlation is 0.3 (!)
    mSzes = {6:1,8:1,9:0.4,17:0.4,18:0.4,19:0.5,20:0.8,21:0.4,22:0.8}
    pdfName = 'gammaFrac'
    for [fdr_i,idx] in zip(fdrs_is,muIdxs):
        LM = linModel(fdr_i,WD)
        pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )

        Mu,Sgm = pdf.getMuStd(LM,0)
        LM.busViolationVar(Sgm)
        LM.getCovMat(getFixCov=False,getTotCov=False,getFullCov=True)
        
        
        vars0 = LM.varKfullU.copy()
        varSortN0 = vars0.argsort()[::-1]

        allLims0 = np.r_[LM.svdLim,LM.svdLimDv]
        nStd0 = np.sign(allLims0)/np.sqrt(vars0)
        
        Mu, Sgm = pdf.getMuStd(LM,idx)
        LM.busViolationVar(Sgm,Mu=Mu)
        LM.getCovMat(getFixCov=False,getTotCov=False,getFullCov=True)
        vars = LM.varKfullU.copy()
        
        allLims = np.r_[LM.svdLim,LM.svdLimDv]
        nStd = np.sign(allLims)/np.sqrt(vars)
        
        varSortN_new = nStd.argsort()
        varSortN_old = nStd[varSortN0].argsort()
        
        xX = varSortN_old
        yY = np.arange(len(varSortN_new))
        positiveNew = nStd[varSortN_new]>0
        
        print(pearsonr(xX,yY))

        fig,ax = plt.subplots(figsize=(2.4,2.4))    
        plt.plot( xX[positiveNew],yY[positiveNew],'.',markersize=mSzes[fdr_i] ); 
        plt.plot( xX[positiveNew==0],yY[positiveNew==0],'.',markersize=mSzes[fdr_i] ); 
        # plt.plot( np.arange(len(varSortN_new)),varSortN_new,'.',markersize=1 ); 
        plt.axis('equal')
        plt.grid()
        plt.xlabel('Sorted Idx., 0% pen.')
        plt.ylabel('Sorted Idx., 100% pen.')
        
        if fdr_i==20:
            ax.annotate('Strongest\nnodes',
                xy=(1850,1950), xycoords='data',
                xytext=(100,1650), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
            ax.annotate('Weakest\nnodes',
                xy=(200,40), xycoords='data',
                xytext=(1000,30), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
        
        plt.tight_layout()
        if 'pltSave' in locals():
            plotSaveFig(sdt('c4','f')+'\\varCheck_'+fdrs[fdr_i],True,True)
        plt.show()
        

if 'f_corVarScanCalc' in locals():
    fdrs_is = [6,8,9,19,20,21,17,18,22]
    # fdrs_is = [22]
    # fdrs_is = [6]

    pdfName = 'gammaFrac'

    stdLim = np.array([0.5,0.6,0.7,0.8,0.9,0.99])
    CorLim = np.array([0.95,0.98,0.99,1.00])

    Errors = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
    Times = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
    Sizes = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
    CovCalcTimes = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )
    # VarCalcTimes = np.zeros( (len(fdrs_is),len(stdLim),len(CorLim)) )

    for i in range(len(fdrs_is)):
        LM = linModel(fdrs_is[i],WD)
        pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
        LM.runLinHc(pdf,fast=True)
        rslA = LM.linHcRsl
        rslAt = rslA['runTime']
        Mu,Sgm = pdf.getMuStd(LM,0)
        for j in range( len(stdLim) ):
            for k in range( len(CorLim) ):
                tCov0 = time.time()
                LM.busViolationVar(Sgm)
                LM.makeCorrModel(stdLim=stdLim[j],corrLim=[CorLim[k]]) # this takes ages and ages!
                CovCalcTimes[i,j,k] = time.time() - tCov0
                LM.runLinHc(pdf,model='cor',fast=True)
                # tVar0 = time.time()
                # LM.busViolationVar(Sgm,calcSrsVals=True)
                # LM.makeVarLinModel(stdLim=[stdLim[j]])
                # VarCalcTimes[i,j,k] = time.time() - tVar0
                # LM.runLinHc(pdf,model='std',fast=True)
                rslB = LM.linHcRsl
                Errors[i,j,k] = np.mean(np.abs(rslB['Vp_pct']-rslA['Vp_pct']))
                Times[i,j,k] = LM.linHcRsl['runTime']
                Sizes[i,j,k] = len(LM.NSetCor[0])/len(LM.varKfullU)*100
    
    
    rsltsDict = {'fdrs_is':fdrs_is,'fdrs':fdrs,'Errors':Errors,'Times':Times,'Sizes':Sizes,'CovCalcTimes':CovCalcTimes,'stdLim':stdLim,'CorLim':CorLim}
    SN = os.path.join(WD,'hcResults','corVarScan.pkl')
    # with open(SN,'wb') as file: # <--------------- UNCOMMENT THESE TO SAVE
        # pickle.dump(rsltsDict,file)

if 'f_corVarScan' in locals() or 'f_corVarCTs' in locals():
    SN = os.path.join(WD,'hcResults','corVarScan.pkl')
    with open(SN,'rb') as file:
        rsltsDictOut = pickle.load(file)

    stdLim = rsltsDictOut['stdLim']
    CorLim = rsltsDictOut['CorLim']
    Errors = rsltsDictOut['Errors']
    Times = rsltsDictOut['Times']
    Sizes = rsltsDictOut['Sizes']
    fdrs_is = rsltsDictOut['fdrs_is']
    CovCalcTimes = rsltsDictOut['CovCalcTimes']
    
    if 'f_corVarScan' in locals():
        # fig,[ax0,ax1,ax2] = plt.subplots(figsize=(8,3.2),ncols=3)
        fig,ax0 = plt.subplots(figsize=(2.65,3.2))
        fig,ax1 = plt.subplots(figsize=(2.65,3.2))
        fig,ax2 = plt.subplots(figsize=(2.65,3.2))
        for i in range(len(CorLim)):
            # ax0.plot(100*stdLim,np.max(Errors[:,:,i],axis=0),'x-',label=( '%.0f' % (100*CorLim[i]) ) + ' %')
            ax0.plot(100*stdLim,np.mean(Sizes[:,:,i],axis=0),'x-',label='$\eta _{\mathrm{Cor}}=$'+( '%.0f' % (100*CorLim[i]) ))
            ax1.plot(100*stdLim,np.max(Errors[:,:,i],axis=0),'x-',label=( '%.0f' % (100*CorLim[i]) ) + ' %')
            ax2.plot(100*stdLim,np.max(Times[:,:,i],axis=0),'x-',label=( '%.0f' % (100*CorLim[i]) ) + ' %')

        ax0.legend(title='Cov. Cutoff, $\eta _{\mathrm{Cor}}$')
        ax0.set_xlabel('Variance cutoff, $\eta _{\mathrm{Var}}$, %')
        ax1.set_xlabel('Variance cutoff, $\eta _{\mathrm{Var}}$, %')
        ax2.set_xlabel('Variance cutoff, $\eta _{\mathrm{Var}}$, %')

        ax1.set_ylim((-0.1,10))

        ax0.set_ylabel('Average size (9 ntwks.), %')
        plt.sca(ax0)
        plt.tight_layout()
        if 'pltSave' in locals():
            plotSaveFig(sdt('c4','f')+'\\corVarScanSze')
        
        ax1.set_ylabel('Max Error (9 ntwks.), %')
        plt.sca(ax1)
        plt.tight_layout()
        if 'pltSave' in locals():
            plotSaveFig(sdt('c4','f')+'\\corVarScanErr')
        
        ax2.set_ylabel('Max Time to Solve (9 ntwks.), s')
        plt.sca(ax2)
        plt.tight_layout()
        if 'pltSave' in locals():
            plotSaveFig(sdt('c4','f')+'\\corVarScanTms')
        
        # ax1.set_title('Average Sizes, %')
        # if 'pltSave' in locals():
            # plotSaveFig(sdt('c4','f')+'\\corVarScan')
        plt.show()
    
    if 'f_corVarCTs' in locals():
        fig,ax = plt.subplots(figsize=(4,2.8))
        i = np.where(CorLim==0.98)[0]
        
        ax.plot(100*stdLim,np.max(CovCalcTimes[:,:,i],axis=0),'_-',color=cm.matlab(0),label='Max')
        ax.plot(100*stdLim,np.mean(CovCalcTimes[:,:,i],axis=0),'x-',color=cm.matlab(0),label='Mean')
        # ax.plot(100*stdLim,np.min(CovCalcTimes[:,:,i],axis=0),'_',color=cm.matlab(0),label='Min')
        
        ax.legend()
        ax.set_xlabel('Variance cutoff, $\eta _{\mathrm{Var}}$, %')
        ax.set_ylim((0.0,40))
        ax.set_ylabel('Time to find precond. indices, s')
        # ax1.set_title('Average Sizes, %')
        plt.tight_layout()
        if 'pltSave' in locals():
            plotSaveFig(sdt('c4','f')+'\\corVarCTs')
        plt.show()