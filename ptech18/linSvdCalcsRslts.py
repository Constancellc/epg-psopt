import pickle, os, sys, win32com.client, time, scipy.stats, getpass
import numpy as np
from dss_python_funcs import *
import matplotlib.pyplot as plt
from matplotlib import cm
import dss_stats_funcs as dsf
from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn, plotBoxWhisk
plt.style.use('tidySettings')

WD = os.path.dirname(sys.argv[0])
SD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy']

pltShow = 1
# pltSave = 1
# pltCc = 1
# f_nStdBefore = 1
# f_nStdAfter = 1
# f_nStdVreg = 1
# f_nStdVregVal = 1
# f_corVars = 1
# f_hcParamSlctnCaseStudy = 1

# # calculating setpoints for linHcCalcs:
# f_nStdVreg_8500 = 1
# f_nStdVreg_epriM1 = 1
# f_nStdVreg_epri24 = 1
# f_nStdVreg_epriJ1 = 1
# f_nStdVreg_epriK1 = 1

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

    LM.busViolationVar(Sgm)
    LM.makeCorrModel()
    # LM.corrPlot()

    vars = LM.varKfullU.copy()
    varSortN = vars.argsort()[::-1]

    # corrLogAbs = np.log10(abs((1-LM.KtotUcorr)) + np.diag(np.ones(len(LM.KtotPu))) +1e-14 )
    corrLogAbs = np.log10(abs((1-LM.KfullUcorr)) + np.diag(np.ones(len(LM.KfullUcorr))) +1e-14 )
    corrLogAbs = corrLogAbs[varSortN][:,varSortN]

    fig,(ax1,ax0) = plt.subplots(figsize=(5.9,3.8),nrows=1,ncols=2, gridspec_kw = {'width_ratios':[1,1.8]})
    ax0.spy(corrLogAbs<-2.0,color=cm.Blues(0.9),markersize=1,marker='.',zorder=4,label='99%') # 99%
    ax0.spy(corrLogAbs<-1.7,color=cm.Blues(0.7),markersize=1,marker='.',zorder=3,label='98%') # 98%
    ax0.spy(corrLogAbs<-1.3,color=cm.Blues(0.5),markersize=1,marker='.',zorder=2,label='95%') # 95%
    ax0.spy(corrLogAbs<-1.0,color=cm.Blues(0.3),markersize=1,marker='.',zorder=1,label='90%') # 90%

    # ax0.plot(0,0,'o',color=cm.Blues(0.45))
    # ax0.plot(0,0,'o',color=cm.Blues(0.6),label='> 98%')
    # ax0.plot(0,0,'o',color=cm.Blues(0.75),label='> 95%')
    # ax0.plot(0,0,'o',color=cm.Blues(0.9),label='> 90%')

    legend = ax0.legend(loc='upper right',borderpad=0.55,fontsize='small',handletextpad=0.28,markerscale=10)
    ax0.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax0.set_yticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_xlabel('Bus Index $i$')
    ax0.set_ylabel('Bus Index $j$')

    ax1.plot(vars[varSortN]/vars[varSortN][0])
    ax1.set_xlim((-50,len(corrLogAbs+50)))
    ax1.set_xlabel('Bus Index $i$')
    ax1.set_ylabel('Var(Bus $i$), normalised')
    ax1.set_yscale('log')
    ax1.set_xticks(ticks=[len(corrLogAbs)//4,len(corrLogAbs)//2,3*len(corrLogAbs)//4])
    ax1.set_xticklabels([])
    ax1.set_yticks(ticks=[1,0.01,0.0001])

    ax0.set_title('Correlation',pad=-5.5)
    ax1.set_title('Variance',pad=5.5)
    plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'corVars_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0.01)
        plt.savefig(SD+'corVars_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
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