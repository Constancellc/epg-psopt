import pickle, os, sys, win32com.client, time, scipy.stats, getpass
import numpy as np
from dss_python_funcs import *
import matplotlib.pyplot as plt
from matplotlib import cm
import dss_stats_funcs as dsf
from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn, plotBoxWhisk

WD = os.path.dirname(sys.argv[0])
SD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

pltShow = 1
# pltSave = 1
# pltCc = 1
# f_nStdBefore = 1
# f_nStdAfter = 1
# f_nStdVreg = 1
# f_nStdVregVal = 1

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

# ============================ EXAMPLE: plotting the number of standard deviations for a network changing ***vregs*** uniformly
fdr_i = 22
pdfName = 'gammaFrac'
print('Load Linear Model feeder:',fdrs[fdr_i],'\nPdf type:',pdfName,'\n',time.process_time())
LM = linModel(fdr_i,WD,QgenPf=1.0)
pdf = hcPdfs(LM.feeder,WD=WD,netModel=LM.netModelNom,pdfName=pdfName )
Mu, Sgm = pdf.getMuStd(LM=LM,prmI=-1) # in W

if 'f_nStdBefore' in locals():
    LM.busViolationVar(Sgm,Mu=Mu) # 100% point
    LM.plotNetBuses('nStd',pltType='max',minMax=[-3.,6.],cmap=plt.cm.inferno,pltShow=False)
    if 'pltSave' in locals():
        plt.savefig(SD+'nStdBefore_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdBefore_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()

if 'f_nStdAfter' in locals():
    optVal = 0.98
    LM.updateDcpleModel(LM.regVreg0*optVal)
    LM.busViolationVar(Sgm,Mu=Mu)
    LM.plotNetBuses('nStd',pltType='max',minMax=[-3.,6.],cmap=cm.inferno,pltShow=False)
    if 'pltSave' in locals():
        plt.savefig(SD+'nStdAfter_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdAfter_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()

# LM.runLinHc(pdf,model='nom') # model options: nom / std / cor / mxt ?
# plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=LM.feeder)

xTickMatch = np.linspace(0.98,1.04,7)
xTickMatchStr = ["%.2f" % x for x in xTickMatch]
xlims = (0.975,1.045)

Q_set = [1.0,-0.98]
aFro = 1e-6
if 'f_nStdVreg' in locals():
    nOpts = 81
    t = time.time()
    opts = np.linspace(0.955,1.025,nOpts)
    N0 = np.zeros((len(Q_set),len(opts)))
    for i in range(len(Q_set)):
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
    # plt.plot(np.outer(opts,[1]*len(N0)),N0.T,'.-')
    plt.plot(np.outer(opts*LM.regVreg0/(166*120),[1]*len(N0)),N0.T,'.-')
    plt.xlabel('Regulator setpoint, $V_{\mathrm{reg}}$ (pu)')

    plt.ylabel('Min. no. of standard deviations to constraint, $\min(N_{\sigma})$')
    plt.legend(('1.0','0.98'),title='PF (lagging)'); 
    plt.xlim(xlims); plt.ylim((-12,9))
    plt.xticks(xTickMatch,xTickMatchStr)
    plt.grid(True); plt.tight_layout()

    if 'pltSave' in locals():
        plt.savefig(SD+'nStdVreg_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
        plt.savefig(SD+'nStdVreg_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)
    if 'pltShow' in locals():
        plt.show()

if 'f_nStdVregVal':
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
fig = plt.figure()
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
plt.legend(title='PF (lag)',loc='center left')
plt.ylim((-1,101))
plt.xlim((-0.8,2*len(xTickMatch)-0.5))
plt.grid(True)
    
if 'pltSave' in locals():
    plt.savefig(SD+'nStdVregVal_'+fdrs[fdr_i]+'.png',bbox_inches='tight', pad_inches=0)
    plt.savefig(SD+'nStdVregVal_'+fdrs[fdr_i]+'.pdf',bbox_inches='tight', pad_inches=0)

if 'pltShow' in locals():
    plt.show()


# ==========================================