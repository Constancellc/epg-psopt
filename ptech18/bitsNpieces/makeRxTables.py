import win32com.client, os, sys, pickle, getpass

WD = os.path.dirname(os.path.dirname(sys.argv[0])) # working one directory up from usual
sys.path.append(WD)

import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from matplotlib import cm, rc
plt.style.use('tidySettings')
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns
clrA,clrB,clrC,clrD,clrE,clrF,clrG = cm.matlab(np.arange(7))
sys.argv=["makepy","OpenDSSEngine.DSS"]
from win32com.client import makepy
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
SDT = sdt('t1','f')

# OPTIONS here vvv
rc('text',usetex=True)
pltSave = 1
fdr_i_set = [0,5,6,8,9,17,18,22,19,20,21]
# fdr_i_set = [0,5,6]
figSze0=(4.0,2.5)
#              ^^^


RXratiosQ1 = []; RXratiosQN = []; feederTidySet = []
X = np.arange(len(fdr_i_set),0,-1)

fig,ax = plt.subplots(figsize=figSze0)
for fdr_i,i in zip(fdr_i_set,range(len(fdr_i_set))):
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']
    feeder=fdrs[fdr_i]
    fn = get_ckt(WD,feeder)[1]
    
    DSSText.Command='Compile ('+fn+'.dss)'
    DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3 model=1 status=variable'
    DSSText.Command='set maxcontroliter=300' # if it isn't this high then J1 fails even for load=1.0!
    DSSText.Command='set maxiterations=300'
    DSSSolution.Solve()
    DSSText.Command='set mode=faultstudy'
    DSSSolution.Solve()
    ABE = DSSCircuit.ActiveBus
    
    busSet = DSSCircuit.AllBusNames
    Zsc1set = np.array([])
    ZscNset = np.array([])
    for bus in busSet:
        DSSCircuit.SetActiveBus(bus)
        Zsc1set = np.r_[Zsc1set,tp_2_ar(ABE.Zsc1)]
        ZscNset = np.r_[ ZscNset,tp_2_ar(ABE.ZscMatrix)[0::ABE.NumNodes] ]

    RXratios1 = Zsc1set.real/Zsc1set.imag
    RXratiosN = ZscNset.real/ZscNset.imag

    feedersTidy = {'eulv':'EU LV','13bus':'13 Bus','34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24'}
    
    RXratiosQ1.append(np.quantile(RXratios1,[0,0.05,0.25,0.5,0.75,0.95,1]))
    RXratiosQN.append(np.quantile(RXratiosN,[0,0.05,0.25,0.5,0.75,0.95,1]))
    ddx = 0.25
    # ax = plotBoxWhisk(ax,X[i],ddx,RXratiosQ1[i][1:-1],clrA,bds=RXratiosQ1[i][[0,-1]]+np.array([1,-1]),transpose=True)
    ax = plotBoxWhisk(ax,X[i],ddx,RXratiosQ1[i][1:-1],clrA,bds=(ax,X[i],ddx,RXratiosQ1[i][1:-1],clrA,bds=(RXratiosQ1[i][[0,-1]]*np.array([0.88,1.12]))+np.array([1,-1]),transpose=True,lineWidth=0.7)
    # ax = plotBoxWhisk(ax,X[i]-0.15,ddx,RXratiosQN[i][1:-1],clrB,bds=RXratiosQN[i][[0,-1]]+np.array([1,-1]),transpose=True)
    feederTidySet.append(feedersTidy[feeder])

plt.yticks(X,feederTidySet)
ax.set_xscale('log')
ax.set_xlim((0.01,100))
ax.set_xlabel('$R/X$ ratio, $\lambda$')

plt.tight_layout()
if 'pltSave' in locals():
    plotSaveFig(os.path.join(SDT,'makeRxTables'),pltClose=True)
plt.show()
# plotBoxWhisk(ax,1,RXratiosQs)