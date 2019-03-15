import win32com.client
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
import pickle


DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
DSSSolution.tolerance=1e-7

pltVxtrm = True
# pltVxtrm = False
savePts = True
# savePts = False

load1 = 0.2
load2 = 1.0
Vp0 = 1.05 # pu
Vm0 = 0.95 # pu
roundInt = 200.

fdr_i = 22
fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder=fdrs[fdr_i]
feeder='041'

WD = os.path.dirname(sys.argv[0])
SD = os.path.join(WD,'lin_models',feeder,'chooseLinPoint')
SN = os.path.join(SD,'chooseLinPoint')+'.pkl'

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSText.command='Compile ('+fn+'.dss)'
DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3 status=variable'
DSSText.command='set maxcontroliter=300' # if it isn't this high then J1 fails even for load=1.0!
DSSText.command='set maxiterations=300'
DSSSolution.Solve()

YZ = DSSCircuit.YNodeOrder
vBase = get_Yvbase(DSSCircuit)

loadMults = np.linspace(-1.5,1.5,31+1)
if feeder=='epriJ1':
    loadMults = np.linspace(-0.4,1.5,21+1) # required for EPRI J1 which is rather temporamental
elif feeder=='8500node':
    loadMults = np.linspace(-0.2,1.5,18+1) # required for IEEE 8500

vmax = []
vmin = []
cnvg = []
for lm in loadMults:
    DSSSolution.LoadMult = lm
    DSSSolution.Solve()
    
    VmagPu = np.array(DSSCircuit.AllBusVmagPu)#
    VmagPu = VmagPu[VmagPu>0.5] # get rid of outliers
    
    vmax = vmax + [VmagPu.max()]
    vmin = vmin + [VmagPu.min()]
    
    cnvg = cnvg + [DSSSolution.Converged]

dV = 1e-5 # get rid of pesky cases just below 1.05
vmaxCl = np.ceil((np.array(vmax)+dV)*roundInt)/roundInt
vminFl = np.floor((np.array(vmin)-dV)*roundInt)/roundInt

idx1 = abs(loadMults-load1).argmin()
idx2 = abs(loadMults-load2).argmin()
Vp = max([Vp0,vmaxCl[idx1],vmaxCl[idx2]])
Vm = min([Vm0,vminFl[idx1],vminFl[idx2]])

kOut = loadMults[abs(np.array(vmax)[loadMults<load1]-Vp).argmin()]
dataOut = {'Feeder':feeder,'Vp':Vp,'Vm':Vm,'k':kOut,'kLo':load1,'kHi':load2}

if pltVxtrm:
    plt.plot(loadMults,vmax,'x-')
    plt.plot(loadMults,vmin,'x-')
    xlm = plt.xlim()
    ylm = plt.ylim()
    plt.plot(xlm,[Vp,Vp],'k--')
    plt.plot(xlm,[Vm,Vm],'k--')
    plt.plot([kOut,kOut],ylm,'k:')
    
    plt.xlim(xlm)
    plt.ylim(ylm)
    plt.grid(True)
    plt.xlabel('Continuation factor, $\kappa$')
    plt.ylabel('Voltage (pu)')
    plt.legend(('Vmax','Vmin'))
    plt.show()

print('Feeder:',feeder)
print('No. converged:',sum(cnvg),'/',len(cnvg))
print('Vmax:',Vp,'pu\nVmin:',Vm,'pu')
print('Kout:',kOut)

if savePts:
    if not os.path.exists(SD):
        os.makedirs(SD)

    with open(SN,'wb') as handle:
        pickle.dump(dataOut,handle)