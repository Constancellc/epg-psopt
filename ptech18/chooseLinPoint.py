import win32com.client, os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *

WD = os.path.dirname(sys.argv[0])
sys.argv=["makepy","OpenDSSEngine.DSS"]

from win32com.client import makepy
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
DSSSolution.Tolerance=1e-7

pltVxtrm = True
# pltVxtrm = False
savePts = True
savePts = False

load1 = 0.2
load2 = 1.0
Vp0 = 1.05 # pu
Vm0 = 0.95 # pu
roundInt = 5000.

fdr_i = 22
fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder=fdrs[fdr_i]
# feeder='041'

SD = os.path.join(WD,'lin_models',feeder,'chooseLinPoint')
SN = os.path.join(SD,'chooseLinPoint')+'.pkl'

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSText.Command='Compile ('+fn+'.dss)'
DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3 status=variable'
# DSSText.Command='Batchedit load..* vmin=0.33 vmax=3.0 model=1'
DSSText.Command='set maxcontroliter=300' # if it isn't this high then J1 fails even for load=1.0!
DSSText.Command='set maxiterations=300'
DSSSolution.Solve()

YZ = DSSCircuit.YNodeOrder
vBase = get_Yvbase(DSSCircuit)

loadMults = np.concatenate([np.linspace(-1.5,1.5,31+1),np.linspace(1.5,-1.5,31+1)])
if feeder=='epriJ1':
    # required for EPRI J1 which is rather temporamental
    loadMults = np.concatenate([np.linspace(-0.4,1.5,21+1),np.linspace(1.5,-0.4,21+1)])
elif feeder=='8500node':
    # required for IEEE 8500
    loadMults = np.concatenate([np.linspace(-0.2,1.5,18+1),np.linspace(1.5,-0.2,18+1)])

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
idx11 = len(loadMults) - idx1 - 1 # by symmetry
idx2 = abs(loadMults-load2).argmin()
idx22 = len(loadMults) - idx2 - 1 # by symmetry

Vp = max([Vp0,vmaxCl[idx1],vmaxCl[idx2],vmaxCl[idx11],vmaxCl[idx22]])
Vm = min([Vm0,vminFl[idx1],vminFl[idx2],vminFl[idx11],vminFl[idx22]])

idxK1 = abs(np.array(vmax)[loadMults<load1]-Vp).argmin()
idxK2 = abs(np.array(vmin)[loadMults<load1]-Vm).argmin()

if idxK1*2>=sum(loadMults<load1):
    idxK1 = (sum(loadMults<load1) - 1 - idxK1)
if idxK2*2>=sum(loadMults<load1):
    idxK2 = (sum(loadMults<load1) - 1 - idxK2)

kOut1 = loadMults[idxK1]
kOut2 = loadMults[idxK2]

checkVminBounds = np.array(vmin)[loadMults<load1]<Vm
if checkVminBounds.any():
    kOut = max([kOut1,kOut2])
else:
    kOut = kOut1


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