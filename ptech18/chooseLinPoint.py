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
pltVxtrm = False
savePts = True
# savePts = False
saveBusCoords = True
saveBusCoords = False
saveBrchBuses = True
saveBrchBuses = False

load1 = 0.2
load2 = 0.66
# Vp0 = 1.05 # pu
# Vm0 = 0.95 # pu
roundInt = 5000.

VpHi = 1.055
VpLo = 1.055
VmHi = 0.95
VmLo = 0.92

fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# fdr_i_set = [0]
for fdr_i in fdr_i_set:
    # fdr_i = 17
    fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
    feeder=fdrs[fdr_i]
    # feeder='041'

    SD = os.path.join(WD,'lin_models',feeder,'chooseLinPoint')
    SN = os.path.join(SD,'chooseLinPoint')+'.pkl'
    SB = os.path.join(SD,'busCoords')+'.pkl'
    SBr = os.path.join(SD,'branches')+'.pkl'

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
        loadMults = np.concatenate([np.linspace(-0.2,1.3,16+1),np.linspace(1.3,-0.2,16+1)])

    vmax = []
    vmin = []
    cnvg = []
    DSSSolutionLoadMult = 1.0
    DSSSolution.Solve()
    
    Vmag0 = np.array(DSSCircuit.AllBusVmag)
    VmagPu = np.array(DSSCircuit.AllBusVmagPu)#
    # Vpu = Vmag/VmagPu
    
    hiVidx = np.where(Vmag0[VmagPu>0.5]>1000)
    loVidx = np.where(Vmag0[VmagPu>0.5]<=1000)

    hiVmax = []
    hiVmin = []
    loVmax = []
    loVmin = []
    
    # VlimMax = 1.06*(Vpu>1000) + 1.05*(Vpu<=1000)
    # VlimMin = 0.95*(Vpu>1000) + 0.92*(Vpu<=1000)
    for lm in loadMults:
        DSSSolution.LoadMult = lm
        DSSSolution.Solve()
        
        VmagPu = np.array(DSSCircuit.AllBusVmagPu)#
        VmagPu = VmagPu[VmagPu>0.5] # get rid of outliers
        
        hiVmax = hiVmax + [VmagPu[hiVidx].max()]
        hiVmin = hiVmin + [VmagPu[hiVidx].min()]
        if len(loVidx[0])>0:
            loVmax = loVmax + [VmagPu[loVidx].max()]
            loVmin = loVmin + [VmagPu[loVidx].min()]
        else:
            loVmax = loVmax + [np.nan]
            loVmin = loVmin + [np.nan]
        
        # vmax = vmax + [VmagPu.max()]
        # vmin = vmin + [VmagPu.min()]
        
        cnvg = cnvg + [DSSSolution.Converged]

    dV = 1e-5 # get rid of pesky cases just below 1.05
    vmaxCl = np.ceil((np.array(vmax)+dV)*roundInt)/roundInt
    vminFl = np.floor((np.array(vmin)-dV)*roundInt)/roundInt

    idx1 = abs(loadMults-load1).argmin()
    idx11 = len(loadMults) - idx1 - 1 # by symmetry
    idx2 = abs(loadMults-load2).argmin()
    idx22 = len(loadMults) - idx2 - 1 # by symmetry

    # Vp = max([Vp0,vmaxCl[idx1],vmaxCl[idx2],vmaxCl[idx11],vmaxCl[idx22]])
    # Vm = min([Vm0,vminFl[idx1],vminFl[idx2],vminFl[idx11],vminFl[idx22]])

    # idxK1 = abs(np.array(vmax)[loadMults<load1]-Vp).argmin()
    # idxK2 = abs(np.array(vmin)[loadMults<load1]-Vm).argmin()
    # if idxK1*2>=sum(loadMults<load1):
        # idxK1 = (sum(loadMults<load1) - 1 - idxK1)
    # if idxK2*2>=sum(loadMults<load1):
        # idxK2 = (sum(loadMults<load1) - 1 - idxK2)
    
    # kOut2 = loadMults[idxK2]
    # kOut1 = loadMults[idxK1]
    # checkVminBounds = np.array(vmin)[loadMults<load1]<Vm
    # if checkVminBounds.any():
        # kOut = max([kOut1,kOut2])
    # else:
        # kOut = kOut1
    
    idxK1hi = abs(np.array(hiVmax)[loadMults<load1]-VpHi).argmin()
    idxK2hi = abs(np.array(hiVmin)[loadMults<load1]-VmHi).argmin()
    idxK1lo = abs(np.array(loVmax)[loadMults<load1]-VpLo).argmin()
    idxK2lo = abs(np.array(loVmin)[loadMults<load1]-VmLo).argmin()
    
    if idxK1hi*2>=sum(loadMults<load1):
        idxK1hi = (sum(loadMults<load1) - 1 - idxK1hi)
    if idxK2hi*2>=sum(loadMults<load1):
        idxK2hi = (sum(loadMults<load1) - 1 - idxK2hi)
    if idxK1lo*2>=sum(loadMults<load1):
        idxK1lo = (sum(loadMults<load1) - 1 - idxK1lo)
    if idxK2lo*2>=sum(loadMults<load1):
        idxK2lo = (sum(loadMults<load1) - 1 - idxK2lo)

    chkBndK1hi = np.array(hiVmax)[loadMults<load1]<VpHi
    chkBndK2hi = np.array(hiVmin)[loadMults<load1]<VmHi
    if len(loVidx[0])>0:
        chkBndK1lo = np.array(loVmax)[loadMults<load1]<VpLo
        chkBndK2lo = np.array(loVmin)[loadMults<load1]<VmLo
    else:
        chkBndK1lo = np.array([0])
        chkBndK2lo = np.array([0])
    
    if not chkBndK1hi.any():
        idxK1hi = 0
    if not chkBndK2hi.any():
        idxK2hi = 0
    if not chkBndK1lo.any():
        idxK1lo = 0
    if not chkBndK2lo.any():
        idxK2lo = 0
    
    kOut1hi = loadMults[idxK1hi]
    kOut2hi = loadMults[idxK2hi]
    kOut1lo = loadMults[idxK1lo]
    kOut2lo = loadMults[idxK2lo]
    
    if idxK1hi + idxK2hi + idxK1lo + idxK2lo==0:
        kOut = loadMults[(loadMults>load1).argmax() - 1]
    else:
        kOut = max([kOut1hi,kOut1lo,kOut2hi,kOut2lo])

    dataOut = {'Feeder':feeder,'Vp':np.nan,'Vm':np.nan,'k':kOut,'kLo':load1,'kHi':load2,'VpHi':VpHi,'VpLo':VpLo,'VmHi':VmHi,'VmLo':VmLo}

    if pltVxtrm:
        # plt.plot(loadMults,vmax,'x-')
        # plt.plot(loadMults,vmin,'x-')
        plt.plot(loadMults,hiVmax,'x-')
        plt.plot(loadMults,hiVmin,'x-')
        plt.plot(loadMults,loVmax,'.-')
        plt.plot(loadMults,loVmin,'.-')
        xlm = plt.xlim()
        ylm = plt.ylim()
        # plt.plot(xlm,[Vp,Vp],'k--')
        # plt.plot(xlm,[Vm,Vm],'k--')
        plt.plot(xlm,[VpHi,VpHi],'k--')
        plt.plot(xlm,[VmHi,VmHi],'k--')
        plt.plot(xlm,[VpLo,VpLo],'k:')
        plt.plot(xlm,[VmLo,VmLo],'k:')
        plt.plot([kOut,kOut],ylm,'k')
        plt.plot([load1,load1],ylm,'k:')
        plt.plot([load2,load2],ylm,'k:')
        
        plt.xlim(xlm)
        plt.ylim(ylm)
        plt.grid(True)
        plt.xlabel('Continuation factor, $\kappa$')
        plt.ylabel('Voltage (pu)')
        # plt.legend(('Vmax','Vmin'))
        plt.legend(('hiVmax','hiVmin','loVmax','loVmin'))
        plt.show()

    print('Feeder:',feeder)
    print('No. converged:',sum(cnvg),'/',len(cnvg))
    # print('Vmax:',Vp,'pu\nVmin:',Vm,'pu')
    print('Kout:',kOut)

    if savePts:
        if not os.path.exists(SD):
            os.makedirs(SD)

        with open(SN,'wb') as handle:
            pickle.dump(dataOut,handle)
            

    if saveBusCoords:
        busCoords = getBusCoords(DSSCircuit,DSSText)
        if not os.path.exists(SD):
            os.makedirs(SD)
        with open(SB,'wb') as handle:
            pickle.dump(busCoords,handle)

    if saveBrchBuses:
        branches = getBranchBuses(DSSCircuit,DSSText)
        if not os.path.exists(SD):
            os.makedirs(SD)
        with open(SBr,'wb') as handle:
            pickle.dump(branches,handle)