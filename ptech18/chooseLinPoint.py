import win32com.client, os, sys, pickle, getpass
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
from dss_voltage_funcs import getCapPos
from matplotlib import cm, rc
plt.style.use('tidySettings')

# rc('text',usetex=True)
figSze0=(5.2,3.0)

WD = os.path.dirname(sys.argv[0])
sys.argv=["makepy","OpenDSSEngine.DSS"]

from win32com.client import makepy
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

pltVxtrm = True
pltVxtrm = False
savePts = True
# savePts = False
saveBusCoords = True
saveBusCoords = False
saveBrchBuses = True
saveBrchBuses = False
saveRegBandwidths = True
saveRegBandwidths = False
pltVxtrmSave = True
pltVxtrmSave = False # use this for plotting for the paper
# pltCapPos = 1

SDfig = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

load1 = 0.2
load2 = 0.66
# Vp0 = 1.05 # pu
# Vm0 = 0.95 # pu
roundInt = 5000.

VpMv = 1.055
VpLv = 1.055
VmMv = 0.95
VmLv = 0.92

fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdr_i_set = [6,8,9,17,18,22,19,20,21]
fdr_i_set = [20]
fdr_i_set = [23]
for fdr_i in fdr_i_set:
    # fdr_i = 17
    fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']
    feeder=fdrs[fdr_i]
    # feeder='041'

    SD = os.path.join(WD,'lin_models',feeder,'chooseLinPoint')
    SN = os.path.join(SD,'chooseLinPoint')+'.pkl'
    SB = os.path.join(SD,'busCoords')+'.pkl'
    SBa = os.path.join(SD,'busCoordsAug')+'.pkl'
    SBr = os.path.join(SD,'branches')+'.pkl'
    SBreg = os.path.join(SD,'regBndwth')+'.pkl'
    
    fn_ckt,fn = get_ckt(WD,feeder)

    DSSText.Command='Compile ('+fn+'.dss)'
    DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3 model=1 status=variable'

    DSSText.Command='set maxcontroliter=300' # if it isn't this high then J1 fails even for load=1.0!
    DSSText.Command='set maxiterations=300'
    
    DSSSolution.Solve()
    
    if pltVxtrmSave:
        loadMults = np.concatenate([np.linspace(-1.5,1.5,31+1)])
    else:
        loadMults = np.concatenate([np.linspace(-1.5,1.5,31+1),np.linspace(1.5,-1.5,31+1)])
        if feeder=='epriJ1': loadMults = np.concatenate([np.linspace(-0.4,1.5,21+1),np.linspace(1.5,-0.4,21+1)])
        if feeder=='8500node': loadMults = np.concatenate([np.linspace(-0.2,1.0,13+1),np.linspace(1.0,-0.2,13+1)])
        # if feeder=='4busYy': loadMults = np.concatenate([np.linspace(-1.5,1.4,30+1),np.linspace(1.4,-1.5,30+1)])    

    vmax = []
    vmin = []
    cnvg = []
    DSSSolution.Solve()
    
    Vmag0 = np.array(DSSCircuit.AllBusVmag)
    VmagPu = np.array(DSSCircuit.AllBusVmagPu)

    mvIdx = np.where(Vmag0[VmagPu>0.5]>1000)
    lvIdx = np.where(Vmag0[VmagPu>0.5]<=1000)
        
    VmagYz = abs(tp_2_ar(DSSCircuit.YNodeVarray))
    mvIdxYz = np.where(VmagYz>1000)
    lvIdxYz = np.where(VmagYz<=1000)
    
    mvMax = []
    mvMin = []
    lvMax = []
    lvMin = []
    capPos = []
    
    for lm in loadMults:
        DSSSolution.LoadMult = lm
        DSSSolution.Solve()
        
        VmagPu = np.array(DSSCircuit.AllBusVmagPu)#
        VmagPu = VmagPu[VmagPu>0.5] # get rid of outliers
        
        mvMax = mvMax + [VmagPu[mvIdx].max()]
        mvMin = mvMin + [VmagPu[mvIdx].min()]
        if len(lvIdx[0])>0:
            lvMax = lvMax + [VmagPu[lvIdx].max()]
            lvMin = lvMin + [VmagPu[lvIdx].min()]
        else:
            lvMax = lvMax + [np.nan]
            lvMin = lvMin + [np.nan]
        
        cnvg = cnvg + [DSSSolution.Converged]
        capPos.append(getCapPos(DSSCircuit))
    
    dV = 1e-5 # get rid of pesky cases just below 1.05
    vmaxCl = np.ceil((np.array(vmax)+dV)*roundInt)/roundInt
    vminFl = np.floor((np.array(vmin)-dV)*roundInt)/roundInt

    idx1 = abs(loadMults-load1).argmin()
    idx11 = len(loadMults) - idx1 - 1 # by symmetry
    idx2 = abs(loadMults-load2).argmin()
    idx22 = len(loadMults) - idx2 - 1 # by symmetry

    idxVpMv = abs(np.array(mvMax)[loadMults<load1]-VpMv).argmin()
    idxVmMv = abs(np.array(mvMin)[loadMults<load1]-VmMv).argmin()
    idxVpLv = abs(np.array(lvMax)[loadMults<load1]-VpLv).argmin()
    idxVmLv = abs(np.array(lvMin)[loadMults<load1]-VmLv).argmin()
    
    if idxVpMv*2>=sum(loadMults<load1):
        idxVpMv = (sum(loadMults<load1) - 1 - idxVpMv)
    if idxVmMv*2>=sum(loadMults<load1):
        idxVmMv = (sum(loadMults<load1) - 1 - idxVmMv)
    if idxVpLv*2>=sum(loadMults<load1):
        idxVpLv = (sum(loadMults<load1) - 1 - idxVpLv)
    if idxVmLv*2>=sum(loadMults<load1):
        idxVmLv = (sum(loadMults<load1) - 1 - idxVmLv)

    chkBndVpMv = np.array(mvMax)[loadMults<load1]<VpMv
    chkBndVmMv = np.array(mvMin)[loadMults<load1]<VmMv
    if len(lvIdx[0])>0:
        chkBndVpLv = np.array(lvMax)[loadMults<load1]<VpLv
        chkBndVmLv = np.array(lvMin)[loadMults<load1]<VmLv
    else:
        chkBndVpLv = np.array([0])
        chkBndVmLv = np.array([0])
    
    if not chkBndVpMv.any():
        idxVpMv = 0
    if not chkBndVmMv.any():
        idxVmMv = 0
    if not chkBndVpLv.any():
        idxVpLv = 0
    if not chkBndVmLv.any():
        idxVmLv = 0
    
    kOutVpMv = loadMults[idxVpMv]
    kOutVmMv = loadMults[idxVmMv]
    kOutVpLv = loadMults[idxVpLv]
    kOutVmLv = loadMults[idxVmLv]
    
    if idxVpMv + idxVmMv + idxVpLv + idxVmLv==0:
        kOut = loadMults[(loadMults>load1).argmax() - 1]
    else:
        kOut = max([kOutVpMv,kOutVpLv,kOutVmMv,kOutVmLv])
    kOutIdx = np.where(loadMults==kOut)[0][0]
    capPosOut = capPos[kOutIdx]
    
    DSSCircuit.Vsources.First
    vSrcBuses = DSSCircuit.ActiveElement.BusNames
    
    if feeder=='epri24' or feeder=='8500node' or feeder=='123bus' or feeder=='epriJ1' or feeder=='epriK1' or feeder=='epriM1': # if there is a regulator 'on' the source bus
        srcReg = 1
    else:
        srcReg = 0
    legLoc = {'eulv':'NorthEast','13bus':'NorthEast','34bus':'NorthWest','37bus':None,'123bus':'NorthEast','8500node':'SouthEast','usLv':None,'epri5':'NorthWest','epri7':'NorthWest','epriJ1':'SouthEast','epriK1':'NorthWest','epriM1':'NorthWest','epri24':'NorthWest','4busYy':None}
    
    dataOut = {'Feeder':feeder,'k':kOut,'kLo':load1,'kHi':load2,'VpMv':VpMv,'VpLv':VpLv,'VmMv':VmMv,'VmLv':VmLv,'mvIdxYz':mvIdxYz,'lvIdxYz':lvIdxYz,'nRegs':DSSCircuit.RegControls.Count,'vSrcBus':vSrcBuses[0],'srcReg':srcReg,'legLoc':legLoc[feeder],'capPosOut':capPosOut}

    if pltVxtrm:
        fig = plt.figure(figsize=figSze0)
        ax = fig.add_subplot(111)
        
        clrs = cm.matlab([0,1])
        ax.set_prop_cycle(color=clrs)
        
        ax.plot(loadMults,mvMax,'x-',markersize=4,label='$\max|V_{\mathrm{MV}}|$')
        ax.plot(loadMults,lvMax,'x-',markersize=4,label='$\max|V_{\mathrm{LV}}|$')
        ax.plot(loadMults,mvMin,'.-',markersize=4,label='$\min|V_{\mathrm{MV}}|$')
        ax.plot(loadMults,lvMin,'.-',markersize=4,label='$\min|V_{\mathrm{LV}}|$')
        
        xlm = ax.get_xlim()
        ylm = ax.get_ylim()
        ax.plot(xlm,[VpMv,VpMv],'k:')
        ax.plot(xlm,[VmMv,VmMv],'k:')
        
        ax.plot(xlm,[VpLv,VpLv],'k--')
        ax.plot(xlm,[VmLv,VmLv],'k--')
        
        yLow = 0.915
        ax.plot([kOut,kOut],[yLow,ylm[1]],'k')
        ax.set_xlim(xlm)

        ax.set_ylim([yLow,ylm[1]])
        ax.set_xlabel('Load power continuation factor, $\kappa$')
        ax.set_ylabel('Voltage (pu)')
        ax.legend(loc='upper right')

        if pltVxtrmSave:
            ax.annotate('Lin. point',(kOut-0.11,ylm[1] - 0.14),rotation=90)
            plt.tight_layout()
            plt.savefig(SDfig+'pltVxtrm_'+feeder)
            plt.savefig(SDfig+'pltVxtrm_'+feeder+'.pdf')
            plt.show()
        else:
            ax.grid(True)
            ax.plot([load1,load1],[yLow,ylm[1]],'k:')
            ax.plot([load2,load2],[yLow,ylm[1]],'k:')
            ax.set_title(feeder)    
            plt.tight_layout()
            plt.show()
            
        

    print('\nFeeder:',feeder)
    print('No. converged:',sum(cnvg),'/',len(cnvg))
    # print('Vmax:',Vp,'pu\nVmin:',Vm,'pu')
    print('Kout:',kOut)
    print('Cap Positions:', capPosOut)

    if 'pltCapPos' in locals():
        plt.plot(loadMults,np.sum(np.array(capPos),axis=1))
        plt.plot(loadMults[kOutIdx],sum(capPosOut),'o'); plt.show()
        

    if savePts:
        if not os.path.exists(SD):
            os.makedirs(SD)

        with open(SN,'wb') as handle:
            pickle.dump(dataOut,handle)
            
    if saveBusCoords:
        busCoords = getBusCoords(DSSCircuit,DSSText)
        busCoordsAug,PDelements,PDparents = getBusCoordsAug(busCoords,DSSCircuit,DSSText)
        if not os.path.exists(SD):
            os.makedirs(SD)
        with open(SB,'wb') as handle:
            pickle.dump(busCoords,handle)
        with open(SBa,'wb') as handle:
            pickle.dump(busCoordsAug,handle)

    if saveBrchBuses:
        branches = getBranchBuses(DSSCircuit)
        if not os.path.exists(SD):
            os.makedirs(SD)
        with open(SBr,'wb') as handle:
            pickle.dump(branches,handle)
    
    if saveRegBandwidths:
        i = DSSCircuit.RegControls.First
        bandWidths = [] # follows the order of the regcontrols
        while i:
            bandWidths.append(DSSCircuit.RegControls.ForwardBand)
            i = DSSCircuit.RegControls.Next
        if not os.path.exists(SD):
            os.makedirs(SD)
        with open(SBreg,'wb') as handle:
            pickle.dump(bandWidths,handle)