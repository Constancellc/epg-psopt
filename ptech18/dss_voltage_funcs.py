import numpy as np
from dss_python_funcs import *
import scipy.linalg as spla

def get_regXfmr(DSSCircuit):
    i = DSSCircuit.RegControls.First
    regXfmr = []
    while i:
        regXfmr.append(DSSCircuit.RegControls.Transformer)
        i = DSSCircuit.RegControls.Next
    return regXfmr
    
def in_regs(DSSCircuit,regXfmr):
    type,name = DSSCircuit.ActiveElement.name.split('.')
    in_regs = (type=='Transformer' and (name in regXfmr))
    return in_regs
    
def get_regIdx(DSSCircuit):
    regXfmr=get_regXfmr(DSSCircuit)
    regBus = []
    regIdx = []
    i = DSSCircuit.Transformers.First
    while i:
        if in_regs(DSSCircuit,regXfmr):
            bus = DSSCircuit.ActiveElement.BusNames[1]
            regBus.append(bus)
            node = bus.split('.')[0]
            regIdx = regIdx + find_node_idx(node_to_YZ(DSSCircuit),bus,False)
        i = DSSCircuit.Transformers.Next
    return regIdx

def get_reIdx(regIdx,n):
    reIdx = []
    for i in range(n):
        if i not in regIdx:
            reIdx = reIdx + [i]
    reIdx = reIdx+regIdx
    return reIdx

def get_regVreg(DSSCircuit):
    i = DSSCircuit.RegControls.First
    regVreg = []
    while i:
        regVreg = regVreg + [DSSCircuit.RegControls.ForwardVreg*DSSCircuit.RegControls.PTratio]
        i = DSSCircuit.RegControls.Next
    return regVreg
    

def kron_red(Ky,Kd,Kt,bV,Vreg):
    n = len(Vreg)
    Kb = np.concatenate((Ky,Kd),axis=1)
    Abl = Kb[:-n]
    Arl = Kb[-n:]
    Abt = Kt[:-n]
    Art = Kt[-n:]
    bVb = bV[:-n]
    bVr = bV[-n:]
    Anew = Abl - Abt.dot(spla.solve(Art,Arl))
    Bnew = bVb + Abt.dot(spla.solve(Art,(Vreg - bVr)))
    return Anew, Bnew
    
    
def get_regZneIdx(DSSCircuit):
    DSSEM = DSSCircuit.Meters
    # get transformers with regulators, YZ, n2y
    regXfmr = get_regXfmr(DSSCircuit)
    n2y = node_to_YZ(DSSCircuit)
    YZ = DSSCircuit.YNodeOrder

    zoneNames = []
    regSze = []
    yzRegIdx = [] # for debugging
    zoneIdx = [] # for debugging
    zoneList = [] # for debugging
    i = DSSEM.First
    while i:
        zoneNames.append(DSSEM.name)
        zoneList.append([])
        zoneIdx.append([])
        yzRegIdx.append([])
        for branch in DSSEM.AllBranchesInZone:
            DSSCircuit.SetActiveElement(branch)
            if in_regs(DSSCircuit,regXfmr)==False:
                for bus in DSSCircuit.ActiveElement.BusNames:
                    zoneList[i-1].append(bus)
                    if bus.count('.')==0:
                        idx = find_node_idx(n2y,bus,False)
                        zoneIdx[i-1].append(idx)
                        for no in idx:
                            if (no in yzRegIdx[i-1])==False:
                                yzRegIdx[i-1].append(no)
                    else:
                        node = bus.split('.')[0]
                        phs = bus.split('.')[1:]
                        for ph in phs:
                            idx = find_node_idx(n2y,node+'.'+ph,False)
                            zoneIdx[i-1].append(idx)
                            for no in idx:
                                if (no in yzRegIdx[i-1])==False:
                                    yzRegIdx[i-1].append(no)
        yzRegIdx[i-1].sort()
        regSze.append(len(yzRegIdx[i-1]))
        i=DSSEM.Next
    print(zoneList)
    regIdx = []
    for yzReg in yzRegIdx:
        regIdx = regIdx+yzReg
    QWE = sum(np.concatenate(yzRegIdx))
    chk = len(YZ)*((len(YZ)-1)//2)
    
    return zoneNames, regIdx, regSze