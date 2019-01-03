import numpy as np
from dss_python_funcs import *

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
    
    
def get_yzRegIdx(DSSCircuit):
    DSSEM = DSSCircuit.Meters
    # get transformers with regulators, YZ, n2y
    regXfmr = get_regXfmr(DSSCircuit)
    n2y = node_to_YZ(DSSCircuit)
    YZ = DSSCircuit.YNodeOrder

    zoneNames = []
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
        i=DSSEM.Next
    print(zoneList)
    regIdx = []
    for yzReg in yzRegIdx:
        regIdx = regIdx+yzReg
    QWE = sum(np.concatenate(yzRegIdx))
    chk = len(YZ)*((len(YZ)-1)//2)
    return zoneNames, regIdx