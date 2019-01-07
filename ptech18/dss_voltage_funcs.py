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
    in_regs = (type.lower()=='transformer' and (name.lower() in regXfmr))
    return in_regs

def getRegSat(DSSCircuit):
    i = DSSCircuit.RegControls.First
    regSat = []
    while i:
        if (DSSCircuit.RegControls.TapNumber)==16:
            regSat = regSat+[0]
        else:
            regSat = regSat+[1]
        i = DSSCircuit.RegControls.Next
    return regSat
    
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
    # for debugging:
    # YvbaseReg = get_Yvbase(DSSCircuit)[3:][v_idx_new][-n:]
    # dt = 0.1/16
    # YZreg=YZnew[-n:]
    # xt = spla.solve(Art,regVreg - Arl.dot(xh) - bVr)/(YvbaseReg*dt)
    
    return Anew, Bnew
    
def zB2zBs(DSSCircuit,zoneBus):
    YZ = DSSCircuit.YNodeOrder
    YZclr = {}
    for yz in YZ:
        node,ph = yz.split('.')
        if node in YZclr.keys():
            YZclr[node]=YZclr[node]+[ph]
        else:
            YZclr[node]=[ph]
    nodeIn = []
    zoneBuses = []
    for bus in zoneBus:
        node = bus.upper().split('.',1)[0]
        if node not in nodeIn:
            nodeIn = nodeIn+[node]
            for ph in YZclr[node]:
                zoneBuses = zoneBuses+[node+'.'+ph]
    return zoneBuses
        
        

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
    zoneBus = [] # for debugging
    zoneRegId = []
    i = DSSEM.First
    while i:
        zoneNames.append(DSSEM.name)
        zoneBus.append([])
        zoneIdx.append([])
        yzRegIdx.append([])
        for branch in DSSEM.AllBranchesInZone:
            DSSCircuit.SetActiveElement(branch)
            if in_regs(DSSCircuit,regXfmr):
                zoneRegId = zoneRegId + [DSSEM.name]
            else:
                for bus in DSSCircuit.ActiveElement.BusNames:
                    zoneBus[i-1].append(bus)
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
        
    zoneBuses = []
    for zb in zoneBus:
        zoneBuses.append(zB2zBs(DSSCircuit,zb))
    
    regUnq = []; zoneTree = {}; zoneList = {}; i=1
    for name in zoneNames:
        regUnq = []; j=0
        for reg in regXfmr:
            if reg[:-1] not in regUnq and name==zoneRegId[j]:
                regUnq = regUnq + [reg[:-1]]
            j+=1
        zoneTree[name] = regUnq
        zoneList[name] = zoneBuses[i-1]
        i+=1;
    regIdx = []
    for yzReg in yzRegIdx:
        regIdx = regIdx+yzReg
    QWE = sum(np.concatenate(yzRegIdx))
    chk = len(YZ)*((len(YZ)-1)//2)
    
    return zoneList, regIdx, zoneTree
    
    
    
    
def get_regIdxMatS(YZx,zoneList,Kp,Kq,nreg):
    zoneX = []
    for yz in YZx:
        ph = int(yz[-1])
        for key in zoneList.keys():
            if yz in zoneList[key]:
                zoneX.append([key,ph])
    
    regIdxPx = np.zeros((nreg,len(YZx)))
    regIdxQx = np.zeros((nreg,len(YZx)))
    i=0
    for zone in zoneX:
        if zone[0]=='msub':
            qwe = 0
        elif zone[0]=='mreg':
            regIdxPx[zone[1]-1,i]=Kp[i]
            regIdxPx[(zone[1]%3),i]=1-Kp[i]
            regIdxQx[zone[1]-1,i]=Kq[i]
            regIdxQx[(zone[1]%3),i]=1-Kq[i]
            
        elif zone[0]=='mregx':
            regIdxPx[zone[1]-1,i]=Kp[i]
            regIdxPx[(zone[1]%3),i]=1-Kp[i]
            regIdxPx[zone[1]-1+3,i]=Kp[i]
            regIdxPx[(zone[1]%3)+3,i]=1-Kp[i]
            regIdxQx[zone[1]-1,i]=Kq[i]
            regIdxQx[(zone[1]%3),i]=1-Kq[i]
            regIdxQx[zone[1]-1+3,i]=Kq[i]
            regIdxQx[(zone[1]%3)+3,i]=1-Kq[i]
            
        elif zone[0]=='mregy':
            regIdxPx[zone[1]-1,i]=Kp[i]
            regIdxPx[zone[1]%3,i]=1-Kp[i]
            regIdxPx[zone[1]-1+6,i]=Kp[i]
            regIdxPx[(zone[1]%3)+6,i]=1-Kp[i]
            
            regIdxQx[zone[1]-1,i]=Kq[i]
            regIdxQx[zone[1]%3,i]=1-Kq[i]
            regIdxQx[zone[1]-1+6,i]=Kq[i]
            regIdxQx[(zone[1]%3)+6,i]=1-Kq[i]
        i+=1
        
    regIdxMatS = np.concatenate((regIdxPx,1j*regIdxQx),axis=1)
    return regIdxMatS