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

def getRegNms(DSSCircuit):
    regNms = {}
    i = DSSCircuit.RegControls.First
    while i:
        name = DSSCircuit.RegControls.Name
        DSSCircuit.SetActiveElement('Transformer.'+DSSCircuit.RegControls.Transformer)
        regNms[name]=DSSCircuit.ActiveElement.NumPhases
        i = DSSCircuit.RegControls.Next
    return regNms

def getRegSat(DSSCircuit):
    i = DSSCircuit.RegControls.First
    regSat = []
    while i:
        if (abs(DSSCircuit.RegControls.TapNumber))==16:
            regSat = regSat+[0]
        else:
            regSat = regSat+[1]
        i = DSSCircuit.RegControls.Next
    return regSat

def getRx(DSSCircuit): # see WB 15-01-19
    i = DSSCircuit.RegControls.First
    R = []
    X = []
    while i:
        R = R + [DSSCircuit.RegControls.ForwardR] # assume that all are just forward
        X = X + [DSSCircuit.RegControls.ForwardX]
        i = DSSCircuit.RegControls.Next
    return R,X

def getRegIcVr(DSSCircuit): # see WB 15-01-19
    i = DSSCircuit.RegControls.First
    Ic = []
    Vr = []
    while i:
        Ic = Ic + [DSSCircuit.RegControls.CTPrimary]
        Vr = Vr + [DSSCircuit.RegControls.ForwardVreg]
        i = DSSCircuit.RegControls.Next
    return Ic,Vr

def getRxVltsMat(DSSCircuit): # see WB 15-01-19
    i = DSSCircuit.RegControls.First
    R,X = getRx(DSSCircuit)
    Ic,Vr = getRegIcVr(DSSCircuit)
    rVltsMat = np.array(R)/(np.array(Ic)*np.array(Vr))
    xVltsMat = np.array(X)/(np.array(Ic)*np.array(Vr))
    
    return rVltsMat,xVltsMat
def setRx(DSSCircuit,R,X):
    i = DSSCircuit.RegControls.First
    while i:
        DSSCircuit.RegControls.ForwardR = R[i]
        DSSCircuit.RegControls.ReverseR = R[i]
        DSSCircuit.RegControls.ForwardX = X[i]
        DSSCircuit.RegControls.ReverseX = X[i]
        i = DSSCircuit.RegControls.Next
    return

def get_regIdx(DSSCircuit):
    regXfmrs=get_regXfmr(DSSCircuit) # needs to be in this order to match the index order of other fncs
    regBus = []
    regIdx = []
    for regXfmr in regXfmrs:
        DSSCircuit.SetActiveElement('Transformer.'+regXfmr)
        if DSSCircuit.ActiveElement.NumPhases==1:
            bus = DSSCircuit.ActiveElement.BusNames[1]
        elif DSSCircuit.ActiveElement.NumPhases==3: # WARNING: assume connected to .1 (true is bus='')
            if DSSCircuit.ActiveElement.BusNames[1].count('.')==4:
                bus = DSSCircuit.ActiveElement.BusNames[1].split('.',1)[0] + '.1'
            else:
                bus = DSSCircuit.ActiveElement.BusNames[1]+'.1'
        regBus.append(bus)
        node = bus.split('.')[0]
        regIdx = regIdx + find_node_idx(node_to_YZ(DSSCircuit),bus,False)
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
    
def kron_red_ltc(Ky,Kd,Kt,bV,Vreg,KvReg):
    n = len(Vreg)
    Kb = np.concatenate((Ky,Kd),axis=1)
    Abl = Kb[:-n]
    Arl = Kb[-n:] - KvReg
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
    Anew = np.concatenate((Anew,KvReg),axis=0)
    Bnew = np.concatenate((Bnew,Vreg),axis=0)
    
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
    # QWE = sum(np.concatenate(yzRegIdx))
    chk = len(YZ)*((len(YZ)-1)//2)
    
    return zoneList, regIdx, zoneTree

def getRegTrn(DSSCircuit,zoneTree):    # find the number of transformers per regulator
    regNms = getRegNms(DSSCircuit)
    regTrn = {} 
    for branch in zoneTree.values():
        for reg in branch:
            nPh = 0
            phs = []
            for regPh in regNms.keys():
                if reg in regPh:
                    nPh = nPh+1
                    if regPh[-1]=='a':
                        phs = phs+[1]
                    if regPh[-1]=='b':
                        phs = phs+[2]
                    if regPh[-1]=='c':
                        phs = phs+[3]
                    if regPh[-1]=='g':
                        phs = [1,2,3]
            regTrn[reg] = [nPh,phs]
    return regTrn
    
def getZoneSet(feeder,DSSCircuit,zoneTree):
    # to help create the right sets:
    regNms =  getRegNms(DSSCircuit) # number of phases in each regulator connected to, in the right order
    regTrn = getRegTrn(DSSCircuit,zoneTree) # number of individually controlled taps
    
    # It would be good to automate this at some point!
    if feeder=='13busRegModRx' or feeder=='13busRegMod3rg':
        zoneSet = {'msub':{},'mreg':{1:[0],2:[1],3:[2]},'mregx':{1:[0,3],2:[1,4],3:[2,5]},'mregy':{1:[0,6],2:[1,7],3:[2,8]}}
    elif feeder=='123busMod' or feeder=='123bus':
        zoneSet = {'msub':{},'mreg1g':{1:[0],2:[],3:[]},'mreg2a':{1:[0,1]},'mreg3':{1:[0,2],3:[3]},'mreg4':{1:[0,4],2:[5],3:[6]}}
    elif feeder=='13busModSng':
        zoneSet = {'msub':{},'mreg0':{1:[0],2:[0],3:[0]},'mregx':{1:[0,1],2:[0,2],3:[0,3]}}
    elif feeder=='34bus':
        zoneSet = {'msub':{1:[],2:[],3:[]},'mreg1':{1:[0],2:[0],3:[0]},'mreg2':{1:[0,3],2:[1,4],3:[2,5]}}
    elif feeder=='13busMod' or feeder=='13bus':
        zoneSet = {'msub':[],'mregs':{1:[0],2:[1],3:[2]}}
    elif feeder=='epriK1':
        zoneSet = {'msub':[],'mt2':{1:[0],2:[],3:[]}}
    elif feeder=='epriM1':
        zoneSet = {'msub':[],'m1_xfmr':{1:[0],2:[],3:[]}}
    else:
        print(feeder)
        print(regNms)
        print(regTrn)
        print(zoneTree,'\n\n')
    return zoneSet

def getZoneX(YZx,zoneList): # goes through each node and adds to the regulator list if it connected to that reg.
    zoneX = []
    for yz in YZx: # for each node
        ph = int(yz[-1]) # get the phase
        for key in zoneList: # for each key in the list of zones
            if yz in zoneList[key]: # if the node is in that zoneList then
                zoneX.append([key,ph]) # add to this list for that regulator
    return zoneX

def get_regIdxMatS(YZx,zoneList,zoneSet,Kp,Kq,nreg,delta):
    # Find all zones and phases of nodes to which regs are attached
    zoneX = getZoneX(YZx,zoneList)
    
    regIdxPx = np.zeros((nreg,len(YZx)))
    regIdxQx = np.zeros((nreg,len(YZx)))
    i=0
    for zone in zoneX:
        for key in zoneSet:
            if zone[0]==key:
                for regIdx in zoneSet[key][zone[1]]: 
                    regIdxPx[regIdx,i] = Kp[i]
                    regIdxQx[regIdx,i] = Kq[i]
                if delta:
                    for regIdx in zoneSet[key][zone[1]%3 + 1]:
                        regIdxPx[regIdx,i] = 1 - Kp[i]
                        regIdxQx[regIdx,i] = 1 - Kq[i]
        i+=1
    regIdxMatS = np.concatenate((regIdxPx,1j*regIdxQx),axis=1)
    return regIdxMatS