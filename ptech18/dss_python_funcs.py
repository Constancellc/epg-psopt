import numpy as np
import os, time, win32com.client
from scipy import sparse
from cvxopt import matrix
import matplotlib.pyplot as plt

from win32com.client import makepy

def get_regXfmr2(DSSCircuit): # copy of get_regXfmr from dss_voltage_funcs
    i = DSSCircuit.RegControls.First
    regXfmr = []
    while i:
        regXfmr.append(DSSCircuit.RegControls.Transformer)
        i = DSSCircuit.RegControls.Next
    return regXfmr


def loadDss():
    sys.argv=["makepy","OpenDSSEngine.DSS"]
    makepy.main()
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
    return DSSObj,DSSObj.Text,DSSObj.ActiveCircuit,DSSObj.ActiveCircuit.Solution
    # [DSSObj, DSSText, DSSCircuit,DSSSolution] = loadDss()

def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

def tp2mat(tuple_ex):
    n = int(np.sqrt(len(tuple_ex)))
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i] = tuple_ex[i*n:(i+1)*n]
    
    return mat
    

def s_2_x(s):
    return np.concatenate((s.real,s.imag))

def vecSlc(vec_like,new_idx):
    if len(new_idx)==0:
        if type(vec_like)==tuple:
            vec_slc = ()
        elif type(vec_like)==list:
            vec_slc = []
    else:
        if type(vec_like)==tuple:
            vec_slc = tuple(np.array(vec_like)[new_idx].tolist())
        elif type(vec_like)==list:
            vec_slc = np.array(vec_like)[new_idx].tolist()
    return vec_slc

def yzD2yzI(yzD,n2y):
    yzI = []
    for bus in yzD:
        yzI = yzI+find_node_idx(n2y,bus,False)
    return yzI

def idx_shf(x_idx,reIdx):
    x_idx_i = []
    for idx in x_idx:
        x_idx_i.append(reIdx.index(idx))
    
    x_idx_new = np.array([],dtype=int)
    
    x_idx_srt = x_idx_i.copy()
    x_idx_srt.sort()
    x_idx_shf = np.array([],dtype=int)
    for i in x_idx_srt:
        x_idx_shf=np.concatenate((x_idx_shf,[x_idx_i.index(i)]))
        x_idx_new=np.concatenate((x_idx_new,[reIdx[i]]))
    
    return x_idx_shf,x_idx_new

def add_generators(DSSObj,genBuses,delta):
    # NB: nominal power is 0.5 kW.
    genNames = []
    for genBus in genBuses:
        DSSObj.ActiveCircuit.SetActiveBus(genBus)
        if not delta: # ie wye
            genName = genBus.replace('.','_')
            genKV = str(DSSObj.ActiveCircuit.ActiveBus.kVBase)
            DSSObj.Text.Command='new generator.'+genName+' phases=1 bus1='+genBus+' kV='+genKV+' kW=0.5 pf=1.0 model=1 vminpu=0.33 vmaxpu=3.0 conn=wye'
        elif delta:
            genKV = str(DSSObj.ActiveCircuit.ActiveBus.kVBase*np.sqrt(3))
            if genBus[-1]=='1':
                genBuses = genBus+'.2'
            if genBus[-1]=='2':
                genBuses = genBus+'.3'
            if genBus[-1]=='3':
                genBuses = genBus+'.1'
            genName = genBuses.replace('.','_')
            DSSObj.Text.Command='new generator.'+genName+' phases=1 bus1='+genBuses+' kV='+genKV+' kW=0.5 pf=1.0 model=1 vminpu=0.33 vmaxpu=3.0 conn=wye'
        genNames = genNames+[genName]
    return genNames

def getTotkW(DSSCircuit):
    GEN = DSSCircuit.Generators
    i = GEN.First
    kwGen = 0
    while i:
        kwGen = kwGen + GEN.kW
        i = GEN.Next
    return kwGen
    
def runCircuit(DSSCircuit,DSSSolution):
    # NB assumes all generators are constant power.
    DSSSolution.Solve()
    TG = getTotkW(DSSCircuit)
    TP = -DSSCircuit.TotalPower[0]
    TL = 1e-3*DSSCircuit.Losses[0]
    PL = -(DSSCircuit.TotalPower[0] + 1e-3*DSSCircuit.Losses[0] - TG)
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    return TP,TG,TL,PL,YNodeV
    

def set_generators(DSSCircuit,genNames,P):
    i = 0
    for genName in genNames:
        DSSCircuit.Generators.Name=genName
        DSSCircuit.Generators.kW = P[i]
        i+=1
    return

def setGenPq(DSSCircuit,genNames,P,Q):
    i = 0
    for genName in genNames:
        DSSCircuit.Generators.Name=genName
        DSSCircuit.Generators.kW = P[i]
        DSSCircuit.Generators.kvar = Q[i]
        i+=1
    return

def ld_vals( DSSCircuit ):
    ii = DSSCircuit.FirstPCElement()
    S=[]; V=[]; I=[]; B=[]; D=[]; N=[]
    while ii!=0:
        if DSSCircuit.ActiveElement.Name[0:4].lower()=='load':
            DSSCircuit.Loads.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            N.append(DSSCircuit.Loads.Name)
            if B[-1][0].count('.')==1:
                D.append(False)
            else:
                D.append(DSSCircuit.Loads.IsDelta)
        ii=DSSCircuit.NextPCElement()
    jj = DSSCircuit.FirstPDElement()
    while jj!=0:
        if DSSCircuit.ActiveElement.Name[0:4].lower()=='capa':
            DSSCircuit.Capacitors.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            D.append(DSSCircuit.Capacitors.IsDelta)
            N.append(DSSCircuit.Capacitors.Name)
        jj=DSSCircuit.NextPDElement()
    return S,V,I,B,D,N

def ldValsOnly( DSSCircuit ):
    # nicked from the oxemf converter
    LDS = DSSCircuit.Loads
    SetACE = DSSCircuit.SetActiveElement
    ACE = DSSCircuit.ActiveElement
    i = LDS.First
    YZ = DSSCircuit.YNodeOrder[3:]
    
    sY = np.zeros(len(YZ),dtype=complex)
    sD = np.zeros(len(YZ),dtype=complex)
    while i:
        SetACE('Load.'+LDS.Name)
        actBus = ACE.BusNames[0].split('.')[0].upper()
        nPh = ACE.NumPhases
        phs = ACE.BusNames[0].split('.')[1:]
        
        if LDS.IsDelta:
            if nPh==1:
                if len(phs)==2:
                    if '1' in phs and '2' in phs:
                        dIdx = YZ.index(actBus+'.1')
                    if '2' in phs and '3' in phs:
                        dIdx = YZ.index(actBus+'.2')
                    if '3' in phs and '1' in phs:
                        dIdx = YZ.index(actBus+'.3')
                    sD[dIdx] = sD[dIdx] + LDS.kW + 1j*LDS.kvar
                if len(phs)==1:
                    yIdx = YZ.index(actBus+'.'+phs[0])
                    sY[yIdx] = sY[yIdx] + (LDS.kW + 1j*LDS.kvar) # if only one phase, then is effectively D!
            if nPh==3:
                for i in range(3):
                    dIdx = YZ.index(actBus+'.'+str(i+1))
                    sD[dIdx] = sD[dIdx] + (LDS.kW + 1j*LDS.kvar)/3
            if nPh==2:
                print('Warning! Load: ',LDS.Name,'2 phase Delta loads not yet implemented.')
        else:
            if '1' in phs or phs==[]:
                yIdx = YZ.index(actBus+'.1')
                sY[yIdx] = sY[yIdx] + (LDS.kW + 1j*LDS.kvar)/nPh
            if '2' in phs or phs==[]:
                yIdx = YZ.index(actBus+'.2')
                sY[yIdx] = sY[yIdx] + (LDS.kW + 1j*LDS.kvar)/nPh
            if '3' in phs or phs==[]:
                yIdx = YZ.index(actBus+'.3')
                sY[yIdx] = sY[yIdx] + (LDS.kW + 1j*LDS.kvar)/nPh
        i = LDS.Next
    sYidx = sY.nonzero()
    sDidx = sD.nonzero()
    # xY = -1e3*np.concatenate([sY[sYidx].real,sY[sYidx].imag])
    xY = -1e3*np.concatenate([sY.real,sY.imag])
    xD = -1e3*np.concatenate([sD[sDidx].real,sD[sDidx].imag])
    
    return xY, xD, sYidx, sDidx

def find_node_idx(n2y,bus,D):
    idx = []
    BS = bus.split('.',1)
    bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
    if ph=='1.2.3' or bus.count('.')==0:
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
    elif ph=='0.0.0' or ph=='0': # nb second part experimental for single phase caps
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
    elif ph=='1.2.3.4': # needed if, e.g. transformers are grounded through a reactor
        idx.append(n2y.get(bus_id+'.1',None))
        idx.append(n2y.get(bus_id+'.2',None))
        idx.append(n2y.get(bus_id+'.3',None))
        idx.append(n2y.get(bus_id+'.4',None))        
    elif D:
        if bus.count('.')==1:
            idx.append(n2y[bus])
        else:
            idx.append(n2y[bus[0:-2]])
    else:
        idx.append(n2y[bus])
    return idx
    
def calc_sYsD( YZ,B,I,V,S,D,n2y ): # YZ as YNodeOrder
    iD = np.zeros(len(YZ),dtype=complex); sD = np.zeros(len(YZ),dtype=complex)
    iY = np.zeros(len(YZ),dtype=complex); sY = np.zeros(len(YZ),dtype=complex)
    for i in range(len(B)):
        for bus in B[i]:
            idx = find_node_idx(n2y,bus,D[i])
            BS = bus.split('.',1)
            bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
            if D[i]:
                if bus.count('.')==2:
                    iD[idx] = iD[idx] + I[i][0]
                    sD[idx] = sD[idx] + S[i].sum()
                else:
                    iD[idx] = iD[idx] + I[i]*np.exp(1j*np.pi/6)/np.sqrt(3)
                    VX = np.array( [V[i][0]-V[i][1],V[i][1]-V[i][2],V[i][2]-V[i][0]] )
                    sD[idx] = sD[idx] + iD[idx].conj()*VX*1e-3
            else:
                if ph[0]!='0':
                    if bus.count('.')>0:
                        iY[idx] = iY[idx] + I[i][0]
                        sY[idx] = sY[idx] + S[i][0]
                    else:
                        iY[idx] = iY[idx] + I[i][0:3]
                        sY[idx] = sY[idx] + S[i][0:3]
    return iY, sY, iD, sD

def node_to_YZ(DSSCircuit):
    n2y = {}
    YNodeOrder = DSSCircuit.YNodeOrder
    for node in DSSCircuit.AllNodeNames:
        n2y[node]=YNodeOrder.index(node.upper())
    return n2y

def get_sYsD(DSSCircuit):
    S,V,I,B,D,N = ld_vals( DSSCircuit )
    n2y = node_to_YZ(DSSCircuit)
    V0 = tp_2_ar(DSSCircuit.YNodeVarray)*1e-3 # kV
    YZ = DSSCircuit.YNodeOrder
    iY, sY, iD, sD = calc_sYsD( YZ,B,I,V,S,D,n2y )
    H = create_Hmat(DSSCircuit)
    H = H[iD.nonzero()]
    sD = sD[iD.nonzero()]
    yzD = [YZ[i] for i in iD.nonzero()[0]]
    iD = iD[iD.nonzero()]
    iTot = iY + (H.T).dot(iD)
    # chka = abs((H.T).dot(iD.conj())*V0 + sY - V0*(iTot.conj()))/abs(sY) # 1a error, kW
    # sD0 = ((H.dot(V0))*(iD.conj()))
    # chkb = abs(sD - sD0)/abs(sD) # 1b error, kW
    # print('Y- error:')
    # print_node_array(YZ,abs(chka))
    # print('D- error:')
    # print_node_array(yzD,abs(chkb))
    return sY,sD,iY,iD,yzD,iTot,H
    
def returnXyXd(DSSCircuit,n2y):
    S,V,I,B,D,N = ld_vals( DSSCircuit )
    V0 = tp_2_ar(DSSCircuit.YNodeVarray)*1e-3 # kV
    YZ = DSSCircuit.YNodeOrder
    iY, sY, iD, sD = calc_sYsD( YZ,B,I,V,S,D,n2y )
    sD = sD[iD.nonzero()]
    xY = -1e3*s_2_x(sY[3:])
    xD = -1e3*s_2_x(sD)
    return xY,xD
    
def create_Hmat(DSSCircuit):
    n2y = node_to_YZ(DSSCircuit)
    Hmat = np.zeros((DSSCircuit.NumNodes,DSSCircuit.NumNodes))
    for bus in DSSCircuit.AllBusNames:
        idx = find_node_idx(n2y,bus,False)
        if idx[0]!=None and idx[1]!=None:
            Hmat[idx[0],idx[0]] = 1
            Hmat[idx[0],idx[1]] = -1
        if idx[1]!=None and idx[2]!=None:
            Hmat[idx[1],idx[1]] = 1
            Hmat[idx[1],idx[2]] = -1
        if idx[2]!=None and idx[0]!=None:
            Hmat[idx[2],idx[2]] = 1
            Hmat[idx[2],idx[0]] = -1        
    return Hmat
    
def cpf_get_loads(DSSCircuit,getCaps=True):
    SS = {}
    BB = {}
    i = DSSCircuit.Loads.First
    while i!=0:
        SS[i]=DSSCircuit.Loads.kW + 1j*DSSCircuit.Loads.kvar
        BB[i]=DSSCircuit.Loads.Name
        i=DSSCircuit.Loads.Next
    imax = DSSCircuit.Loads.Count
    if getCaps:
        j = DSSCircuit.Capacitors.First
        while j!=0:
            SS[imax+j]=1j*DSSCircuit.Capacitors.kvar
            BB[imax+j]=DSSCircuit.Capacitors.Name
            j = DSSCircuit.Capacitors.Next
    return BB,SS

def cpf_set_loads(DSSCircuit,BB,SS,k,setCaps=True,capPos=None):
    i = DSSCircuit.Loads.First
    while i!=0:
        # DSSCircuit.Loads.Name=BB[i]
        DSSCircuit.Loads.kW = k*SS[i].real
        DSSCircuit.Loads.kvar = k*SS[i].imag
        i=DSSCircuit.Loads.Next
    imax = DSSCircuit.Loads.Count
    if setCaps:
        if setCaps==True:
            if capPos==None:
                capPos = [k]*DSSCircuit.Capacitors.Count
            else:
                capPos = (k*np.array(capPos)).tolist()
        elif setCaps=='linCaps':
            if capPos==None:
                capPos = [1]*DSSCircuit.Capacitors.Count
            # otherwise use capPos
        j = DSSCircuit.Capacitors.First
        while j!=0:
            DSSCircuit.Capacitors.Name=BB[j+imax]
            DSSCircuit.Capacitors.kvar=capPos[j-1]*SS[j+imax].imag + 1e-4 # so that the # of caps doesn't change...
            j=DSSCircuit.Capacitors.Next
            
    return

def find_tap_pos(DSSCircuit):
    TC_No=[]
    i = DSSCircuit.RegControls.First
    while i!=0:
        TC_No.append(DSSCircuit.RegControls.TapNumber)
        i = DSSCircuit.RegControls.Next
    return TC_No

def fix_tap_pos(DSSCircuit, TC_No):
    i = DSSCircuit.RegControls.First
    while i!=0:
        DSSCircuit.RegControls.TapNumber = TC_No[i-1]
        i = DSSCircuit.RegControls.Next

def fix_cap_pos(DSSCircuit, CP_No):
    # warning: side effect disables capacitors not in CP_No from here.
    i = DSSCircuit.Capacitors.First
    while i!=0:
        DSSCircuit.SetActiveElement('Capacitor.'+DSSCircuit.Capacitors.Name)
        DSSCircuit.ActiveElement.Enabled=CP_No[i-1]
        i = DSSCircuit.Capacitors.Next

def createYbus( DSSObj,tapNo,capNo ):
    # DSSObj.Text.Command='Compile ('+fn+')'
    ctrlModel = DSSObj.ActiveCircuit.Solution.ControlMode
    print('Load Ybus\n',time.process_time())
    fix_tap_pos(DSSObj.ActiveCircuit, tapNo)
    fix_cap_pos(DSSObj.ActiveCircuit, capNo)
    DSSObj.Text.Command='set controlmode=off'
    DSSObj.Text.Command='vsource.source.enabled=no';
    DSSObj.Text.Command='batchedit load..* enabled=no';
    DSSObj.ActiveCircuit.Solution.Solve()
    
    SysY = DSSObj.ActiveCircuit.SystemY
    SysY_dct = {}
    i = 0
    for i in range(len(SysY)):
        if i%2 == 0:
            Yi = SysY[i] + 1j*SysY[i+1]
            if abs(Yi)!=0.0:
                j = i//2
                SysY_dct[j] = Yi
    del SysY
    
    SysYV = np.array(list(SysY_dct.values()))
    SysYK = np.array(list(SysY_dct.keys()))
    Ybus0 = sparse.coo_matrix((SysYV,(SysYK,np.zeros(len(SysY_dct),dtype=int))))
    n = int(np.sqrt(Ybus0.shape[0]))
    Ybus = Ybus0.reshape((n,n))
    Ybus = Ybus.tocsc()

    YNodeOrder = DSSObj.ActiveCircuit.YNodeOrder
    
    DSSObj.ActiveCircuit.Solution.ControlMode = ctrlModel # return control to nominal state
    print('Ybus loaded\n',time.process_time())
    return Ybus, YNodeOrder


def create_tapped_ybus_very_slow( DSSObj,fn_y,TC_No0 ):
    DSSObj.Text.Command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.Command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    
    SysY = DSSObj.ActiveCircuit.SystemY
    SysY_dct = {}
    i = 0
    for i in range(len(SysY)):
        if i%2 == 0:
            Yi = SysY[i] + 1j*SysY[i+1]
            if abs(Yi)!=0.0:
                j = i//2
                SysY_dct[j] = Yi
    del SysY
    
    SysYV = np.array(list(SysY_dct.values()))
    SysYK = np.array(list(SysY_dct.keys()))
    Ybus0 = sparse.coo_matrix((SysYV,(SysYK,np.zeros(len(SysY_dct),dtype=int))))
    n = int(np.sqrt(Ybus0.shape[0]))
    Ybus_ = Ybus0.reshape((n,n))
    Ybus_ = Ybus_.tocsc()
    
    Ybus = Ybus_[3:,3:]
    YNodeOrder_ = DSSObj.ActiveCircuit.YNodeOrder
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder
        
        
def create_tapped_ybus_slow( DSSObj,fn_y,TC_No0 ):
    DSSObj.Text.Command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.Command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    Ybus0 = tp_2_ar(DSSObj.ActiveCircuit.SystemY)
    n = int(np.sqrt(len(Ybus0)))
    Ybus_ = Ybus0.reshape((n,n))
    Ybus = Ybus_[3:,3:]
    YNodeOrder_ = DSSObj.ActiveCircuit.YNodeOrder
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 ):
    DSSObj.Text.Command='Compile ('+fn_y+')'
    fix_tap_pos(DSSObj.ActiveCircuit, TC_No0)
    DSSObj.Text.Command='set controlmode=off'
    DSSObj.ActiveCircuit.Solution.Solve()
    Ybus_,YNodeOrder_,n = build_y(DSSObj,fn_ckt)
    Ybus = Ybus_[3:,3:]
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def build_y(DSSObj,fn_ckt):
    # DSSObj.Text.Command='Compile ('+fn_z+'.dss)'
    YNodeOrder = DSSObj.ActiveCircuit.YNodeOrder
    DSSObj.Text.Command='show Y'
    os.system("TASKKILL /F /IM notepad.exe")
    
    fn_y = fn_ckt+'\\'+DSSObj.ActiveCircuit.Name+'_SystemY.txt'
    fn_csv = fn_ckt+'\\'+DSSObj.ActiveCircuit.Name+'_SystemY_csv.txt'

    file_r = open(fn_y,'r')
    stream = file_r.read()

    stream=stream.replace('[','')
    stream=stream.replace('] = ',',')
    stream=stream.replace('j','')
    stream=stream[89:]
    stream=stream.replace('\n','j\n')
    stream=stream.replace(' ','')

    file_w = open(fn_csv,'w')
    file_w.write(stream)

    file_r.close()
    file_w.close()

    rc_data = np.loadtxt(fn_csv,delimiter=',',dtype=complex)
    n_y = int(rc_data[-1,0].real)

    I = np.concatenate((rc_data[:,0]-1,rc_data[:,1]-1)).real
    J = np.concatenate((rc_data[:,1]-1,rc_data[:,0]-1)).real
    V = np.concatenate((rc_data[:,2],rc_data[:,2]))
    Ybus = sparse.coo_matrix((V,(I,J)),shape=(n_y,n_y),dtype=complex).tocsr()
    
    n = len(YNodeOrder)
    for i in range(n):
        Ybus[i,i] = Ybus[i,i]/2
    
    os.remove(fn_y)
    os.remove(fn_csv)
    return Ybus, YNodeOrder, n

            # splt = DSSCircuit.ActiveElement.BusNames[0].upper().split('.')
def get_idxs(e_idx,DSSCircuit,ELE):
    i = ELE.First
    while i:
        for BN in DSSCircuit.ActiveElement.BusNames:
            splt = BN.upper().split('.')
            if len(splt) > 1:
                for j in range(1,len(splt)):
                    if splt[j]!='0': # ignore ground
                        e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]+'.'+splt[j]))
            else:
                try:
                    e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]))
                except:
                    for ph in range(1,4):
                        e_idx.append(DSSCircuit.YNodeOrder.index(splt[0]+'.'+str(ph)))
        i = ELE.Next
    return e_idx

def get_element_idxs(DSSCircuit,ele_types):
    e_idx = []
    for ELE in ele_types:
        e_idx = get_idxs(e_idx,DSSCircuit,ELE)
    return e_idx

def get_Yvbase(DSSCircuit):
    Yvbase = []
    for yz in DSSCircuit.YNodeOrder:
        bus_id = yz.split('.')
        i = DSSCircuit.SetActiveBus(bus_id[0]) # return needed or this prints a number
        Yvbase.append(1e3*DSSCircuit.ActiveBus.kVBase)
    return np.array(Yvbase)
    
def feeder_to_fn(WD,feeder):
    paths = []
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1])
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1]+'\master')
    return paths
    
def print_node_array(YZ,thing):
    for i in range(len(YZ)):
        print(YZ[i]+': '+str(thing[i]))

def get_ckt(WD,feeder):
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr',feeder]
    # fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr'] # for editing
    ckts = {'feeder_name':['fn_ckt','fn']}
    ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
    ckts[fdrs[1]]=feeder_to_fn(WD,fdrs[1])
    ckts[fdrs[2]]=feeder_to_fn(WD,fdrs[2])
    ckts[fdrs[3]]=feeder_to_fn(WD,fdrs[3])
    ckts[fdrs[4]]=feeder_to_fn(WD,fdrs[4])
    ckts[fdrs[5]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_z']
    ckts[fdrs[6]]=[WD+'\\ieee_tn\\34Bus_copy',WD+'\\ieee_tn\\34Bus_copy\\ieee34Mod1_z']
    ckts[fdrs[7]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37_z']
    ckts[fdrs[8]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123Master_z']
    ckts[fdrs[9]]=[WD+'\\ieee_tn\\8500-Node_copy',WD+'\\ieee_tn\\8500-Node_copy\\Master-unbal_z']
    ckts[fdrs[10]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37_z_mod']
    ckts[fdrs[11]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regMod3rg_z']
    ckts[fdrs[12]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regModRx_z']
    ckts[fdrs[13]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_regModSng_z']
    ckts[fdrs[14]]=[WD+'\\ieee_tn\\usLv',WD+'\\ieee_tn\\usLv\\master_z']
    ckts[fdrs[15]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123MasterMod_z']
    ckts[fdrs[16]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13NodecktMod_z']
    ckts[fdrs[17]]=[WD+'\\ieee_tn\\ckt5',WD+'\\ieee_tn\\ckt5\\Master_ckt5_z']
    ckts[fdrs[18]]=[WD+'\\ieee_tn\\ckt7',WD+'\\ieee_tn\\ckt7\\Master_ckt7_z']
    ckts[fdrs[19]]=[WD+'\\ieee_tn\\j1',WD+'\\ieee_tn\\j1\\Master_noPV_z']
    ckts[fdrs[20]]=[WD+'\\ieee_tn\\k1',WD+'\\ieee_tn\\k1\\Master_NoPV_z']
    ckts[fdrs[21]]=[WD+'\\ieee_tn\\m1',WD+'\\ieee_tn\\m1\\Master_NoPV_z']
    ckts[fdrs[22]]=[WD+'\\ieee_tn\\ckt24',WD+'\\ieee_tn\\ckt24\\master_ckt24_z']
    ckts[fdrs[23]]=[WD+'\\ieee_tn\\4Bus-YY-Bal',WD+'\\ieee_tn\\4Bus-YY-Bal\\4Bus-YY-Bal_z']
    ckts[fdrs[24]]=[WD+'\\ieee_tn\\k1',WD+'\\ieee_tn\\k1\\Master_NoPV_z_cvr']
    ckts[fdrs[25]]=[WD+'\\ieee_tn\\ckt24',WD+'\\ieee_tn\\ckt24\\master_ckt24_z_cvr']
    ckts[fdrs[26]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123Master_cvr']
    
    if not feeder in ckts.keys():
        if feeder[0]=='n' and int(feeder[1:])<26:
            dir0 = WD+'\\manchester_models\\batch_manc_ntwx\\network_'+feeder[1:]
            ckts[fdrs[-1]] = [dir0,dir0+'\\masterNetwork'+feeder[1:]]
        elif feeder[0]=='n' and feeder[1:]=='26':
                dir0 = WD+'\\manchester_models\\batch_manc_ntwx\\network_7'
                ckts[fdrs[-1]] = [dir0,dir0+'\\masterNetwork7feeder1']
        elif feeder[0]=='n' and feeder[1:]=='27':
                dir0 = WD+'\\manchester_models\\batch_manc_ntwx\\network_27'
                ckts[fdrs[-1]] = [dir0,dir0+'\\masterNetwork1']
        elif len(feeder)==3:
            dir0 = WD+'\\manchester_models\\batch_manc_ntwx\\network_'+str(int(feeder[0:2]))+'\\Feeder_'+feeder[-1]
            ckts[fdrs[-1]]=[dir0,dir0+'\\Master']
    return ckts[feeder]

def loadLinMagModel(feeder,lin_point,WD,lp_taps,regModel=True):
    # lp_taps either 'Nmt' or 'Lpt'.
    stt = WD+'\\lin_models\\'+feeder+'\\'+feeder+lp_taps
    end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
    LM = {}
    LM['Ky'] = np.load(stt+'Ky'+end)
    
    LM['bV'] = np.load(stt+'bV'+end)
    LM['xhy0'] = np.load(stt+'xhy0'+end)
    LM['xhyCap0'] = np.load(stt+'xhyCap0'+end)
    LM['xhyLds0'] = np.load(stt+'xhyLds0'+end)
    LM['vKvbase'] = np.load(stt+'vKvbase'+end)
    LM['vYNodeOrder'] = np.load(stt+'vYNodeOrder'+end)
    LM['SyYNodeOrder'] = np.load(stt+'SyYNodeOrder'+end)
    LM['v_idx'] = np.load(stt+'v_idx'+end)
    try:
        LM['Kd'] = np.load(stt+'Kd'+end)
        LM['xhd0'] = np.load(stt+'xhd0'+end)
        LM['xhdCap0'] = np.load(stt+'xhdCap0'+end)
        LM['xhdLds0'] = np.load(stt+'xhdLds0'+end)
        LM['SdYNodeOrder'] = np.load(stt+'SdYNodeOrder'+end)
    except:
        LM['Kd'] = np.empty(shape=(LM['Ky'].shape[0],0))
        LM['xhd0'] = np.array([])
        LM['xhdCap0'] = np.array([])
        LM['xhdLds0'] = np.array([])
        LM['SdYNodeOrder'] = np.array([])
    try: 
        LM['Kt'] = np.load(stt+'Kt'+end)
    except:
        LM['Kt'] = np.empty(shape=(LM['Ky'].shape[0],0))
        
    if regModel:
        LM['WyReg'] = np.load(stt+'WyReg'+end)
        LM['WregBus'] = np.load(stt+'WregBus'+end)
        # LM['WtReg'] = np.load(stt+'MtReg'+end)
        LM['WtReg'] = np.load(stt+'WtReg'+end)
        LM['aIreg'] = np.load(stt+'aIreg'+end)
        try:
            LM['WdReg'] = np.load(stt+'WdReg'+end)
        except:
            LM['WdReg'] = np.empty(shape=(LM['WtReg'].shape[0],0))
    return LM
    
def loadLtcModel(feeder,lin_point,WD,lp_taps):
    # lp_taps either 'Nmt' or 'Lpt'.
    stt = WD+'\\lin_models\\'+feeder+'\\ltc_model\\'+feeder+lp_taps+'Ltc'
    end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
    LM = {}
    LM['A'] = np.load(stt+'A'+end)
    LM['B'] = np.load(stt+'B'+end)
    # LM['s_idx'] = np.load(stt+'s_idx'+end)
    LM['v_idx'] = np.load(stt+'v_idx'+end)
    LM['Vbase'] = np.load(stt+'Vbase'+end)
    LM['xhy0'] = np.load(stt+'xhy0'+end)
    LM['xhd0'] = np.load(stt+'xhd0'+end)
    LM['vYNodeOrder'] = np.load(stt+'vYNodeOrder'+end)
    LM['SyYNodeOrder'] = np.load(stt+'SyYNodeOrder'+end)
    LM['SdYNodeOrder'] = np.load(stt+'SdYNodeOrder'+end)
    return LM

def loadNetModel(feeder,lin_point,WD,lp_taps,netModel):
    # lp_taps either 'Nmt' or 'Lpt'.
    if netModel==1:
        stt = WD+'\\lin_models\\'+feeder+'\\ltc_model\\'+feeder+lp_taps+'Ltc'
    if netModel==2:
        stt = WD+'\\lin_models\\'+feeder+'\\fxd_model\\'+feeder+lp_taps+'Fxd'
    end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
    LM = {}
    LM['A'] = np.load(stt+'A'+end)
    LM['B'] = np.load(stt+'B'+end)
    LM['s_idx'] = np.load(stt+'s_idx'+end)
    LM['v_idx'] = np.load(stt+'v_idx'+end)
    LM['Vbase'] = np.load(stt+'Vbase'+end)
    LM['xhy0'] = np.load(stt+'xhy0'+end)
    LM['xhd0'] = np.load(stt+'xhd0'+end)
    LM['xhyCap0'] = np.load(stt+'xhyCap0'+end)
    LM['xhdCap0'] = np.load(stt+'xhdCap0'+end)    
    LM['xhyLds0'] = np.load(stt+'xhyLds0'+end)
    LM['xhdLds0'] = np.load(stt+'xhdLds0'+end)
    LM['vYNodeOrder'] = np.load(stt+'vYNodeOrder'+end)
    LM['SyYNodeOrder'] = np.load(stt+'SyYNodeOrder'+end)
    LM['SdYNodeOrder'] = np.load(stt+'SdYNodeOrder'+end)
    
    if netModel==1:
        LM['idxShf'] = np.load(stt+'idxShf'+end)
        LM['regIdxMatVlts'] = np.load(stt+'regIdxMatVlts'+end)
        LM['regVreg'] = np.load(stt+'regVreg'+end)
    
    if netModel==2:
        LM['idxShf'] = np.load(stt+'idxShf'+end)
        LM['regVreg'] = np.load(stt+'regVreg'+end)
    
    return LM
    
def getMu_Kk(feeder,tapOn):
    # fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1',feeder]
    if feeder=='13bus' and tapOn:
        mu_kk = 0.9 # 13 BUS with LTC
    if feeder=='34bus' and tapOn:
        mu_kk = 0.4 # 34 BUS with LTC
    if feeder=='123bus' and tapOn:        
        mu_kk = 3.0 # 123 BUS with LTC
    if feeder=='eulv':
        mu_kk = 0.6 # EU LV
    if feeder=='epriK1' and not tapOn:
        mu_kk = 0.7 # EPRI K1, no LTC
    if feeder=='epriK1' and tapOn:
        mu_kk = 1.20 # EPRI K1, with LTC
    if feeder=='epri7' and not tapOn:
        mu_kk = 0.5 # EPRI ckt7
    if feeder=='usLv':
        mu_kk = 1.75 # US LV
    if feeder=='041':
        mu_kk = 1.5 # 041
    if feeder=='011':
        mu_kk = 0.7 # 011
    if feeder=='193':
        mu_kk = 0.5 # 193 # NOT WORKING for DSS Solve
    if feeder=='213':
        mu_kk = 0.9 # 213 # NOT WORKING for DSS Solve
    if feeder=='162':
        mu_kk = 0.7 # 162
    if feeder=='031':
        mu_kk = 0.6 # 031 # NOT WORKING for DSS Solve
    if feeder=='024':
        mu_kk = 0.45 # 024
    if feeder=='021':
        mu_kk = 0.6 # 021
    if feeder=='074':
        mu_kk = 0.45 # 074
    if feeder=='epri5':
        mu_kk = 0.7
    return mu_kk
    
def getBusCoords(DSSCircuit,DSSText):
    DSSCircuit.Lines.First
    lineName = DSSCircuit.Lines.Name
    
    nM = len(DSSCircuit.Meters.AllNames)
    if nM > 0:
        Mel = []
        i = DSSCircuit.Meters.First
        while i:
            Mel = Mel + [DSSCircuit.Meters.MeteredElement]
            DSSCircuit.Meters.MeteredElement='line.'+lineName
            i = DSSCircuit.Meters.Next
    else:
        DSSText.Command='new energymeter.srcEM element=line.'+lineName
    DSSCircuit.Solution.Solve()
    DSSText.Command='interpolate'
    ABN = DSSCircuit.AllBusNames
    busCoords = {}
    
    for bus in ABN:
        DSSCircuit.SetActiveBus(bus)
        if DSSCircuit.ActiveBus.Coorddefined:
            busCoords[bus] = (DSSCircuit.ActiveBus.x,DSSCircuit.ActiveBus.y)
        else:
            busCoords[bus] = (np.nan,np.nan)
    
    if nM > 0:
        i = DSSCircuit.Meters.First
        while i:
            DSSCircuit.Meters.MeteredElement=Mel[i-1]
            i = DSSCircuit.Meters.Next
    return busCoords

def getBusCoordsAug(busCoords,DSSCircuit,DSSText):
    # approach:
    # go through all branch elements.
    # if both coordinates defined: continue.
    # if only one coordinate defined: coordinate on second element to match first.
    # if neither: step up one element. define coordinate as the same as the coordinate at the bottom of the first element.
    busCoordsAug = busCoords.copy()
    
    ABN = DSSCircuit.AllBusNames
    PDE = DSSCircuit.PDElements
    i = DSSCircuit.PDElements.First
    PDparents = []
    PDelements = []
    while i:
        if not PDE.IsShunt:
            PDelements = PDelements + [PDE.Name]
            buses = DSSCircuit.ActiveElement.BusNames
            
            bus0 = buses[0].split('.')[0]
            bus1 = buses[1].split('.')[0]
            coord0 = busCoordsAug[bus0]
            coord1 = busCoordsAug[bus1]
            
            if np.isnan(coord0[0]) and not np.isnan(coord1[0]):
                busCoordsAug[bus0] = busCoordsAug[bus1]
            if not np.isnan(coord0[0]) and np.isnan(coord1[0]):
                busCoordsAug[bus1] = busCoordsAug[bus0]
            
            parent = PDE.ParentPDElement
            if np.isnan(coord0[0]) and np.isnan(coord1[0]):
                busesPrt = DSSCircuit.ActiveElement.BusNames
                
                bus0prt = busesPrt[0].split('.')[0]
                bus1prt = busesPrt[1].split('.')[0]
                coord0prt = busCoordsAug[bus0prt]
                coord1prt = busCoordsAug[bus1prt]
                
                if not np.isnan(coord0prt[0]) or not np.isnan(coord1prt[0]):
                    if np.isnan(coord0prt[0]):
                        coordPrt = coord1prt
                    else:
                        coordPrt = coord0prt
                    busCoordsAug[bus0] = coordPrt
                    busCoordsAug[bus1] = coordPrt
            if parent:
                PDparents = PDparents + [PDE.Name]
            else:
                PDparents = PDparents + [None]
        i = PDE.Next
    
    return busCoordsAug,PDelements,PDparents
    
    
    
    

def getBranchBuses(DSSCircuit):
    i = DSSCircuit.PDElements.First
    branches = {}
    while i:
        if not DSSCircuit.PDElements.IsShunt:
            DSSCircuit.SetActiveElement(DSSCircuit.PDElements.Name)
            branches[DSSCircuit.PDElements.Name]=DSSCircuit.ActiveElement.BusNames
        i = DSSCircuit.PDElements.Next
    return branches
    
def getBranchNames(DSSCircuit,xfmrSet=False):
    if not xfmrSet:
        i = DSSCircuit.PDElements.First
        branchNames = []
        while i:
            if not DSSCircuit.PDElements.IsShunt:
                branchNames = branchNames + [DSSCircuit.PDElements.Name]
            i = DSSCircuit.PDElements.Next
    elif xfmrSet:
        # nb this does NOT get the current in regulators because that changes with tap positions.
        regXfmrs = get_regXfmr2(DSSCircuit)
        i = DSSCircuit.Transformers.First
        branchNames = []
        while i:
            if DSSCircuit.Transformers.Name not in regXfmrs:
                branchNames = branchNames + ['Transformer.'+DSSCircuit.Transformers.Name]
            i = DSSCircuit.Transformers.Next
    return tuple(branchNames)

def makeYprim(yprimTuple):
    Yprm = tp_2_ar(yprimTuple)
    Yprm = Yprm.reshape((np.sqrt(Yprm.shape)[0].astype('int32'),np.sqrt(Yprm.shape)[0].astype('int32')))
    return Yprm

def countBranchNodes(DSSCircuit):
    i = DSSCircuit.PDElements.First
    nodeNum = 0
    while i:
        if not DSSCircuit.PDElements.IsShunt:
            nodeNum = nodeNum + len(DSSCircuit.ActiveElement.NodeOrder)
        i = DSSCircuit.PDElements.Next
    return nodeNum

def getBranchYprims(DSSCircuit,branchNames):
    nbNodes = countBranchNodes(DSSCircuit)
    YprimMat = sparse.lil_matrix((nbNodes,nbNodes),dtype=complex)
    Yprim_i = 0
    busSet = []
    brchSet = []
    trmlSet = []
    unqIdent = []
    for branch in branchNames:
        DSSCircuit.SetActiveElement(branch)
        NodeOrder = DSSCircuit.ActiveElement.NodeOrder
        nNodes = len(NodeOrder)
        Yprim = makeYprim(DSSCircuit.ActiveElement.Yprim)
        YprimMat[Yprim_i:Yprim_i+nNodes,Yprim_i:Yprim_i+nNodes] = Yprim
        
        Yprim_i = Yprim_i + nNodes
        
        brchSet = brchSet + nNodes*[DSSCircuit.ActiveElement.Name]
        Buses = DSSCircuit.ActiveElement.BusNames
        NodeSet = []
        
        bus_i = 0
        for node in NodeOrder:
            if node in NodeSet:
                bus_i+=1
                NodeSet = []
            if Buses[bus_i].count('.')==0:
                busSet = busSet + [Buses[bus_i]+'.'+str(node)]
            else:
                busSet = busSet + [Buses[bus_i].split('.')[0]+'.'+str(node)]
            NodeSet = NodeSet + [node]
            trmlSet = trmlSet + [bus_i]
            unqIdent = unqIdent + [DSSCircuit.ActiveElement.Name+'..'+busSet[-1]]
            
        if bus_i + 1 < len(Buses):
            print('Warning: not been through all buses in branches.')
    
    if YprimMat[-1,-1]==0:
        print('--- NOTE: YprimMat final element zero; removing additional elements.')
        YprimMat = YprimMat[:Yprim_i,:Yprim_i]
    return YprimMat.tocsr(), tuple(busSet), tuple(brchSet), tuple(trmlSet), tuple(unqIdent)
    # return sparse.csr_matrix(YprimMat), tuple(busSet), tuple(brchSet), tuple(trmlSet), tuple(unqIdent)

def getYzW2V(WbusSet,YZ):
    yzW2V = []
    for wbus in WbusSet:
        try:
            yzW2V.append(YZ.index(wbus.upper()))
        except:
            if wbus[-1]=='0':
                yzW2V.append(-1) # ground node error
            else:
                print('No node',wbus.upper())
    return tuple(yzW2V)

def getV2iBrY(DSSCircuit,YprimMat,busSet):
    # get the modified voltage to branch current matrix for finding branch currents from voltages
    YZ = DSSCircuit.YNodeOrder
    idx = []
    for bus in busSet:
        if bus[-1]=='0':
            idx = idx + [-1] # ground node
        else:
            try:
                idx = idx + [YZ.index(bus)]
            except:
                try:
                    idx = idx + [YZ.index(bus.upper())]
                except:
                    idx = idx + [-1]
                    print('Bus '+bus+' not found, set to ground.')
    # YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    # YNodeVgnd = np.concatenate((YNodeV,np.array([0+0j])))
    # YNodeVprim = YNodeVgnd[idx] # below is equivalent to this, but does not require reindexing
    # Iprim = YprimMat.dot(YNodeVprim)
    Adj = sparse.lil_matrix((len(idx),DSSCircuit.NumNodes+1),dtype=int) # NB if not sparse this step is very slow <----!
    Adj[np.arange(0,len(idx)),idx] = 1
    Adj.tocsr()
    v2iBrY = YprimMat.dot(Adj)
    v2iBrY = v2iBrY[:,:-1] # get rid of ground voltage
    # Iprim = v2iBrY.dot(YNodeV)
    return v2iBrY

def printBrI(Wunq,Iprim):
    i=0 # checking currents
    for unq in Wunq:
        print(unq+':'+str(Iprim[i].real)+', I imag:'+str(Iprim[i].imag))
        i+=1

def pf2kq(pf):
    return np.sqrt(1 - pf**2)/pf

def kq2pf(kq):
    return np.sign(kq)/np.sqrt(1 + kq**2)
    
    
def basicTable(caption,label,heading,data,TD):
    # creates a simple table. caption, label, TD (table directory) are strings;
    # heading is a list of strings, and data is a list of lists of strings, each 
    # sublist the same length as heading.
    if not(TD[-1]=='\\'):
        TD = TD+'\\'
    
    headTxt = ''
    for head in heading:
        headTxt = headTxt + head + ' & '

    headTxt = headTxt[:-3]
    headTxt = headTxt + ' \\\\\n'
    
    nL = len(heading)*'l'

    dataTxt = ''
    for line in data:
        if len(line)!=len(heading):
            print('\nWarning: length of line does not match heading length.\n')
        for point in line:
            dataTxt = dataTxt + point + ' & '
        dataTxt = dataTxt[:-3]
        dataTxt = dataTxt + ' \\\\\n'

    latexText = '% Generated using basicTable.\n\\centering\n\\caption{'+caption+'}\\label{t:'+label+'}\n\\begin{tabular}{'+nL+'}\n\\toprule\n'+headTxt+'\\midrule\n'+dataTxt+'\\bottomrule\n\\end{tabular}\n'

    with open(TD+label+'.tex','wt') as handle:
        handle.write(latexText)
    return latexText

def plotSaveFig(SN,pltSave=True,pltClose=False):
    if pltSave:
        plt.savefig(SN+'.png',bbox_inches='tight',pad_inches=0.01)
        plt.savefig(SN+'.pdf',bbox_inches='tight',pad_inches=0.01)
    if pltClose: plt.close()

def np2lsStr(npArray,nF=2):
    strThing = "%."+str(nF)+"f"
    if npArray.ndim==1:
        listArray = []
        for elem in npArray:
            listArray.append( (strThing % elem) )
    elif npArray.ndim==2:
        listArray = []
        for row in npArray:
            listArray.append([])
            for elem in row:
                listArray[-1].append( (strThing % elem) )
    return listArray

def set_ax_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def sdt(chapter=None,type=None):
    SDT0 = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','thesis')
    if type is None:
        type='f'
    
    if type=='f':
        extension='figures'
    elif type=='t':
        extension='tables'
    
    if chapter=='c1':
        SDT = os.path.join(SDT0,'c1introduction','c1'+extension)
    elif chapter=='c2':
        SDT = os.path.join(SDT0,'c2litreview','c2'+extension)
    elif chapter=='c3' or chapter=='t1':
        SDT = os.path.join(SDT0,'c3tech1','c3'+extension)
    elif chapter=='c4' or chapter=='t2':
        SDT = os.path.join(SDT0,'c4tech2','c4'+extension)
    elif chapter=='c5' or chapter=='t3':
        SDT = os.path.join(SDT0,'c5tech3','c5'+extension)
    elif chapter=='c6':
        SDT = os.path.join(SDT0,'c6conclusions','c6'+extension)
    elif chapter is None:
        SDT = SDT0
    return SDT