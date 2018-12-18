import numpy as np
import os
from scipy import sparse

def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

def s_2_x(s):
    return np.concatenate((s.real,s.imag))
    
def ld_vals( DSSCircuit ):
    ii = DSSCircuit.FirstPCElement()
    S=[]; V=[]; I=[]; B=[]; D=[]; N=[]
    while ii!=0:
        if DSSCircuit.ActiveElement.name[0:4].lower()=='load':
            DSSCircuit.Loads.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            D.append(DSSCircuit.Loads.IsDelta)
            N.append(DSSCircuit.Loads.Name)
        ii=DSSCircuit.NextPCElement()
    jj = DSSCircuit.FirstPDElement()
    while jj!=0:
        if DSSCircuit.ActiveElement.name[0:4].lower()=='capa':
            DSSCircuit.Capacitors.Name=DSSCircuit.ActiveElement.Name.split(sep='.')[1]
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            D.append(DSSCircuit.Capacitors.IsDelta)
            N.append(DSSCircuit.Capacitors.Name)
        jj=DSSCircuit.NextPDElement()
    return S,V,I,B,D,N

def find_node_idx(n2y,bus,D):
    idx = []
    BS = bus.split('.',1)
    bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
    if ph=='1.2.3' or bus.count('.')==0:
        idx.append(n2y[bus_id+'.1'])
        idx.append(n2y[bus_id+'.2'])
        idx.append(n2y[bus_id+'.3'])
    elif ph=='0.0.0':
        try:
            idx.append(n2y[bus_id+'.1'])
        except:
            pass
        try:
            idx.append(n2y[bus_id+'.2'])
        except:
            pass
        try:
            idx.append(n2y[bus_id+'.3'])
                except:
            pass
    elif D:
        idx.append(n2y[bus[0:-2]])
    else:
        idx.append(n2y[bus])
    return idx
    
def calc_sYsD( YZ,B,I,S,D,n2y ): # YZ as YNodeOrder
    iD = np.zeros(len(YZ),dtype=complex); sD = np.zeros(len(YZ),dtype=complex)
    iY = np.zeros(len(YZ),dtype=complex); sY = np.zeros(len(YZ),dtype=complex)
    for i in range(len(B)):
        for bus in B[i]:
            idx = find_node_idx(n2y,bus,D[i])
            BS = bus.split('.',1)
            bus_id,ph = [BS[0],BS[-1]] # catch cases where there is no phase
            if D[i]:
                if bus.count('.')==2:
                    # iD[idx[ph-1]] = iD[idx[ph-1]] + I[i]
                    # sD[idx[ph-1]] = sD[idx[ph-1]] + S[i][0] + S[i][1]
                    iD[idx] = iD[idx] + I[i][0]
                    sD[idx] = sD[idx] + S[i].sum()
                else:
                    iD[idx] = iD[idx] + I[i]*np.exp(1j*np.pi/6)/np.sqrt(3)
                    sD[idx] = sD[idx] + S[i]
            else:
                if ph[0]!='0':
                    if bus.count('.')>0:
                        # ph=int(bus[-1])
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
    YZ = DSSCircuit.YNodeOrder
    iY, sY, iD, sD = calc_sYsD( YZ,B,I,S,D,n2y )
    return sY,sD,iY,iD
    

def cpf_get_loads(DSSCircuit):
    SS = {}
    BB = {}
    i = DSSCircuit.Loads.First
    while i!=0:
        SS[i]=DSSCircuit.Loads.kW + 1j*DSSCircuit.Loads.kvar
        BB[i]=DSSCircuit.Loads.Name
        i=DSSCircuit.Loads.next
    imax = DSSCircuit.Loads.Count
    j = DSSCircuit.Capacitors.First
    while j!=0:
        SS[imax+j]=1j*DSSCircuit.Capacitors.kvar
        BB[imax+j]=DSSCircuit.Capacitors.Name
        j = DSSCircuit.Capacitors.Next
    return BB,SS

def cpf_set_loads(DSSCircuit,BB,SS,k):
    i = DSSCircuit.Loads.First
    while i!=0:
        DSSCircuit.Loads.Name=BB[i]
        DSSCircuit.Loads.kW = k*SS[i].real
        i=DSSCircuit.Loads.Next
    imax = DSSCircuit.Loads.Count
    j = DSSCircuit.Capacitors.First
    while j!=0:
        DSSCircuit.Capacitors.Name=BB[j+imax]
        DSSCircuit.Capacitors.kVar=k*SS[j+imax].imag
        j=DSSCircuit.Capacitors.next
    return

def find_tap_pos(DSSCircuit):
    TC_No=[]
    TC_bus=[]
    i = DSSCircuit.RegControls.First
    while i!=0:
        TC_No.append(DSSCircuit.RegControls.TapNumber)
        i = DSSCircuit.RegControls.Next
    return TC_No,TC_bus

def build_y(DSSObj,fn_ckt):
    # DSSObj.Text.command='Compile ('+fn_z+'.dss)'
    YNodeOrder = DSSObj.ActiveCircuit.YNodeOrder
    DSSObj.Text.command='show Y'
    os.system("TASKKILL /F /IM notepad.exe")
    
    fn_y = fn_ckt+'\\'+DSSObj.ActiveCircuit.name+'_SystemY.txt'
    fn_csv = fn_ckt+'\\'+DSSObj.ActiveCircuit.name+'_SystemY_csv.txt'

    file_r = open(fn_y,'r')
    stream = file_r.read()

    stream=stream.replace('[','')
    # stream=stream.replace(']',',')
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

def get_idxs(e_idx,DSSCircuit,ELE):
    i = ELE.First
    while i:
        for bn in DSSCircuit.ActiveElement.BusNames:
            bn = bn.upper()
            try:
                e_idx.append(DSSCircuit.YNodeOrder.index(bn))
            except:
                for ph in range(1,4):
                    e_idx.append(DSSCircuit.YNodeOrder.index(bn+'.'+str(ph)))
        i = ELE.next
    return e_idx

def get_element_idxs(DSSCircuit,ele_types):
    e_idx = []
    for ELE in ele_types:
            e_idx = get_idxs(e_idx,DSSCircuit,ELE)
    return e_idx
def feeder_to_fn(WD,feeder):
    paths = []
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1])
    paths.append(WD+'\\manchester_models\\network_'+feeder[3]+'\\Feeder_'+feeder[1]+'\master')
    return paths