import numpy as np

def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

def s_2_x(s):
    return np.concatenate((s.real,s.imag))
    
def ld_vals( DSSCircuit ):
    ii = DSSCircuit.FirstPCElement()
    S=[]; V=[]; I=[]; B=[]; D=[]
    while ii!=0:
        if DSSCircuit.ActiveElement.name[0:4].lower()=='load':
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.BusNames)
            D.append(DSSCircuit.Loads.IsDelta)
        ii=DSSCircuit.NextPCElement()
    jj = DSSCircuit.FirstPDElement()
    while jj!=0:
        if DSSCircuit.ActiveElement.name[0:4].lower()=='capa':
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            B.append(DSSCircuit.ActiveElement.bus)
            D.append(DSSCircuit.Capacitors.IsDelta)
        jj=DSSCircuit.NextPDElement()
    return S,V,I,B,D

def find_node_idx(n2y,bus,D):
    idx = []
    if D:
        try:
            idx.append(n2y[bus[0:-3]])
        except:
            idx.append(n2y[bus+'.1'])
            idx.append(n2y[bus+'.2'])
            idx.append(n2y[bus+'.3'])
    else:
        try:
            idx.append(n2y[bus])
        except:
            idx.append(n2y[bus+'.1'])
            idx.append(n2y[bus+'.2'])
            idx.append(n2y[bus+'.3'])
    return idx
    
def calc_sYsD( YZ,B,I,S,D,n2y ): # YZ as YNodeOrder
    iD = np.zeros(len(YZ),dtype=complex);sD = np.zeros(len(YZ),dtype=complex);
    iY = np.zeros(len(YZ),dtype=complex);sY = np.zeros(len(YZ),dtype=complex)
    for i in range(len(B)):
        for bus in B[i]:
            idx = find_node_idx(n2y,bus,D[i])
            if D[i]:
                if bus.count('.')==2:
                    ph = int(bus[-3])
                    iD[idx[ph-1]] = iD[idx[ph-1]] + I[i]
                    sD[idx[ph-1]] = sD[idx[ph-1]] + S[i][0] + S[i][1]
                else:
                    iD[idx] = iD[idx] + I[i]*np.exp(1j*np.pi/6)/np.sqrt(3)
                    sD[idx] = sD[idx] + S[i]
            else:
                if bus.count('.')>0:
                    # ph=int(bus[-1])
                    iY[idx] = iY[idx] + I[i][0]
                    sY[idx] = sY[idx] + S[i][0]
                else:
                    iY[idx] = iY[idx] + I[i]
                    sY[idx] = sY[idx] + S[i]
    return iY, sY, iD, sD

def node_to_YZ(DSSCircuit):
    n2y = {}
    YNodeOrder = DSSCircuit.YNodeOrder
    for node in DSSCircuit.AllNodeNames:
        n2y[node]=YNodeOrder.index(node.upper())
    return n2y

def get_sYsD(DSSCircuit):
    S,V,I,B,D = ld_vals( DSSCircuit )
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
