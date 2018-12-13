import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time 

print('Start...\n',time.process_time())

WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
# WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

try:
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
    print("Unable to stat the OpenDSS Engine")
    raise SystemExit

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
		

def assemble_ybus(SystemY):
    AA = SystemY[np.arange(0,SystemY.size,2)]
    BB = SystemY[np.arange(1,SystemY.size,2)]
    n = int(sqrt(AA.size))
    Ybus = sparse.dok_matrix(np.reshape(AA+1j*BB,(n,n))) # perhaps other sparse method might work better?
    return Ybus
	
def create_ybus(DSSCircuit):
    SystemY = np.array(DSSCircuit.SystemY)
    Ybus = assemble_ybus(SystemY)
    YNodeOrder = DSSCircuit.YNodeOrder; 
    n = Ybus.shape[0];
    return Ybus, YNodeOrder, n
	
def find_tap_pos(DSSCircuit):
    TC_No=[]
    TC_bus=[]
    i = DSSCircuit.RegControls.First
    while i!=0:
        TC_No.append(DSSCircuit.RegControls.TapNumber)
    return TC_No,TC_bus

def create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 ):
    DSSText = DSSObj.Text;
    DSSText.command='Compile ('+fn_y+')'
    DSSCircuit=DSSObj.ActiveCircuit
    i = DSSCircuit.RegControls.First
    while i!=0:
        DSSCircuit.RegControls.TapNumber=TC_No0[i]
        i = DSSCircuit.RegControls.Next
    DSSCircuit.Solution.Solve
    Ybus_,YNodeOrder_,n = create_ybus(DSSCircuit)
    Ybus = Ybus_[3:,3:]
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

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

def nrel_linearization_My(Ybus,Vh,V0):
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    a = spla.spsolve(Yll,Yl0.dot(-V0))
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    My_i = Vh_diag.dot(Yll)
    My_0 = spla.inv(My_i.tocsc())
    My = sparse.hstack((My_0,-1j*My_0))
    return My,a
    
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\master_z'
feeder='eulv'

fn_y = fn+'_y'
sn = WD + '\\lin_models\\' + feeder

lin_points=np.array([0.3,0.6,1.0])
k = np.arange(-0.7,1.8,0.1)
# k = np.arange(-0.1,0.5,0.1)

ve=np.zeros([k.size,lin_points.size])
ve0=np.zeros([k.size,lin_points.size])

for K in range(len(lin_points)):
    lin_point = lin_points[K]
    # lin_point=0.3
    # run the dss
    DSSText.command='Compile ('+fn+'.dss)'
    TC_No0,TC_bus = find_tap_pos(DSSCircuit)
    TR_name = []
    print('Load Ybus\n',time.process_time())
    Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 )
    # YNodeOrder = DSSCircuit.YNodeOrder # put in if not creating ybus as above

    # Reproduce delta-y power flow eqns (1)
    DSSText.command='Compile ('+fn+'.dss)'
    DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    DSSSolution.Solve()
    BB00,SS00 = cpf_get_loads(DSSCircuit)

    k00 = lin_point/SS00[1].real
    
    cpf_set_loads(DSSCircuit,BB00,SS00,k00)
    DSSSolution.Solve()
    
    # get the Y, D currents/power
    S,V,I,B,D = ld_vals(DSSCircuit)
    n2y = node_to_YZ(DSSCircuit)
    iY,sY,iD,sD = calc_sYsD(YNodeOrder,B,I,S,D,n2y)
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    # --------------------
    xhy0 = -1e3*np.array([[sY.real],[sY.imag]])
    
    V0 = YNodeV[0:3]
    Vh = YNodeV[3:]

    print('Create linear model:\n',time.process_time())
    My,a = nrel_linearization_My( Ybus,Vh,V0 )
    
    # now, check these are working
    v_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
    v_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)

    print('Start validation\n',time.process_time())
    for i in range(len(k)):
        DSSText.command='Compile ('+fn+')'
        DSSText.command='Set controlmode=off'
        DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
        # cpf_set_loads(DSSCircuit,BB00,SS00,k[i]/lin_point)
        cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point)
        DSSSolution.Solve()
        v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
        S,V,I,B,D = ld_vals(DSSCircuit)
        iY,sY,iD,sD = calc_sYsD(YNodeOrder,B,I,S,D,n2y)
        xhy = -1e3*np.concatenate((sY[3:].real,sY[3:].imag))
        v_l[i,:] = My.dot(xhy) + a
        ve[i,K] = np.linalg.norm( v_l[i,:] - v_0[i,3:] )/np.linalg.norm(v_0[i,3:])

print('Complete.\n',time.process_time())
plt.plot(k,ve)
plt.show()