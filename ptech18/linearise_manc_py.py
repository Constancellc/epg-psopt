import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time 
from dss_python_funcs import *

print('Start...\n',time.process_time())

# ======== specify working directories
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
# WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

try:
    DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
    print("Unable to stat the OpenDSS Engine")
    raise SystemExit

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

def create_tapped_ybus( DSSObj,fn_y,fn_ckt,feeder,TR_name,TC_No0 ):
    DSSText = DSSObj.Text;
    DSSText.command='Compile ('+fn_y+')'
    DSSCircuit=DSSObj.ActiveCircuit
    i = DSSCircuit.RegControls.First
    while i!=0:
        DSSCircuit.RegControls.TapNumber=TC_No0[i]
        i = DSSCircuit.RegControls.Next
    DSSCircuit.Solution.Solve()
    
    # Ybus_,YNodeOrder_,n = create_ybus(DSSCircuit)
    Ybus_,YNodeOrder_,n = build_y(DSSObj,fn_ckt)
    Ybus = Ybus_[3:,3:]
    YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
    return Ybus, YNodeOrder

def nrel_linearization_My(Ybus,Vh,V0):
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    a = spla.spsolve(Yll,Yl0.dot(-V0))
    
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    My_i = Vh_diag.dot(Yll)
    My_0 = spla.inv(My_i.tocsc())
    My = sparse.hstack((My_0,-1j*My_0))
    return My,a

def nrel_linearization_Ky(My,Vh,sY):
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real ).toarray()
    b = abs(Vh) - Ky.dot(-1e3*s_2_x(sY[3:]))
    return Ky, b
    
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

fn_ckt = WD+'\\LVTestCase_copy'
fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\master_z'
feeder='eulv'

fn_y = fn+'_y'

sn = WD + '\\lin_models\\' + feeder

# lin_points=np.array([0.3,0.6,1.0])
lin_points=np.array([1.0])
k = np.arange(-0.7,1.8,0.1)
# k = np.arange(-0.1,0.5,0.1)

ve=np.zeros([k.size,lin_points.size])
vae=np.zeros([k.size,lin_points.size])

for K in range(len(lin_points)):
    lin_point = lin_points[K]
    # lin_point=0.3
    # run the dss
    DSSText.command='Compile ('+fn+'.dss)'
    TC_No0,TC_bus = find_tap_pos(DSSCircuit)
    TR_name = []
    test = tp_2_ar(DSSCircuit.YNodeVarray)
    print('Load Ybus\n',time.process_time())
    Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,feeder,TR_name,TC_No0 )
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
    # S,V,I,B,D = ld_vals(DSSCircuit)
    # n2y = node_to_YZ(DSSCircuit)
    # iY,sY,iD,sD = calc_sYsD(YNodeOrder,B,I,S,D,n2y)
    sY,sD,iY,iD = get_sYsD(DSSCircuit)
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    # --------------------
    xhy0 = -1e3*np.array([[sY.real],[sY.imag]])
    
    V0 = YNodeV[0:3]
    Vh = YNodeV[3:]

    print('Create linear model My:\n',time.process_time())
    My,a = nrel_linearization_My( Ybus,Vh,V0 )
    
    print('Create linear model Ky:\n',time.process_time())
    Ky,b = nrel_linearization_Ky(My,Vh,sY)
    # now, check these are working
    v_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
    va_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
    v_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
    va_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)

    print('Start validation\n',time.process_time())
    for i in range(len(k)):
        DSSText.command='Compile ('+fn+')'
        DSSText.command='Set controlmode=off'
        DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
        # cpf_set_loads(DSSCircuit,BB00,SS00,k[i]/lin_point)
        cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point)
        DSSSolution.Solve()
        v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
        va_0[i,:] = abs(v_0[i,:])
        # S,V,I,B,D = ld_vals(DSSCircuit)
        # iY,sY,iD,sD = calc_sYsD(YNodeOrder,B,I,S,D,n2y)
        sY,sD,iY,iD = get_sYsD(DSSCircuit)
        # xhy = -1e3*np.concatenate((sY[3:].real,sY[3:].imag))
        xhy = -1e3*s_2_x(sY[3:])
        v_l[i,:] = My.dot(xhy) + a
        ve[i,K] = np.linalg.norm( v_l[i,:] - v_0[i,3:] )/np.linalg.norm(v_0[i,3:])
        # va_l[i,:] = Ky.dot(xhy) + b
        # vae[i,K] = np.linalg.norm( va_l[i,:] - va_0[i,3:] )/np.linalg.norm(va_0[i,3:])
print('Complete.\n',time.process_time())

plt.plot(k,ve), plt.show()
# plt.plot(k,vae), plt.show()