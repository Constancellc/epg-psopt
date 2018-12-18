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
# WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

def create_tapped_ybus( DSSObj,fn_y,fn_ckt,feeder,TR_name,TC_No0 ):
    DSSText = DSSObj.Text
    DSSText.command='Compile ('+fn_y+')'
    DSSCircuit=DSSObj.ActiveCircuit
    i = DSSCircuit.RegControls.First
    while i!=0:
        DSSCircuit.RegControls.TapNumber=TC_No0[i-1]
        i = DSSCircuit.RegControls.Next
    DSSCircuit.Solution.Solve()
    
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

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution



# --------------- circuit info
fdr_i = 5 # do NOT set equal to 2!
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus']
ckts = {'feeder_name':['fn_ckt','fn']}
ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
ckts[fdrs[1]]=feeder_to_fn(WD,fdrs[1])
ckts[fdrs[2]]=feeder_to_fn(WD,fdrs[2])
ckts[fdrs[3]]=feeder_to_fn(WD,fdrs[3])
ckts[fdrs[4]]=feeder_to_fn(WD,fdrs[4])
ckts[fdrs[5]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt']
ckts[fdrs[6]]=[WD+'\\ieee_tn\\34Bus_copy',WD+'\\ieee_tn\\34Bus_copy\\ieee34Nodeckt_z']
ckts[fdrs[7]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37']

fn_ckt = ckts[fdrs[fdr_i]][0]
fn = ckts[fdrs[fdr_i]][1]
feeder=fdrs[fdr_i]

fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

# lin_points=np.array([0.3,0.6,1.0])
lin_points=np.array([1.0])
k = np.arange(-0.6,1.8,0.2)
test_model = True

ve=np.zeros([k.size,lin_points.size])
vae=np.zeros([k.size,lin_points.size])
vvae=np.zeros([k.size,lin_points.size])

for K in range(len(lin_points)):
    lin_point = lin_points[K]
    # lin_point=0.3
    # run the dss
    DSSText.command='Compile ('+fn+'.dss)'
    TC_No0,TC_bus = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
    TR_name = []
    test = tp_2_ar(DSSCircuit.YNodeVarray)
    print('Load Ybus\n',time.process_time())
    Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,feeder,TR_name,TC_No0 )

    # Reproduce delta-y power flow eqns (1)
    DSSText.command='Compile ('+fn+'.dss)'
    DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    DSSSolution.Solve()
    BB00,SS00 = cpf_get_loads(DSSCircuit)

    k00 = lin_point/SS00[1].real
    
    cpf_set_loads(DSSCircuit,BB00,SS00,k00)
    DSSSolution.Solve()
    sY,sD,iY,iD = get_sYsD(DSSCircuit)
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    # --------------------
    xhy0 = -1e3*np.hstack((sY.real[3:],sY.imag[3:]))
    
    V0 = YNodeV[0:3]
    Vh = YNodeV[3:]

    print('Create linear model My:\n',time.process_time())
    My,a = nrel_linearization_My( Ybus,Vh,V0 )
    
    print('Create linear model Ky:\n',time.process_time())
    Ky,b = nrel_linearization_Ky(My,Vh,sY)
    
    DSSText.command='Compile ('+fn+')'
    DSSText.command='Set controlmode=off'
    DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'

    # NB!!! -3 required for Ky which has the first three elements chopped off!
    v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
    v_idx = np.array(get_element_idxs(DSSCircuit,v_types)) - 3
    p_idx = np.array(get_element_idxs(DSSCircuit,[DSSCircuit.Loads])) - 3
    s_idx = np.concatenate((p_idx,p_idx+len(YNodeOrder)-3))

    v_idx = v_idx[v_idx>=0]
    s_idx = s_idx[s_idx>=0]

    KyV = Ky[v_idx,:][:,s_idx]
    bV = b[v_idx]

    # now, check these are working
    v_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
    va_0 = np.zeros((len(k),len(YNodeOrder)))
    vva_0 = np.zeros((len(k),len(v_idx)))
    v_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
    va_l = np.zeros((len(k),len(YNodeOrder)-3))
    vva_l = np.zeros((len(k),len(v_idx)))

    if test_model:
        print('Start validation\n',time.process_time())
        for i in range(len(k)):
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point)
            DSSSolution.Solve()
            v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
            va_0[i,:] = abs(v_0[i,:])
            vva_0[i,:] = va_0[i,3:][v_idx]
            sY,sD,iY,iD = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])
            v_l[i,:] = My.dot(xhy) + a
            ve[i,K] = np.linalg.norm( v_l[i,:] - v_0[i,3:] )/np.linalg.norm(v_0[i,3:])
            va_l[i,:] = Ky.dot(xhy) + b
            vae[i,K] = np.linalg.norm( va_l[i,:] - va_0[i,3:] )/np.linalg.norm(va_0[i,3:])
            vva_l[i,:] = KyV.dot(xhy[s_idx]) + bV
            vvae[i,K] = np.linalg.norm( vva_l[i,:] - vva_0[i,:] )/np.linalg.norm(vva_0[i,:])
    header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
    lp_str = str(round(lin_point*100).astype(int)).zfill(3)
    np.savetxt(sn0+'Ky'+lp_str+'.txt',KyV,header=header_str)
    np.savetxt(sn0+'bV'+lp_str+'.txt',bV,header=header_str)
    np.savetxt(sn0+'xhy0'+lp_str+'.txt',xhy0[s_idx],header=header_str)
print('Complete.\n',time.process_time())

if test_model:
    plt.plot(k,ve), plt.show()
    # plt.plot(k,vae), plt.show()
    # plt.plot(k,vvae), plt.show()