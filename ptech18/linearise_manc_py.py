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

def nrel_linearization(Ybus,Vh,V0,H):
    Yll = Ybus[3:,3:].tocsc()
    Yl0 = Ybus[3:,0:3].tocsc()
    H0 = sparse.csc_matrix(H[:,3:])
    a = spla.spsolve(Yll,Yl0.dot(-V0))
    
    Ylli = spla.inv(Yll)
    
    Vh_diagi = sparse.dia_matrix( (1/(Vh.conj()),0),shape=(len(Vh),len(Vh)) )
    HVh_diagi = sparse.dia_matrix( (1/(H0.dot(Vh.conj())),0),shape=(H0.shape[0],H0.shape[0]) )
    
    My_0 = Ylli.dot(Vh_diagi)
    Md_0 = Ylli.dot(H0.T.dot(HVh_diagi))
    # Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    # My_i = Vh_diag.dot(Yll)
    # My_0 = spla.inv(My_i.tocsc())
    My = sparse.hstack((My_0,-1j*My_0))
    Md = sparse.hstack((Md_0,-1j*Md_0))
    return My,Md,a

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

def nrel_linearization_K(My,Md,Vh,sY,sD):
    Vh_diag = sparse.dia_matrix( (Vh.conj(),0),shape=(len(Vh),len(Vh)) )
    Vhai_diag = sparse.dia_matrix( (np.ones(len(Vh))/abs(Vh),0),shape=(len(Vh),len(Vh)) )
    Ky = Vhai_diag.dot( Vh_diag.dot(My).real ).toarray()
    Kd = Vhai_diag.dot( Vh_diag.dot(Md).real ).toarray()
    b = abs(Vh) - Ky.dot(-1e3*s_2_x(sY[3:]))- Kd.dot(-1e3*s_2_x(sD))
    return Ky, Kd, b

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node']
ckts = {'feeder_name':['fn_ckt','fn']}
ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
ckts[fdrs[1]]=feeder_to_fn(WD,fdrs[1])
ckts[fdrs[2]]=feeder_to_fn(WD,fdrs[2])
ckts[fdrs[3]]=feeder_to_fn(WD,fdrs[3])
ckts[fdrs[4]]=feeder_to_fn(WD,fdrs[4])
ckts[fdrs[5]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_z']
ckts[fdrs[6]]=[WD+'\\ieee_tn\\34Bus_copy',WD+'\\ieee_tn\\34Bus_copy\\ieee34Mod1_z_mod']
ckts[fdrs[7]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37']
ckts[fdrs[8]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123Master_z']
ckts[fdrs[9]]=[WD+'\\ieee_tn\\8500-Node_copy',WD+'\\ieee_tn\\8500-Node_copy\\Master-unbal_z']

fn_ckt = ckts[fdrs[fdr_i]][0]
fn = ckts[fdrs[fdr_i]][1]
feeder=fdrs[fdr_i]

fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

lin_points=np.array([0.3,0.6,1.0])
lin_points=np.array([1.0])
k = np.arange(-0.6,1.8,0.2)
test_model = True

ve=np.zeros([k.size,lin_points.size])
vve=np.zeros([k.size,lin_points.size])
vae=np.zeros([k.size,lin_points.size])
vvae=np.zeros([k.size,lin_points.size])

for K in range(len(lin_points)):
    lin_point = lin_points[K]
    # run the dss
    DSSText.command='Compile ('+fn+'.dss)'
    TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
    print('Load Ybus\n',time.process_time())
    
    # Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 )
    # Ybus0, YNodeOrder0 = create_tapped_ybus_slow( DSSObj,fn_y,TC_No0 )
    Ybus, YNodeOrder = create_tapped_ybus_very_slow( DSSObj,fn_y,TC_No0 )
    
    # print('Calculate condition no.:\n',time.process_time())
    # cndY = np.linalg.cond(Ybus.toarray())
    # print(np.log10(cndY))
    # cndY0 = np.linalg.cond(Ybus0)
    # print(np.log10(cndY0))
    # print("%.16f"%Ybus[0,0].imag)
    # print('Complete.\n',time.process_time())
    
    # Reproduce delta-y power flow eqns (1)
    DSSText.command='Compile ('+fn+'.dss)'
    fix_tap_pos(DSSCircuit, TC_No0)
    DSSText.command='Set Controlmode=off'
    # DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    DSSSolution.Solve()
    BB00,SS00 = cpf_get_loads(DSSCircuit)

    # k00 = lin_point/SS00[1].real
    k00 = lin_point
    
    cpf_set_loads(DSSCircuit,BB00,SS00,k00)
    DSSSolution.Solve()
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    
    chkc = abs(iTot + Ybus.dot(YNodeV))/abs(iTot) # 1c needs checking outside
    chkc_n = np.linalg.norm(iTot + Ybus.dot(YNodeV))/np.linalg.norm(iTot) # 1c needs checking outside
    print_node_array(DSSCircuit.YNodeOrder,chkc)
    # plt.plot( chkc_nom[np.isinf(chkc)==False] ), plt.show()
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    # --------------------
    xhy0 = -1e3*s_2_x(sY[3:])
    xhd0 = -1e3*s_2_x(sD) # not [3:] like sY!
    
    V0 = YNodeV[0:3]
    Vh = YNodeV[3:]

    if len(H)==0:
        print('Create linear models My:\n',time.process_time())
        My,a = nrel_linearization_My( Ybus,Vh,V0 )
        print('Create linear models Ky:\n',time.process_time())
        Ky,b = nrel_linearization_Ky(My,Vh,sY)
    else:
        print('Create linear models M:\n',time.process_time())
        My,Md,a = nrel_linearization( Ybus,Vh,V0,H )
        print('Create linear models K:\n',time.process_time())
        Ky,Kd,b = nrel_linearization_K(My,Md,Vh,sY,sD)

    DSSText.command='Compile ('+fn+')'
    fix_tap_pos(DSSCircuit, TC_No0)
    DSSText.command='Set controlmode=off'
    # DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'

    # NB!!! -3 required for models which have the first three elements chopped off!
    v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
    v_idx = np.array(get_element_idxs(DSSCircuit,v_types)) - 3
    v_idx = v_idx[v_idx>=0]
    
    p_idx = np.array(sY[3:].nonzero())
    s_idx = np.concatenate((p_idx,p_idx+len(sY)-3),axis=1)[0]

    MyV = My[v_idx,:][:,s_idx]
    aV = a[v_idx]
    KyV = Ky[v_idx,:][:,s_idx]
    bV = b[v_idx]
    if len(H)!=0: # already gotten rid of s_idx
        MdV = Md[v_idx,:]
        KdV = Kd[v_idx,:]

    # now, check these are working
    v_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
    vv_0 = np.zeros((len(k),len(v_idx)),dtype=complex)
    va_0 = np.zeros((len(k),len(YNodeOrder)))
    vva_0 = np.zeros((len(k),len(v_idx)))
    v_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
    vv_l = np.zeros((len(k),len(v_idx)),dtype=complex)
    va_l = np.zeros((len(k),len(YNodeOrder)-3))
    vva_l = np.zeros((len(k),len(v_idx)))
    
    Convrg = []
    if test_model:
        print('Start validation\n',time.process_time())
        for i in range(len(k)):
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
            vv_0[i,:] = v_0[i,3:][v_idx]
            va_0[i,:] = abs(v_0[i,:])
            vva_0[i,:] = va_0[i,3:][v_idx]
            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])
            if len(H)==0:
                v_l[i,:] = My.dot(xhy) + a
                vv_l[i,:] = MyV.dot(xhy[s_idx]) + aV
                va_l[i,:] = Ky.dot(xhy) + b
                vva_l[i,:] = KyV.dot(xhy[s_idx]) + bV
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                v_l[i,:] = My.dot(xhy) + Md.dot(xhd) + a
                vv_l[i,:] = MyV.dot(xhy[s_idx]) + MdV.dot(xhd) + aV
                va_l[i,:] = Ky.dot(xhy) + Kd.dot(xhd) + b
                vva_l[i,:] = KyV.dot(xhy[s_idx]) + KdV.dot(xhd) + bV
            ve[i,K] = np.linalg.norm( v_l[i,:] - v_0[i,3:] )/np.linalg.norm(v_0[i,3:])
            vve[i,K] = np.linalg.norm( vv_l[i,:] - vv_0[i,:] )/np.linalg.norm(vv_0[i,:])
            vae[i,K] = np.linalg.norm( va_l[i,:] - va_0[i,3:] )/np.linalg.norm(va_0[i,3:])
            vvae[i,K] = np.linalg.norm( vva_l[i,:] - vva_0[i,:] )/np.linalg.norm(vva_0[i,:])
    header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
    lp_str = str(round(lin_point*100).astype(int)).zfill(3)
    # np.savetxt(sn0+'Ky'+lp_str+'.txt',KyV,header=header_str)
    # np.savetxt(sn0+'xhy0'+lp_str+'.txt',xhy0[s_idx],header=header_str)
    # np.savetxt(sn0+'bV'+lp_str+'.txt',bV,header=header_str)
    # if len(H)!=0:
        # np.savetxt(sn0+'Kd'+lp_str+'.txt',KdV,header=header_str)
        # np.savetxt(sn0+'xhd0'+lp_str+'.txt',xhd0,header=header_str)
print('Complete.\n',time.process_time())
print(Convrg)
if test_model:
    plt.plot(k,ve), plt.show()
    plt.plot(k,vve,'x-'), plt.show()
    # plt.plot(k,vae), plt.show()
    # plt.plot(k,vvae), plt.show()