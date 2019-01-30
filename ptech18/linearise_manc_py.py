import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
from dss_python_funcs import *
from dss_vlin_funcs import *
import getpass

# ======== specify working directories
if getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
elif getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"


DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
DSSSolution.tolerance=1e-7

# ------------------------------------------------------------ circuit info
test_model = True
fdr_i = 13
fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod','13busRegModRx','13busModSng','usLv','123busMod']; lp_taps='Nmt'
feeder='041'
feeder=fdrs[fdr_i]
lp_taps='Lpt'

lin_points=np.array([0.3,0.6,1.0])
# lin_points=np.array([0.6])
k = np.arange(-1.5,1.6,0.1)
# k = np.array([-0.5,0,0.5,1.0,1.5])

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

fn_y = fn+'_y'
dir0 = WD + '\\lin_models\\' + feeder
sn0 = dir0 + '\\' + feeder + lp_taps

print('Start, feeder:',feeder,'\n',time.process_time())

ve=np.zeros([k.size,lin_points.size])
vve=np.zeros([k.size,lin_points.size])
vae=np.zeros([k.size,lin_points.size])
vvae=np.zeros([k.size,lin_points.size])
DVslv_e=np.zeros([k.size,lin_points.size])

for K in range(len(lin_points)):
    lin_point = lin_points[K]
    # run the dss
    DSSText.command='Compile ('+fn+'.dss)'
    DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    BB00,SS00 = cpf_get_loads(DSSCircuit)
    if lp_taps=='Nmt':
        TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
    elif lp_taps=='Lpt':
        cpf_set_loads(DSSCircuit,BB00,SS00,lin_point)
        DSSSolution.Solve()
        TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
    print('Load Ybus\n',time.process_time())
    
    # Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 ) # for LV networks
    Ybus, YNodeOrder = create_tapped_ybus_very_slow( DSSObj,fn_y,TC_No0 )
    
    # print('Calculate condition no.:\n',time.process_time()) # for debugging
    # cndY = np.linalg.cond(Ybus.toarray())
    # print(np.log10(cndY))
    
    # Reproduce delta-y power flow eqns (1)
    DSSText.command='Compile ('+fn+'.dss)'
    fix_tap_pos(DSSCircuit, TC_No0)
    DSSText.command='Set Controlmode=off'
    DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    DSSSolution.Solve()
    # BB00,SS00 = cpf_get_loads(DSSCircuit)
    
    Yvbase = get_Yvbase(DSSCircuit)[3:]
    
    cpf_set_loads(DSSCircuit,BB00,SS00,lin_point)
    DSSSolution.Solve()
    YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    # sY0,sD0,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    # chkc = abs(iTot + Ybus.dot(YNodeV))/abs(iTot) # 1c needs checking outside
    # chkc_n = np.linalg.norm(iTot + Ybus.dot(YNodeV))/np.linalg.norm(iTot) # 1c needs checking outside
    # print_node_array(DSSCircuit.YNodeOrder,chkc)
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
    v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3
    v_idx = v_idx[v_idx>=0]
    YvbaseV = Yvbase[v_idx]
    
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
    
    Vslv = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
    
    v_l = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
    vv_l = np.zeros((len(k),len(v_idx)),dtype=complex)
    va_l = np.zeros((len(k),len(YNodeOrder)-3))
    vva_l = np.zeros((len(k),len(v_idx)))
    
    
    Convrg = []
    TP = np.zeros((len(lin_points),len(k)),dtype=complex)
    TL = np.zeros((len(lin_points),len(k)),dtype=complex)
    if test_model:
        print('Start validation\n',time.process_time())
        for i in range(len(k)):
            print(i)
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            TP[K,i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
            TL[K,i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
            
            v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
            vv_0[i,:] = v_0[i,3:][v_idx]
            va_0[i,:] = abs(v_0[i,:])
            vva_0[i,:] = va_0[i,3:][v_idx]
            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])
            
            # Vslv[i,:] = fixed_point_solve(Ybus,v_0[i,:],-1e3*sY[3:],-1e3*sD,H)
            
            if len(H)==0:
                # v_l[i,:] = My.dot(xhy) + a
                vv_l[i,:] = MyV.dot(xhy[s_idx]) + aV
                # va_l[i,:] = Ky.dot(xhy) + b
                vva_l[i,:] = KyV.dot(xhy[s_idx]) + bV
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                v_l[i,:] = My.dot(xhy) + Md.dot(xhd) + a
                vv_l[i,:] = MyV.dot(xhy[s_idx]) + MdV.dot(xhd) + aV
                va_l[i,:] = Ky.dot(xhy) + Kd.dot(xhd) + b
                vva_l[i,:] = KyV.dot(xhy[s_idx]) + KdV.dot(xhd) + bV

            # ve[i,K] = np.linalg.norm( (v_l[i,:] - v_0[i,3:])/Yvbase )/np.linalg.norm(v_0[i,3:]/Yvbase)
            vve[i,K] = np.linalg.norm( (vv_l[i,:] - vv_0[i,:])/YvbaseV )/np.linalg.norm(vv_0[i,:]/YvbaseV)
            # vae[i,K] = np.linalg.norm( (va_l[i,:] - va_0[i,3:])/Yvbase )/np.linalg.norm(va_0[i,3:]/Yvbase)
            vvae[i,K] = np.linalg.norm( (vva_l[i,:] - vva_0[i,:])/YvbaseV )/np.linalg.norm(vva_0[i,:]/YvbaseV)
            # DVslv_e[i,K] = np.linalg.norm( (Vslv[i,:] - v_0[i,3:]) )/np.linalg.norm(v_0[i,3:])
            
            # plt.plot(abs(v_l[i]/Yvbase),'rx-')
            # plt.plot(abs(v_0[i,3:]/Yvbase),'ko-')
    header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
    lp_str = str(round(lin_point*100).astype(int)).zfill(3)
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
    np.save(sn0+'Ky'+lp_str+'.npy',KyV)
    np.save(sn0+'xhy0'+lp_str+'.npy',xhy0[s_idx])
    np.save(sn0+'bV'+lp_str+'.npy',bV)
    
    np.save(sn0+'vKvbase'+lp_str+'.npy',YvbaseV)
    np.save(sn0+'vYNodeOrder'+lp_str+'.npy',vecSlc(YNodeOrder[3:],v_idx))
    np.save(sn0+'SyYNodeOrder'+lp_str+'.npy',vecSlc(YNodeOrder[3:],p_idx))
    np.save(sn0+'SdYNodeOrder'+lp_str+'.npy',yzD)
    
    if len(H)!=0:
        np.save(sn0+'Kd'+lp_str+'.npy',KdV)
        np.save(sn0+'xhd0'+lp_str+'.npy',xhd0)
print('Complete.\n',time.process_time())

if test_model:
    # plt.figure()
    # plt.plot(k,ve), plt.title(feeder+', My error'), 
    # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    # plt.show()
    # # plt.savefig(fig_loc+'figA')
    plt.figure()
    plt.plot(k,vve), plt.title(feeder+', MyV error')
    plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    plt.show()
    # plt.savefig('figB')
    # plt.figure()
    # plt.plot(k,vae), plt.title(feeder+', Ky error')
    # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    # plt.show()
    # # # plt.savefig('figC')
    plt.figure()
    plt.plot(k,vvae), plt.title(feeder+', KyV error')
    plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    plt.show()
    # plt.savefig('figD')
    # plt.figure()
    # plt.plot(k,DVslv_e), plt.title(feeder+', DVslv error')
    # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    # plt.show()
    # # plt.savefig('figE')

saveCc = True
if saveCc:
    MyCC = My[:,s_idx]
    xhyCC = xhy0[s_idx]
    aCC = a
    V0CC = V0
    YbusCC = Ybus

    dirCC = WD + '\\lin_models\\ccModels\\' + feeder
    snCC = dirCC + '\\' + feeder + lp_taps

    if not os.path.exists(dirCC):
        os.makedirs(dirCC)

    np.save(snCC+'MyCc'+lp_str+'.npy',MyCC)
    np.save(snCC+'xhyCc'+lp_str+'.npy',xhyCC)
    np.save(snCC+'aCc'+lp_str+'.npy',aCC)
    np.save(snCC+'V0Cc'+lp_str+'.npy',V0CC)
    np.save(snCC+'YbusCc'+lp_str+'.npy',YbusCC)