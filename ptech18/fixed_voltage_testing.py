# Testing the fixed voltage method.

# steps: 
# 1. load linear model
# 2. split into upstream/downstream of regulator(s)
# 3. reorder & remove elements as appropriate
# 4. run continuation analysis.

import getpass
import numpy as np
import win32com.client
import matplotlib.pyplot as plt
import time
from dss_python_funcs import *
from dss_voltage_funcs import *
from scipy import sparse
from cvxopt import spmatrix
from scipy import random
import scipy.linalg as spla

# based on monte_carlo.py
print('Start.\n',time.process_time())

FD = r"C:\Users\\"+getpass.getuser()+"\Documents\DPhil\malcolm_updates\wc181217\\"
if getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
elif getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
test_model_plt = True
fdr_i = 11
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod']
feeder=fdrs[fdr_i]

k = np.arange(-1.5,1.6,0.025)
# k = np.arange(0,1.0,1.0)

fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]
lin_point=0.6
lp_taps='Lpt'
   
fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

# get_sYsD(DSSCircuit)

DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)
# BB00,SS00 = cpf_get_loads(DSSCircuit)
DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point)
    DSSSolution.Solve()
YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)

DSSText.command='set controlmode=off'
YZ = DSSCircuit.YNodeOrder

zoneList, regZonIdx0, zoneTree = get_regZneIdx(DSSCircuit)
regZonIdx = (np.array(regZonIdx0[3:])-3).tolist()

regIdx = get_regIdx(DSSCircuit)
reIdx = (np.array(get_reIdx(regIdx,len(YZ))[3:])-3).tolist()

YZnew = vecSlc(YZ[3:],reIdx) # checksum

Ky,Kd,Kt,bV,xhy0,xhd0 = loadLinMagModel(feeder,lin_point,WD,lp_taps)

# get index shifts
v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3
v_idx = v_idx[v_idx>=0]

v_idx_shf,v_idx_new = idx_shf(v_idx,reIdx)

sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)

p_idx_yz = np.array(sY[3:].nonzero())
p_idx_shf,p_idx_new = idx_shf(p_idx_yz[0],reIdx)
s_idx_shf = np.concatenate((p_idx_shf,p_idx_shf+len(p_idx_shf)))

p_idx = np.array(sY[3:].nonzero())
s_idx = np.concatenate((p_idx,p_idx+len(sY)-3),axis=1)[0]
s_idx_new = np.concatenate((p_idx_new,p_idx_new+len(sY)-3))

yzI = yzD2yzI(yzD,node_to_YZ(DSSCircuit))
yzI_shf,yzI_new = idx_shf(yzI,reIdx)
sD_idx_shf = np.concatenate((yzI_shf,yzI_shf+len(yzI_shf)))

Sd = YNodeVnom[yzI]*(iD.conj())/1e3

Kp = Sd[yzI_shf].real/sD[yzI_shf].real
Kq = Sd[yzI_shf].imag/sD[yzI_shf].imag

xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))

YZp = vecSlc(YZ[3:],p_idx_new) # verified
YZd = vecSlc(YZ,yzI_new)

zoneSet = {'msub':[],'mreg':[0],'mregx':[0,3],'mregy':[0,6]} # this will need automating...!
regIdxMatY = get_regIdxMatS(YZp,zoneList,zoneSet,np.ones(len(YZp)),np.ones(len(YZp)),len(regIdx))
regIdxMatD = get_regIdxMatS(YZd,zoneList,zoneSet,Kp,Kq,len(regIdx))
xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))
regIdxMat = np.concatenate((regIdxMatY,regIdxMatD),axis=1)

# Sreg = regIdxMat.dot(xhR)/1e3 # for debugging.

YZv_idx = vecSlc(vecSlc(YZ[3:],v_idx),v_idx_shf)
KyR = Ky[v_idx_shf,:][:,s_idx_shf]
KdR = Kd[v_idx_shf,:][:,sD_idx_shf]
bVR = bV[v_idx_shf]
KtR = Kt[v_idx_shf,:]

regVreg = get_regVreg(DSSCircuit)
Anew,Bnew = kron_red(KyR,KdR,KtR,bVR,regVreg)

# now, check these are working
print('Start Testing.\n',time.process_time())

ve=np.zeros([k.size])
veN=np.zeros([k.size])

ve_ctl=np.zeros([k.size])
veN_ctl=np.zeros([k.size])

v_0 = np.zeros((len(k),len(YZ)))

vv_0 = np.zeros((len(k),len(v_idx)))
vv_0R = np.zeros((len(k),len(v_idx)))
vv_0_ctl = np.zeros((len(k),len(v_idx)))
vv_0R_ctl = np.zeros((len(k),len(v_idx)))

vv_l = np.zeros((len(k),len(v_idx)))
vv_lN = np.zeros((len(k),len(v_idx)))
vv_l_ctr = np.zeros((len(k),len(v_idx)))
vv_lN_ctr = np.zeros((len(k),len(v_idx)))



RegSat = np.zeros((len(k),len(regIdx)),dtype=int)

Convrg = []
TP = np.zeros(len(k),dtype=complex)
TL = np.zeros(len(k),dtype=complex)

for i in range(len(k)):
    cpf_set_loads(DSSCircuit,BB0,SS0,k[i])
    DSSSolution.Solve()
    Convrg.append(DSSSolution.Converged)
    TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
    TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
    
    v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray)) # for some reason complains about complex
    vv_0[i,:] = v_0[i,3:][v_idx]
    vv_0R[i,:] = vv_0[i,:][v_idx_shf]
    
    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    xhy = -1e3*s_2_x(sY[3:])
    
    if len(H)==0:
        vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
    else:
        xhd = -1e3*s_2_x(sD) # not [3:] like sY
        vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
        
        xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
        vv_lN[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
    
    ve[i] = np.linalg.norm( vv_l[i,:] - vv_0[i,:] )/np.linalg.norm(vv_0[i,:])
    veN[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])


DSSText.command='set controlmode=static'
for i in range(len(k)):
    cpf_set_loads(DSSCircuit,BB0,SS0,k[i])
    DSSSolution.Solve()
    Convrg.append(DSSSolution.Converged)
    TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
    TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])

    v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray)).real # for some reason complains about complex
    vv_0_ctl[i,:] = v_0[i,3:][v_idx]
    vv_0R_ctl[i,:] = vv_0_ctl[i,:][v_idx_shf]

    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    xhy = -1e3*s_2_x(sY[3:])

    RegSat[i] = getRegSat(DSSCircuit)

    if len(H)==0:
        vv_l_ctr[i,:] = Ky.dot(xhy[s_idx]) + bV
    else:
        xhd = -1e3*s_2_x(sD) # not [3:] like sY
        vv_l_ctr[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
        xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
        vv_lN_ctr[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
    ve_ctl[i] = np.linalg.norm( vv_l_ctr[i,:] - vv_0_ctl[i,:] )/np.linalg.norm(vv_0_ctl[i,:])
    veN_ctl[i] = np.linalg.norm( vv_lN_ctr[i,:] - vv_0R_ctl[i,:] )/np.linalg.norm(vv_0R_ctl[i,:])
print('Testing Complete.\n',time.process_time())

unSat = RegSat.min(axis=1)==1
sat = RegSat.min(axis=1)==0
Yvbase = get_Yvbase(DSSCircuit)[3:][v_idx]
Yvbase_new = get_Yvbase(DSSCircuit)[3:][v_idx_new]

if test_model_plt:
    plt.figure()
    pltA, = plt.plot(k,ve,'b')
    pltB, = plt.plot(k,veN,'r') # plt.plot(k,veN)
    plt.title(feeder+', K error')
    plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    plt.legend([pltA,pltB],['Lin fixed','Lin not fixed'])
    plt.show()
    # plt.savefig(fig_loc+'reg_off_err.png')
    
    plt.figure()
    plt.plot(k,ve,'k:')
    plt.plot(k[unSat],ve_ctl[unSat],'b') 
    plt.plot(k[unSat],veN_ctl[unSat],'r')
    plt.plot(k[sat],ve_ctl[sat],'b-.')
    plt.plot(k[sat],veN_ctl[sat],'r-.')
    plt.title(feeder+', K error')
    plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    plt.show()
    # plt.savefig(fig_loc+'reg_on_err.png')
    
    krnd = np.around(k,5) # this breaks at 0.000 for the error!
    idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )
    
    # plt.figure(figsize=(12,4))
    # for i in range(len(idxs)):
        # plt.subplot(1,len(idxs),i+1)
        # plt.plot(vv_0[idxs[i]]/Yvbase,'o')
        # plt.plot(vv_l[idxs[i]]/Yvbase,'x')
        # plt.plot(vv_lN[idxs[i]]/Yvbase,'+')
        # plt.xlabel('Bus index'); 
        # plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
        # if i==0:
            # plt.ylabel('Voltage Magnitude (pu)')
            # plt.legend(('DSS, fxd regs,','Lin fxd','Lin not fxd'))
    # plt.show()
    
    # plt.figure(figsize=(12,4))
    # for i in range(len(idxs)):
        # plt.subplot(1,len(idxs),i+1)
        # plt.title('K = '+str(krnd[idxs[i]]))
        # plt.plot(vv_0R_ctl[idxs[i]]/Yvbase_new,'o')
        # plt.plot(vv_lN_ctr[idxs[i]]/Yvbase_new,'*')
        # plt.plot(vv_l_ctr[idxs[i]]/Yvbase,'x')
        # plt.xlabel('Bus index'); 
        # plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
        # if i==0:
            # plt.ylabel('Voltage Magnitude (pu)')
            # plt.legend(('DSS, not fxd regs','Lin not fxd','Lin fxd'))
    # plt.show()
    # # plt.savefig(fig_loc+'err_exmpl.png')
