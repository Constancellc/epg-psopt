# Testing the fixed voltage method.

# steps: 
# 1. load linear model
# 2. split into upstream/downstream of regulator(s)
# 3. find LTC matrices
# 4. reorder & remove elements as appropriate
# 5. run continuation analysis.

# A bunch of notes on the main method in WB 7-01-19 and 15-01-19

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

print('Start.\n',time.process_time())

FD = r"C:\Users\\"+getpass.getuser()+"\Documents\DPhil\malcolm_updates\wc181217\\"
if getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
    fig_loc = r"C:\Users\Matt\Documents\DPhil\malcolm_updates\wc190128\\ltcVoltageTesting_"
elif getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
    fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\ltcVoltageTesting_"

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
test_model_plt = True
# test_model_plt = False
test_model_bus = True
# test_model_bus = False
fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod']
feeder=fdrs[fdr_i]

k = np.arange(-1.5,1.6,0.025)
k = np.arange(-1.5,1.6,0.1)
# k = np.arange(0,1.0,1.0)

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]
lin_point=0.6
lp_taps='Lpt'
sn0 = WD + '\\lin_models\\' + feeder

# get_sYsD(DSSCircuit)

# 1. Nominal Voltage Solution at Linearization point. Load Linear models.
DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point)
    DSSSolution.Solve()
YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)

DSSText.command='set controlmode=off'
YZ = DSSCircuit.YNodeOrder

LM = loadLinMagModel(feeder,lin_point,WD,lp_taps)
Ky=LM['Ky'];Kd=LM['Kd'];Kt=LM['Kt'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']

# 2. Split the model into upstream/downstream.
zoneList, regZonIdx0, zoneTree = get_regZneIdx(DSSCircuit)
regZonIdx = (np.array(regZonIdx0[3:])-3).tolist()

regIdx = get_regIdx(DSSCircuit)
reIdx = (np.array(get_reIdx(regIdx,len(YZ))[3:])-3).tolist()

YZnew = vecSlc(YZ[3:],reIdx) # checksum

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

Kp = Sd[yzI_shf].real/sD[yzI_shf].real # not sure here?
Kq = Sd[yzI_shf].imag/sD[yzI_shf].imag # not sure here?

xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))

YZp = vecSlc(YZ[3:],p_idx_new) # verified
YZd = vecSlc(YZ,yzI_new) 

Yvbase_new = get_Yvbase(DSSCircuit)[3:][v_idx_new]
# 3. FIND LTC MATRICES ====
rReg,xReg = getRxVltsMat(DSSCircuit)
Rreg = np.diag(rReg)
Xreg = np.diag(xReg)

zoneSet = getZoneSet(feeder,DSSCircuit,zoneTree) # NB this is not yet automated

regIdxMatY = get_regIdxMatS(YZp,zoneList,zoneSet,np.ones(len(YZp)),np.ones(len(YZp)),len(regIdx))
regIdxMatD = get_regIdxMatS(YZd,zoneList,zoneSet,Kp,Kq,len(regIdx))

xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))
regIdxMat = np.concatenate((regIdxMatY,regIdxMatD),axis=1) # matrix used for finding power through regulators
# Sreg = regIdxMat.dot(xhR)/1e3 # for debugging.

regIdxMatYs = regIdxMatY[:,0:len(xhy0)//2].real
regIdxMatDs = regIdxMatD[:,0:len(xhd0)//2].real
regIdxMatVlts = -np.concatenate( (Rreg.dot(regIdxMatYs),Xreg.dot(regIdxMatYs),Rreg.dot(regIdxMatDs),Xreg.dot(regIdxMatDs)),axis=1 )
# dVregRx = regIdxMatVlts.dot(xhR) # for debugging; output in volts.

# 4. PERFORM REINDEXING/KRON REDUCTION OF PF MATRICES ==============
regVreg = get_regVreg(DSSCircuit)
YZv_idx = vecSlc(vecSlc(YZ[3:],v_idx),v_idx_shf)
KyR = Ky[v_idx_shf,:][:,s_idx_shf]
KdR = Kd[v_idx_shf,:][:,sD_idx_shf]
bVR = bV[v_idx_shf]
KtR = Kt[v_idx_shf,:]

get_regVreg(DSSCircuit)
Anew,Bnew = kron_red(KyR,KdR,KtR,bVR,regVreg)
Altc,Bltc = kron_red_ltc(KyR,KdR,KtR,bVR,regVreg,regIdxMatVlts)

# 5. VALIDATION ==============
v_0 = np.zeros((len(k),len(YZ)))

veR=np.zeros([k.size])
veN=np.zeros([k.size])
veL=np.zeros([k.size])

vv_0 = np.zeros((len(k),len(v_idx)))
vv_0R = np.zeros((len(k),len(v_idx))) # reordered

vv_l = np.zeros((len(k),len(v_idx)))
vv_lR = np.zeros((len(k),len(v_idx))) # reordered (fixed taps)
vv_lN = np.zeros((len(k),len(v_idx))) # free, no LTC
vv_lL = np.zeros((len(k),len(v_idx))) # free, with LTC

RegSat = np.zeros((len(k),len(regIdx)),dtype=int)

Convrg = []
TP = np.zeros(len(k),dtype=complex)
TL = np.zeros(len(k),dtype=complex)

DSSText.command='set controlmode=static'

if test_model_plt or test_model_bus:
    print('--- Start Testing --- \n',time.process_time())
    for i in range(len(k)):
        print(i,'/',len(k)-1)
        cpf_set_loads(DSSCircuit,BB0,SS0,k[i])
        DSSSolution.Solve()
        Convrg.append(DSSSolution.Converged)
        TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
        TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])

        v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray)).real # for some reason complains about complex
        vv_0R[i,:] = v_0[i,3:][v_idx_new]

        sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
        xhy = -1e3*s_2_x(sY[3:]) # in W

        RegSat[i] = getRegSat(DSSCircuit)

        if len(H)==0:
            vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
        else:
            xhd = -1e3*s_2_x(sD) # not [3:] like sY
            vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
            vv_lR[i,:] = vv_l[i,:][v_idx_shf]
            xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
            vv_lN[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
            # vv_lL[i,:] = np.concatenate((Altc.dot(xnew) + Bltc,np.array(regVreg) + regIdxMatVlts.dot(xnew) ))
            vv_lL[i,:] = Altc.dot(xnew) + Bltc # NB note no need to append regVreg
        veR[i] = np.linalg.norm( vv_lR[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
        veN[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
        veL[i] = np.linalg.norm( vv_lL[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
    print('Testing Complete.\n',time.process_time())
    unSat = RegSat.min(axis=1)==1
    sat = RegSat.min(axis=1)==0

# SAVE MODEL ============
dir0 = WD + '\\lin_models\\' + feeder + '\\ltc_model'
sn0 = dir0 + '\\' + feeder + lp_taps + 'Ltc'
lp_str = str(round(lin_point*100)).zfill(3)

if not os.path.exists(dir0):
        os.makedirs(dir0)

np.save(sn0+'A'+lp_str+'.npy',Altc)
np.save(sn0+'B'+lp_str+'.npy',Bltc)
np.save(sn0+'s_idx'+lp_str+'.npy',s_idx_new)
np.save(sn0+'v_idx'+lp_str+'.npy',v_idx_new)
np.save(sn0+'xhy0'+lp_str+'.npy',xhy0[s_idx_shf])
np.save(sn0+'xhd0'+lp_str+'.npy',xhd0[sD_idx_shf])
np.save(sn0+'YZ'+lp_str+'.npy',YZnew)
np.save(sn0+'Vbase'+lp_str+'.npy',Yvbase_new)

# np.save(sn0+'Avreg'+lp_str+'.npy',regIdxMatVlts)
# np.save(sn0+'Bvreg'+lp_str+'.npy',regVreg)

# PLOTTING ============
if test_model_plt:
    plt.figure()
    plt.plot(k[unSat],veR[unSat],'b') 
    plt.plot(k[unSat],veN[unSat],'r')
    plt.plot(k[unSat],veL[unSat],'g')
    # plt.plot(k[sat],ve[sat],'b-.')
    # plt.plot(k[sat],veN[sat],'r-.')
    # plt.plot(k[sat],veL[sat],'g-.')
    plt.plot(k[sat],veR[sat],'b.')
    plt.plot(k[sat],veN[sat],'r.')
    plt.plot(k[sat],veL[sat],'g.')
    plt.title(feeder+', K error')
    plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    plt.legend(('Fixed taps','Control, R, X = 0','Control, actual R, X'))
    plt.show()
    # plt.savefig(fig_loc+'reg_on_err.png')
    
    krnd = np.around(k,5) # this breaks at 0.000 for the error!
    idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )

if test_model_bus:
    krnd = np.around(k,5) # this breaks at 0.000 for the error!
    idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )
    
    plt.figure(figsize=(12,4))
    for i in range(len(idxs)):
        plt.subplot(1,len(idxs),i+1)
        plt.title('K = '+str(krnd[idxs[i]]))
        plt.plot(vv_0R[idxs[i]]/Yvbase_new,'o')
        plt.plot(vv_lR[idxs[i]]/Yvbase_new,'.')
        plt.plot(vv_lN[idxs[i]]/Yvbase_new,'+')
        plt.plot(vv_lL[idxs[i]]/Yvbase_new,'x')
        plt.xlabel('Bus index'); plt.grid(True)
        plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
        if i==0:
            plt.ylabel('Voltage Magnitude (pu)')
            plt.legend(('DSS, fxd regs,','Fixed Tap','Free, no LTC','Free, with LTC'))
    plt.show()
    

