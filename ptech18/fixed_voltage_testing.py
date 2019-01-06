# Testing the fixed voltage method.

# steps: 
# 1. load linear model
# 2. split into upstream/downstream of regulator(s)
# 3. reorder & remove elements as appropriate
# 4. run continuation analysis.

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

FD = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc181217\\"
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
# WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"


DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
fdr_i = 11
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod']
feeder=fdrs[fdr_i]

k = np.arange(-1.5,1.6,0.025)
# k = np.arange(0,1.0,1.0)
test_model_plt = True

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]
lin_point=0.6
lp_taps='Lpt'
   
fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

get_sYsD(DSSCircuit)

DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)
# BB00,SS00 = cpf_get_loads(DSSCircuit)
DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point)
    DSSSolution.Solve()

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

YZp = vecSlc(YZ[3:],p_idx_new) # verified
YZd = vecSlc(YZ,yzI_new) 

regIdxPy = np.zeros((len(regIdx),len(YZp)))

zoneY = []
for yz in YZp:
    ph = int(yz[-1])
    for key in zoneList.keys():
        if yz in zoneList[key]:
            zoneY = zoneY+[key]
zoneD = []
for yz in YZd:
    ph = int(yz[-1])
    for key in zoneList.keys():
        if yz in zoneList[key]:
            zoneD = zoneD+[key]
print(zoneY+zoneD)

# sD_idx_shf = np.concatenate((yzI_shf,yzI_shf+len(yzI_shf)))

# YZv_idx = vecSlc(vecSlc(YZ[3:],v_idx),v_idx_shf)
# KyR = Ky[v_idx_shf,:][:,s_idx_shf]
# KdR = Kd[v_idx_shf,:][:,sD_idx_shf]
# bVR = bV[v_idx_shf]
# KtR = Kt[v_idx_shf,:]

# regVreg = get_regVreg(DSSCircuit)
# Anew,Bnew = kron_red(KyR,KdR,KtR,bVR,regVreg)

# # now, check these are working
# print('Start Testing.\n',time.process_time())

# ve=np.zeros([k.size])
# veR=np.zeros([k.size])
# veN=np.zeros([k.size])

# ve_ctl=np.zeros([k.size])
# veN_ctl=np.zeros([k.size])

# v_0 = np.zeros((len(k),len(YZ)))

# vv_0 = np.zeros((len(k),len(v_idx)))
# vv_0R = np.zeros((len(k),len(v_idx)))
# vv_l = np.zeros((len(k),len(v_idx)))
# vv_lR = np.zeros((len(k),len(v_idx)))
# vv_lN = np.zeros((len(k),len(v_idx)))
# RegSat = np.zeros((len(k),len(regIdx)),dtype=int)

# Convrg = []
# TP = np.zeros(len(k),dtype=complex)
# TL = np.zeros(len(k),dtype=complex)

# for i in range(len(k)):
    # cpf_set_loads(DSSCircuit,BB0,SS0,k[i])
    # DSSSolution.Solve()
    # Convrg.append(DSSSolution.Converged)
    # TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
    # TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
    
    # v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray)) # for some reason complains about complex
    # vv_0[i,:] = v_0[i,3:][v_idx]
    # vv_0R[i,:] = vv_0[i,:][v_idx_shf]
    
    # sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    # xhy = -1e3*s_2_x(sY[3:])
    
    # if len(H)==0:
        # vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
    # else:
        # xhd = -1e3*s_2_x(sD) # not [3:] like sY
        # vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
        # vv_lR[i,:] = KyR.dot(xhy[s_idx_new]) + KdR.dot(xhd[sD_idx_shf]) + bVR
        
        # xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
        # vv_lN[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
    
    # ve[i] = np.linalg.norm( vv_l[i,:] - vv_0[i,:] )/np.linalg.norm(vv_0[i,:])
    # veR[i] = np.linalg.norm( vv_lR[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
    # veN[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])


# DSSText.command='set controlmode=static'
# for i in range(len(k)):
    # cpf_set_loads(DSSCircuit,BB0,SS0,k[i])
    # DSSSolution.Solve()
    # Convrg.append(DSSSolution.Converged)
    # TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
    # TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
    
    # v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray)).real # for some reason complains about complex
    # vv_0[i,:] = v_0[i,3:][v_idx]
    # vv_0R[i,:] = vv_0[i,:][v_idx_shf]
    
    # sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
    # xhy = -1e3*s_2_x(sY[3:])
    
    # RegSat[i] = getRegSat(DSSCircuit)
    
    # if len(H)==0:
        # vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
    # else:
        # xhd = -1e3*s_2_x(sD) # not [3:] like sY
        # vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
        # xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
        # vv_lN[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
    # ve_ctl[i] = np.linalg.norm( vv_l[i,:] - vv_0[i,:] )/np.linalg.norm(vv_0[i,:])
    # veN_ctl[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
# print('Testing Complete.\n',time.process_time())

# unSat = RegSat.min(axis=1)==1
# sat = RegSat.min(axis=1)==0

# if test_model_plt:
    # plt.figure()
    # plt.plot(k,ve), plt.plot(k,veN), plt.title(feeder+', K error')
    # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    # plt.show()
    # plt.figure()
    # plt.plot(k,ve,':')
    # plt.plot(k[unSat],ve_ctl[unSat],'x-') 
    # plt.plot(k[unSat],veN_ctl[unSat],'x-')
    # plt.plot(k[sat],ve_ctl[sat],'x-')
    # plt.plot(k[sat],veN_ctl[sat],'x-')
    # plt.title(feeder+', K error')
    # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
    # plt.show()
