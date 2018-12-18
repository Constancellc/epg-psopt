import win32com.client
import numpy as np
# import dss_python_funcs as dspf
import matplotlib.pyplot as plt
from dss_python_funcs import *

# NB at the moment on considers the case where there is a single transformer tap to play with.

try:
	DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
	print("Unable to stat the OpenDSS Engine")
	raise SystemExit

DSSText = DSSObj.Text
DSSCircuit=DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

# Things to do: 
# 1. load a circuit;
# WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

fdr_i = 4 # do NOT set equal to 2!
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4']
ckts = {'feeder_name':['fn_ckt','fn']}
ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
ckts[fdrs[1]] = feeder_to_fn(WD,fdrs[1])
ckts[fdrs[2]] = feeder_to_fn(WD,fdrs[2])
ckts[fdrs[3]] = feeder_to_fn(WD,fdrs[3])
ckts[fdrs[4]] = feeder_to_fn(WD,fdrs[4])

# fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\13Bus_copy\\IEEE13Nodeckt'
# feeder='eulv'
fn_ckt = ckts[fdrs[fdr_i]][0]
fn = ckts[fdrs[fdr_i]][1]
feeder=fdrs[fdr_i]

sn0 = WD + '\\lin_models\\' + feeder

Nreg = 0 # which transformer/regulator to consider;
test_model = False

lin_points = np.array([0.3,0.6,1.])

for i in range(len(lin_points)):
    print('Creating model, linpoint=',lin_points[i])
    # 2. Solve at the right linearization point
    DSSText.command='Compile ('+fn+'.dss)'
    lin_point = lin_points[i]
    BB00,SS00 = cpf_get_loads(DSSCircuit)
    k00 = lin_point/SS00[1].real
    cpf_set_loads(DSSCircuit,BB00,SS00,k00)
    DSSSolution.Solve()
    V0 = np.array(DSSCircuit.AllBusVmagPu)[3:]
    DSSText.command='set controlmode=off'
    # 3. increment tap changers; find new voltages
    DSSCircuit.Transformers.First

    for j in range(Nreg):
        DSSCircuit.Transformers.Next

    tap_0 = DSSCircuit.Transformers.Tap
    dT = np.array([-0.0625,0.0625]) # one tap each side
    dt = np.diff(dT)
    dV = np.empty((2,len(V0)))
    dVdt = np.empty(len(V0))
    j=0
    for T in dT:
        DSSCircuit.Transformers.Tap=tap_0 + T
        DSSSolution.Solve()
        # dV[j] = np.array(DSSCircuit.AllBusVmagPu)[3:]
        dV[j] = np.array(DSSCircuit.AllBusVmag)[3:]
        j+=1
    dVdt = (dV[1]-dV[0])/dt
    
    v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
    v_idx = np.array(get_element_idxs(DSSCircuit,v_types)) - 3
    v_idx = v_idx[v_idx>=0]
    
    dVdt = dVdt[v_idx]
    lp_str = str(round(lin_point*100).astype(int)).zfill(3)
    header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
    np.savetxt(sn0+'Kt'+lp_str+'.txt',dVdt,header=header_str)

    if test_model:
        print(lin_point)
        plt.xlabel('Bus id'), plt.ylabel('dVdt'), plt.grid(True)
        plt.plot(dVdt), plt.show()
print('Complete.')