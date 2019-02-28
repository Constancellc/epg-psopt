import win32com.client
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_voltage_funcs import *
import getpass


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
if getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
elif getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

# circuit details copied from linearise_manc_py.
fdr_i = 8
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder=fdrs[fdr_i]
ckt=get_ckt(WD,feeder)

fn_ckt = ckt[0]
fn = ckt[1]
lp_taps='Lpt'

dir0 = WD + '\\lin_models\\' + feeder
sn0 = dir0 + '\\' + feeder + lp_taps
test_model = False

lin_points = np.array([0.3, 0.6, 1.0])
lin_points = np.array([0.6])

for i in range(len(lin_points)):
    print('Creating model', feeder,', linpoint=',lin_points[i])
    
    # 2. Solve at the right linearization point
    DSSText.command='Compile ('+fn+'.dss)'
    lin_point = lin_points[i]
    BB00,SS00 = cpf_get_loads(DSSCircuit)
    cpf_set_loads(DSSCircuit,BB00,SS00,lin_point)
    DSSSolution.Solve()
    DSSText.command='set controlmode=off'
    v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
    v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3
    v_idx = v_idx[v_idx>=0]
    dt = 2*0.00625
    Yvbase = get_Yvbase(DSSCircuit)[3:][v_idx]
    
    # 3. increment tap changers; find new voltages
    j = DSSCircuit.RegControls.First
    dVdt = np.zeros((len(v_idx),DSSCircuit.RegControls.Count))
    
    while j!=0:
        tap0 = DSSCircuit.RegControls.TapNumber
        if abs(tap0)<16:
            tap_hi = tap0+1; tap_lo=tap0-1
            dt = 2*0.00625
        elif tap0==16:
            tap_hi = tap0; tap_lo=tap0-1
            dt = 0.00625
        else:
            tap_hi = tap0+1; tap_lo=tap0
            dt = 0.00625
        DSSCircuit.RegControls.Tapnumber = tap_hi
        DSSSolution.Solve()
        V1 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:])[v_idx] # NOT the same order as AllBusVmag!
        DSSCircuit.RegControls.Tapnumber = tap_lo
        DSSSolution.Solve()
        V0 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:])[v_idx]
        # dVdt[:,j-1] = (V1 - V0)/(dt*Yvbase)
        dVdt[:,j-1] = (V1 - V0)/(dt)
        
        DSSCircuit.RegControls.Tapnumber = tap0
        j = DSSCircuit.RegControls.Next
        
    lp_str = str(round(lin_point*100).astype(int)).zfill(3)
    header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
    np.save(sn0+'Kt'+lp_str+'.npy',dVdt)
    if test_model:
        print(lin_point)
        plt.xlabel('Bus id'), plt.ylabel('dVdt'), plt.grid(True)
        plt.plot(dVdt), plt.grid(True), plt.show()

# # for debugging
# YZ = DSSCircuit.YNodeOrder
# YZidx = vecSlc(DSSCircuit.YNodeOrder[3:],v_idx)
# YZregs0 = vecSlc(YZidx,dVdt[:,0]>0.5)
# YZregs1 = vecSlc(YZidx,dVdt[:,1]>0.5)
# YZregs2 = vecSlc(YZidx,dVdt[:,2]>0.5)

# # for visualizing the matrices
# ax1 = plt.subplot(121)
# ax2 = plt.subplot(122)
# for i in range(dVdt.shape[1]):
    # ax1.plot(dVdt[:,i]/Yvbase,'x')
    # ax2.plot(dVdt[:,i]/Yvbase - np.round(dVdt[:,i]/Yvbase),'x')
# plt.show()

print('Complete.')