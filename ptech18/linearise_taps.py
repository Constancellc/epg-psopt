import win32com.client
import numpy as np
import dss_python_funcs as dspf
import matplotlib.pyplot as plt

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
fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\13Bus_copy\\IEEE13Nodeckt'

Nreg = 0 # which transformer/regulator to consider;

DSSText.command='Compile ('+fn+'.dss)'
DSSSolution.Solve()
V0 = np.array(DSSCircuit.AllBusVmagPu)

DSSText.command='set controlmode=off'

# 3. increment tap changers; find new voltages
DSSCircuit.Transformers.First

for i in range(Nreg):
    DSSCircuit.Transformers.Next

tap_0 = DSSCircuit.Transformers.Tap
dT = np.array([-0.0625,0.0625]) # one tap each side
dt = np.diff(dT)

dV = np.empty((2,len(V0)))
dVdt = np.empty(len(V0))

i=0
for T in dT:
    DSSCircuit.Transformers.Tap=tap_0 + T
    DSSSolution.Solve()
    dV[i] = np.array(DSSCircuit.AllBusVmagPu)
    i+=1

dVdt = (dV[1]-dV[0])/dt

sY,sD,iY,iD = dspf.get_sYsD(DSSCircuit)
    
1dVdt = dVdt[sY.nonzero()]
# np.savetxt(WD+'\\kT_lin_'+DSSCircuit.name+'.txt',dVdt)

# plt.xlabel('Bus id'), plt.ylabel('dVdt'), plt.grid(True)
# plt.plot(dVdt), plt.show()