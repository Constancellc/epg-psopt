import win32com.client
import numpy as np
import dss_python_funcs
from dss_python_funcs import *

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
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
# WD = "C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\13Bus_copy\\IEEE13Nodeckt'

Nreg = 1 # which transformer/regulator to consider;

DSSText.command='Compile ('+fn+'.dss)'
DSSSolution.Solve()
V0 = np.array(DSSCircuit.AllBusVmagPu)

DSSText.command='set controlmode=off'

# 3. increment tap changers; find new voltages
DSSCircuit.Transformers.First
for i in range(Nreg):
    DSSCircuit.Transformers.Next
# DSSCircuit.Transformers.Next

tap_0 = DSSCircuit.Transformers.Tap
dT = np.array([-0.0625,0.0625]) # one tap each side
dt = np.diff(dT)

dV = np.empty((2,len(V0)))
dVdt = np.empty(len(V0))

for T in dT:
    DSSCircuit.Transformers.Tap=tap_0 + dt
    DSSSolution.Solve()
	dV[i] = V0 - np.array(DSSCircuit.AllBusVmagPu)
	dVdt[i] = -dV[i]/dt
	i+=1

