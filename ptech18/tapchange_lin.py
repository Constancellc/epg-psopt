import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse
import matplotlib.pyplot as plt

try:
	DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
	print("Unable to stat the OpenDSS Engine")
	raise SystemExit

DSSText = DSSObj.Text
DSSCircuit=DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

fig_loc = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc181126\\figures\\"
# fig_loc = r"C:\\Users\Matt\Documents\DPhil\malcolm_updates\wc181126\\tap_changes\\"


# Things to do: 
# 1. load a circuit;
# WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
# fn = WD+'\\LVTestCase_copy\\master_z'
fn = WD+'\\13Bus_copy\\IEEE13Nodeckt'

Nreg = 4

# 2. solve; find nominal voltages; 
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
dT = np.arange(-0.1,0.101,0.04)
dt = 0.01

dV = np.empty((len(dT),len(V0)))
dVdt = np.empty((len(dT),len(V0)))

i=0
for dt in dT:
	DSSCircuit.Transformers.Tap=tap_0 + dt
	DSSSolution.Solve()
	dV[i] = V0 - np.array(DSSCircuit.AllBusVmagPu)
	dVdt[i] = -dV[i]/dt
	i+=1


# 4. plot changes to voltages versus node number
# plt.scatter(dT,dVdt[:,0]) # fast
# fig = plt.figure()
# for i in range(len(dVdt[0])):
	# plt.scatter(dT,dVdt[:,i])
# plt.grid()
# plt.xlabel('Tap change, dt (%)')
# plt.ylabel('Change in voltage per unit tap, dVdt')
# fig.gca().set_axisbelow(True)
# plt.show()
# fig.savefig(fig_loc+'dVdt_'+DSSCircuit.Name+'_'+DSSCircuit.Transformers.Name+'.pdf')

# fig = plt.figure()
# for i in range(len(dV[0])):
	# plt.scatter(dT,dV[:,i])
# plt.grid()
# plt.axis('equal')
# plt.xlabel('Tap change, dt (%)')
# plt.ylabel('Change in voltage, dV')
# fig.gca().set_axisbelow(True)
# plt.show()
# fig.savefig(fig_loc+'dV_'+DSSCircuit.Name+'_'+DSSCircuit.Transformers.Name+'.pdf')
# fig.savefig(fig_loc+'dV_eulv.pdf')
















