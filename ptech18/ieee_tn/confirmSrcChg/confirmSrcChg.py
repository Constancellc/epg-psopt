import win32com.client, sys, os
import numpy as np
import matplotlib.pyplot as plt

WD = os.path.dirname(sys.argv[0])
sys.path.insert(0,os.path.dirname(os.path.dirname(WD)))

from dss_python_funcs import *

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

fn0 = os.path.join(WD,'ckt7','Master_ckt7.dss')
fn1 = os.path.join(WD,'ckt7','Master_ckt7_src.dss')
fn2 = os.path.join(WD,'ckt7','Master_ckt7_nosrc.dss')

DSSText.command='Compile '+fn0
DSSSolution.Solve
vbase0 = get_Yvbase(DSSCircuit)
Vmag0 = abs(tp_2_ar(DSSCircuit.YNodeVarray))

DSSText.command='Compile '+fn1
DSSSolution.Solve
vbase1 = get_Yvbase(DSSCircuit)
Vmag1 = abs(tp_2_ar(DSSCircuit.YNodeVarray))

DSSText.command='Compile '+fn2
DSSSolution.Solve
vbase2 = get_Yvbase(DSSCircuit)
Vmag2 = abs(tp_2_ar(DSSCircuit.YNodeVarray))

plt.subplot(121)
plt.plot(Vmag0/vbase0)
plt.plot(Vmag1[3:]/vbase1[3:])
plt.plot(Vmag2/vbase2)
plt.ylabel('|V| (pu)')
plt.xlabel('Bus ID')
plt.legend(('Nominal Z','Line source Z','No source Z'))

plt.subplot(122)
plt.plot((Vmag0/vbase0) - (Vmag1[3:]/vbase1[3:]))
plt.ylabel('$\Delta$ |V| (pu)')
plt.xlabel('Bus ID')
plt.tight_layout()
plt.show()