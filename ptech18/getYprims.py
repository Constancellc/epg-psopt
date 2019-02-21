import numpy as np
from dss_python_funcs import *
import getpass
import win32com.client
import pickle

if getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
    sn = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190204\\charFuncMcVal_"
elif getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
    sn = r"C:\Users\Matt\Documents\DPhil\malcolm_updates\wc190204\\charFuncMcVal_"

# CHOOSE Network
fdr_i = 0
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1']
feeder = fdrs[fdr_i]
# feeder = '213'

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution
DSSText.command='Compile ('+fn+'.dss)'

LNS = DSSCircuit.Lines
# ACE = DSSCircuit.ActiveElement
# SetACE = DSSCircuit.SetActiveElement


lnsYprims = {}

i = LNS.First
while i:
    # SetACE('Line.'+LNS.Name)
    YprimFlat = tp_2_ar(LNS.Yprim)
    n = int(np.sqrt(len(YprimFlat)))
    Yprim = np.reshape(YprimFlat,(n,n))
    bus1 = LNS.bus1
    bus2 = LNS.bus2
    lnsYprims[LNS.Name] = [bus1,bus2,Yprim]
    i = LNS.Next

f = open("lnsYprims.pkl","wb")
pickle.dump(lnsYprims,f)
f.close()

g = open("lnsYprims.pkl",'rb')
data = pickle.load(g)
g.close()