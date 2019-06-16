import numpy as np
from dss_python_funcs import *
import getpass
import win32com.client
import pickle
import sys,os,getpass

WD = os.path.dirname(sys.argv[0])

# CHOOSE Network
fdr_i = 0
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']
feeder = fdrs[fdr_i]
# feeder = '213'
reg_point = 'Nmt'
lin_point = '060'
lin_point = '100'

SD = os.path.join(WD,'lin_models','ccModels',feeder)

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution
DSSText.Command='Compile ('+fn+'.dss)'

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

# f = open("lnsYprims.pkl","wb")
f = open(os.path.join(SD,'lnsYprims.pkl'),"wb")
pickle.dump(lnsYprims,f)
f.close()

# g = open("lnsYprims.pkl",'rb')
g = open(os.path.join(SD,'lnsYprims.pkl'),'rb')
data = pickle.load(g)
g.close()



# TESTING
# LOAD the actual line flow at the solution
V0 = np.load(os.path.join(SD,feeder+reg_point+'V0Cc'+lin_point+'.npy'))
alin = np.load(os.path.join(SD,feeder+reg_point+'aCc'+lin_point+'.npy'))
xhy = np.load(os.path.join(SD,feeder+reg_point+'xhyCc'+lin_point+'.npy'))
Ybus = np.load(os.path.join(SD,feeder+reg_point+'YbusCc'+lin_point+'.npy'))
My = np.load(os.path.join(SD,feeder+reg_point+'MyCc'+lin_point+'.npy'))
YNodeOrder = np.load(os.path.join(SD,feeder+reg_point+'YNodeOrderCc'+lin_point+'.npy'))

Vlin = My.dot(xhy) + alin

Vtot = np.concatenate((V0,Vlin))

data0 = data['line1'] # Choose a line here
# EU LV Version
buses = []
for node in YNodeOrder:
    buses = buses+[node.split('.')[0]]

bus1 = data0[0]
bus2 = data0[1]
Yprim = data0[2]

idx1 = [i for i, x in enumerate(buses) if x == bus1]
idx2 = [i for i, x in enumerate(buses) if x == bus2]

Vidx = Vtot[idx1+idx2]
Iphs = Yprim.dot(Vidx)
Sinj = Vidx*(Iphs.conj())
Sloss = sum(Sinj)

# # FULL VERSION
# buses = []
# phses = []
# for node in YNodeOrder:
    # buses = buses+[node.split('.')[0]]
    # phses = phses+[node.split('.')[1]]

# phses = np.array(phses)
# bus1 = data0[0].split('.')[0].upper()
# bus2 = data0[1].split('.')[0].upper()
# phs1 = data0[0].split('.')[1:] # nb: str type
# phs2 = data0[1].split('.')[1:]
# Yprim = data0[2]

# idxNom1 = [i for i, x in enumerate(buses) if x == bus1]
# idxNom2 = [i for i, x in enumerate(buses) if x == bus2]
# phses1 = phses[idxNom1] # NP str type
# phses2 = phses[idxNom2]

# # check the phase order
# if phs1 == [] and phs2 == []:
    # idx1 = idxNom1
    # idx2 = idxNom2
# else: # for some circuits phases are given explicitly
    # idx1 = []
    # idx2 = []
    # for phs in phses1:
        # idx1 = idx1 + [idxNom1[phs1.index(phs)]]
    # for phs in phses2:
        # idx2 = idx2 + [idxNom2[phs2.index(phs)]]

# Vidx = Vtot[idx1+idx2]
# I = Yprim.dot(Vidx)


