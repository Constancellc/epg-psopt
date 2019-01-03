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

# based on monte_carlo.py
print('Start.\n',time.process_time())

FD = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc181217\\"
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
# WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"


DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolutiobn = DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
fdr_i = 11
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod']
feeder=fdrs[fdr_i]

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]
lin_point=1.0

def vecSlc(vec_like,new_idx):
    if type(vec_like)==tuple:
        vec_slc = tuple(np.array(vec_like)[new_idx].tolist())
    return vec_slc

fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

DSSText.command='Compile ('+fn+'.dss)'
zoneNames, yzRegIdx = get_yzRegIdx(DSSCircuit)

YZ = DSSCircuit.YNodeOrder
YZnew = vecSlc(YZ,yzRegIdx)

Ky,Kd,Kt,bV,xhy0,xhd0 = loadLinMagModel(feeder,lin_point,WD)

v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
v_idx = np.array(get_element_idxs(DSSCircuit,v_types)) - 3
v_idx = v_idx[v_idx>=0]

sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
p_idx = np.array(sY[3:].nonzero())

# s_idx = np.concatenate((p_idx,p_idx+len(sY)-3),axis=1)[0]


# example from online.
x0 = [2,-1,2,-2,1,4,3]
I0 = [1,2,0,2,3,2,0]
J0 = [0,0,1,1,2,3,4]
A = sparse.coo_matrix(x0,(I0,J0))
A0 = spmatrix(x0,I0,J0)
print(A0.I) # these both work fine.
print(A0.J) # these both work fine.
print(A0) 

# new example.
n = 4
x = np.ones(n).tolist()
I = list(range(n))
J = random.permutation(range(n)).tolist()
B0 = spmatrix(x,I,J)
print(B0.I)
print(B0.J)

# # final example
# n = len(YZ)
# x = np.ones(n).tolist()
# I = list(range(n))
# J = random.permutation(range(n)).tolist()
# B = sparse.coo_matrix((x,(I,J)))
# print(B)
# B0 = spmatrix(x,I,J)
# print(B0.I)

# x = np.ones(len(YZ)).tolist()
# I = list(range(len(YZ)))
# J = random.permutation(range(len(YZ))).tolist()
# B = sparse.coo_matrix((x,(I,J)))
# print(B)
# B0 = spmatrix(x,I,J)
# print(B0.J)
