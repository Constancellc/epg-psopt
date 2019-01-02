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
fdr_i = 10
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','13bus']
ckts = {'feeder_name':['fn_ckt','fn']}
ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
ckts[fdrs[1]]=feeder_to_fn(WD,fdrs[1])
ckts[fdrs[2]]=feeder_to_fn(WD,fdrs[2])
ckts[fdrs[3]]=feeder_to_fn(WD,fdrs[3])
ckts[fdrs[4]]=feeder_to_fn(WD,fdrs[4])
ckts[fdrs[5]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_z']
ckts[fdrs[6]]=[WD+'\\ieee_tn\\34Bus_copy',WD+'\\ieee_tn\\34Bus_copy\\ieee34Mod1_z_mod']
ckts[fdrs[7]]=[WD+'\\ieee_tn\\37Bus_copy',WD+'\\ieee_tn\\37Bus_copy\\ieee37_z']
ckts[fdrs[8]]=[WD+'\\ieee_tn\\123Bus_copy',WD+'\\ieee_tn\\123Bus_copy\\IEEE123Master_z']
ckts[fdrs[9]]=[WD+'\\ieee_tn\\8500-Node_copy',WD+'\\ieee_tn\\8500-Node_copy\\Master-unbal_z']
ckts[fdrs[10]]=[WD+'\\ieee_tn\\13Bus_copy',WD+'\\ieee_tn\\13Bus_copy\\IEEE13Nodeckt_mod_z']




fn_ckt = ckts[fdrs[fdr_i]][0]
fn = ckts[fdrs[fdr_i]][1]
feeder=fdrs[fdr_i]

fn_y = fn+'_y'
sn0 = WD + '\\lin_models\\' + feeder

DSSText.command='Compile ('+fn+'.dss)'
DSSEM = DSSCircuit.Meters

i = DSSEM.First
while i:
    # print(DSSEM.AllEndElements)
    print(DSSEM.AllBranchesInZone)
    i=DSSEM.Next




