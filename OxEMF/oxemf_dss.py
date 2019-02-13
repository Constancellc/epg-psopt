# The main file of interest is Network_3ph_pf.py - this is used to create a Network_3ph object. The Network_3ph_pf.setup_network_ieee13() sets up a network object as the ieee13 bus. As discussed, the main idea would be to have an alternative function which could replace this method, to load in bus & line data from an external file.

# Two main panda dataframes need to be set:
# bus_df, with columns ['name','number','load_type','connect','Pa','Pb','Pc','Qa','Qb','Qc’], and 
# line_df with columns ['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc’]

# The file zbus_3ph_pf_test.py can be used for testing the Network_3ph class.
import getpass
import sys
if getpass.getuser()=='Matt':
    WD0 = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\OxEMF\OxEMF_3ph_PF"
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
if getpass.getuser()=='chri3793':
    WD0 = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\OxEMF"
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

sys.path.insert(0, WD0 + r"\OxEMF_3ph_PF")
sys.path.insert(0, WD)

import os
import pandas as pd
import win32com.client
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Network_3ph_pf import Network_3ph
from dss_python_funcs import get_ckt, tp2mat, tp_2_ar

fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1']; lp_taps='Nmt'
feeder='021'
feeder = fdrs[fdr_i]

fn_ckt = get_ckt(WD,feeder)[1] + '_oxemf'

saveModel = True
# saveModel = False

dir0 = WD0 + '\\ntwx\\' + feeder
sn0 = dir0 + '\\' + feeder + lp_taps


DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

DSSText.command='Compile ('+fn_ckt+'.dss)'
DSSText.command='Set Controlmode=off' # turn all regs/caps off

LDS = DSSCircuit.Loads
CAP = DSSCircuit.Capacitors
LNS = DSSCircuit.Lines
TRN = DSSCircuit.Transformers
ACE = DSSCircuit.ActiveElement
SetACE = DSSCircuit.SetActiveElement

nLds = LDS.Count
nCap = CAP.Count
nTrn = TRN.Count
nLns = LNS.Count

# line_df = pd.DataFrame(data=None, index=np.arange(nTrn+nLns), columns=['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc'])
line_df = pd.DataFrame(data=np.zeros((nLns,14),dtype=complex), index=LNS.AllNames, columns=['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc'])

# Create Line DF from lines and transformers ====
# line_df.loc['632645']

i = LNS.First
while i:
    lineName = LNS.name
    line_df.loc[lineName,'busA']=LNS.Bus1.split('.')[0]
    line_df.loc[lineName,'busB']=LNS.Bus2.split('.')[0]
    lineLen = LNS.Length
    zmat0 = (tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix))*lineLen # ohms
    bmat0 = (1j*2*np.pi*60*tp2mat(LNS.Cmatrix)*1e-9)*lineLen # ohms
    SetACE('Line.'+LNS.Name)
    
    nPh = ACE.NumPhases
    phs = list(map(int,ACE.BusNames[0].split('.')[1:]))
    
    Zmat = np.zeros((3,3),dtype=complex)
    Bmat = np.zeros((3,3),dtype=complex)
    if nPh==1:
        Zmat[phs[0]-1,phs[0]-1] = zmat0[0,0]
        Bmat[phs[0]-1,phs[0]-1] = bmat0[0,0]
    if nPh==2:
        Zmat[phs[0]-1,phs[0]-1] = zmat0[0,0]
        Zmat[phs[1]-1,phs[0]-1] = zmat0[0,1]
        Zmat[phs[0]-1,phs[1]-1] = zmat0[0,1]
        Zmat[phs[1]-1,phs[1]-1] = zmat0[1,1]
        
        Bmat[phs[0]-1,phs[0]-1] = bmat0[0,0]
        Bmat[phs[1]-1,phs[0]-1] = bmat0[0,1]
        Bmat[phs[0]-1,phs[1]-1] = bmat0[0,1]
        Bmat[phs[1]-1,phs[1]-1] = bmat0[1,1]
    if nPh==3:
        Zmat = zmat0
        Bmat = bmat0
        
    line_df.loc[lineName,'Zaa']=Zmat[0,0]
    line_df.loc[lineName,'Zbb']=Zmat[1,1]
    line_df.loc[lineName,'Zcc']=Zmat[2,2]
    line_df.loc[lineName,'Zab']=Zmat[0,1]
    line_df.loc[lineName,'Zac']=Zmat[0,2]
    line_df.loc[lineName,'Zbc']=Zmat[1,2]
    
    line_df.loc[lineName,'Baa']=Bmat[0,0]
    line_df.loc[lineName,'Bab']=Bmat[0,1]
    line_df.loc[lineName,'Bbb']=Bmat[1,1]
    line_df.loc[lineName,'Bcc']=Bmat[2,2]
    line_df.loc[lineName,'Bac']=Bmat[0,2]
    line_df.loc[lineName,'Bbc']=Bmat[1,2]
    
    # line_df.loc[lineName,'Zaa']=Zmat[0,0]
    # line_df.loc[lineName,'Baa']=Bmat[0,0]
    # if nPh>1:
        # line_df.loc[lineName,'Zbb']=Zmat[1,1]
        # line_df.loc[lineName,'Zab']=Zmat[0,1]

        # line_df.loc[lineName,'Bab']=Bmat[0,1]
        # line_df.loc[lineName,'Bbb']=Bmat[1,1]
    # if nPh==3:
        # line_df.loc[lineName,'Zcc']=Zmat[2,2]
        # line_df.loc[lineName,'Zac']=Zmat[0,2]
        # line_df.loc[lineName,'Zbc']=Zmat[1,2]
    
        # line_df.loc[lineName,'Bcc']=Bmat[2,2]
        # line_df.loc[lineName,'Bac']=Bmat[0,2]
        # line_df.loc[lineName,'Bbc']=Bmat[1,2]
    i = LNS.Next
    
# i = TRN.First + nLns # only Lines implemented at this stage.
# while i-nLns:
    # SetACE('Transformer.'+TRN.Name)
    # line_df.loc[i-1]['busA']=ACE.BusNames[0]
    # line_df.loc[i-1]['busB']=ACE.BusNames[1]
    
    # YPrimLin = tp_2_ar(ACE.YPrim)
    # # YPrimLin = np.reshape(YPrimLin,(8,8))
    
    # # Zmat = tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix)
    # # Bmat = 2*np.pi*60*tp2mat(LNS.Cmatrix)
    
    # # nPh = ACE.NumPhases
    # # line_df.loc[i-1]['Zaa']=Zmat[0,0]
    # # line_df.loc[i-1]['Baa']=Bmat[0,0]
    # # if nPh>1:
        # # line_df.loc[i-1]['Zbb']=Zmat[1,1]
        # # line_df.loc[i-1]['Zab']=Zmat[0,1]

        # # line_df.loc[i-1]['Bab']=Bmat[0,1]
        # # line_df.loc[i-1]['Bbb']=Bmat[1,1]
    # # if nPh==3:
        # # line_df.loc[i-1]['Zcc']=Zmat[2,2]
        # # line_df.loc[i-1]['Zac']=Zmat[0,2]
        # # line_df.loc[i-1]['Zbc']=Zmat[1,2]
    
        # # line_df.loc[i-1]['Bcc']=Bmat[2,2]
        # # line_df.loc[i-1]['Bac']=Bmat[0,2]
        # # line_df.loc[i-1]['Bbc']=Bmat[1,2]
    # i = TRN.Next + nLns


# create bus_df from loads and capacitors ======
nBus = DSSCircuit.NumBuses
bus_df = pd.DataFrame(data=np.zeros((nBus,10)), index=DSSCircuit.AllBusNames, columns=["name","number","load_type","connect","Pa","Pb","Pc","Qa","Qb","Qc"])

bus_df['name'] = DSSCircuit.AllBusNames
bus_df['number'] = np.arange((nBus))

# Find the slack bus:
VSRC = DSSCircuit.Vsources
VSRC.First
SetACE('Vsource.'+VSRC.Name)
bus_df.loc[:,'connect'] = 'Y'
bus_df.loc[ACE.BusNames[0],'load_type'] = 'S'


i = LDS.First

while i:
    SetACE('Load.'+LDS.Name)
    actBus = ACE.BusNames[0].split('.')[0]
    
    if LDS.Model==1:
        load_type = 'PQ'
    elif LDS.Model==2:
        load_type = 'Z'
    elif LDS.Model==5:
        load_type = 'I'
    else:
        print('Warning! Load: ',LDS.Name,'Load model not a ZIP load. Setting as PQ.')
        load_type = 'PQ'
    
    if bus_df.loc[actBus,'load_type']==0 or bus_df.loc[actBus,'load_type']==load_type:
        bus_df.loc[actBus,'load_type'] = load_type
    else:
        bus_df.loc[actBus,'load_type'] = 'Mxd'

    nPh = ACE.NumPhases
    phs = ACE.BusNames[0].split('.')[1:]
    if LDS.IsDelta:
        bus_df.loc[actBus,'connect'] = 'D'
        if nPh==1:
            if '1' in phs and '2' in phs:
                bus_df.loc[actBus,'Pa'] = LDS.kW + bus_df.loc[actBus,'Pa']
                bus_df.loc[actBus,'Qa'] = LDS.kVar + bus_df.loc[actBus,'Qa']
            if '2' in phs and '3' in phs:
                bus_df.loc[actBus,'Pb'] = LDS.kW + bus_df.loc[actBus,'Pb']
                bus_df.loc[actBus,'Qb'] = LDS.kVar + bus_df.loc[actBus,'Qb']
            if '3' in phs and '1' in phs:
                bus_df.loc[actBus,'Pc'] = LDS.kW + bus_df.loc[actBus,'Pc']
                bus_df.loc[actBus,'Qc'] = LDS.kVar + bus_df.loc[actBus,'Qc']
        if nPh==3:
            bus_df.loc[actBus,'Pa'] = LDS.kW/3 + bus_df.loc[actBus,'Pa']
            bus_df.loc[actBus,'Pb'] = LDS.kW/3 + bus_df.loc[actBus,'Pb']
            bus_df.loc[actBus,'Pc'] = LDS.kW/3 + bus_df.loc[actBus,'Pc']
            bus_df.loc[actBus,'Qa'] = LDS.kVar/3 + bus_df.loc[actBus,'Qa']
            bus_df.loc[actBus,'Qb'] = LDS.kVar/3 + bus_df.loc[actBus,'Qb']
            bus_df.loc[actBus,'Qc'] = LDS.kVar/3 + bus_df.loc[actBus,'Qc']
        if nPh==2:
            print('Warning! Load: ',LDS.Name,'2 phase Delta loads not yet implemented.')
    else:
        bus_df.loc[actBus,'connect'] = 'Y'
        if '1' in phs or phs==[]:
            bus_df.loc[actBus,'Pa'] = LDS.kW/nPh + bus_df.loc[actBus,'Pa']
            bus_df.loc[actBus,'Qa'] = LDS.kVar/nPh + bus_df.loc[actBus,'Qa']
        if '2' in phs or phs==[]:
            bus_df.loc[actBus,'Pb'] = LDS.kW/nPh + bus_df.loc[actBus,'Pb']
            bus_df.loc[actBus,'Qb'] = LDS.kVar/nPh + bus_df.loc[actBus,'Qb']
        if '3' in phs or phs==[]:
            bus_df.loc[actBus,'Pc'] = LDS.kW/nPh + bus_df.loc[actBus,'Pc']
            bus_df.loc[actBus,'Qc'] = LDS.kVar/nPh + bus_df.loc[actBus,'Qc']
    
    i = LDS.Next

i = CAP.First + nLds
while i-nLds:
    SetACE('Capacitor.'+CAP.Name)
    actBus = ACE.BusNames[0].split('.')[0]
    bus_df.loc[actBus,'number'] = i-1
    
    if bus_df.loc[actBus,'load_type']==0 or bus_df.loc[actBus,'load_type']=='Z':
        bus_df.loc[actBus,'load_type'] = 'Z'
    else:
        bus_df.loc[actBus,'load_type'] = 'Mxd'
    
    nPh = ACE.NumPhases
    phs = ACE.BusNames[0].split('.')[1:]
    if CAP.IsDelta:
        bus_df.loc[actBus,'connect'] = 'D'
        if nPh==1:
            if '1' in phs and '2' in phs:
                bus_df.loc[actBus,'Qa'] = -CAP.kVar + bus_df.loc[actBus,'Qa']
            if '2' in phs and '3' in phs:
                bus_df.loc[actBus,'Qb'] = -CAP.kVar + bus_df.loc[actBus,'Qb']
            if '3' in phs and '1' in phs:
                bus_df.loc[actBus,'Qc'] = -CAP.kVar + bus_df.loc[actBus,'Qc']
        if nPh==3:
            bus_df.loc[actBus,'Qa'] = -CAP.kVar/3 + bus_df.loc[actBus,'Qa']
            bus_df.loc[actBus,'Qb'] = -CAP.kVar/3 + bus_df.loc[actBus,'Qb']
            bus_df.loc[actBus,'Qc'] = -CAP.kVar/3 + bus_df.loc[actBus,'Qc']
        if nPh==2:
            print('Warning! Cap: ',CAP.Name,'2 phase Delta loads not yet implemented.')
    else:
        bus_df.loc[actBus,'connect'] = 'Y'
        if '1' in phs or phs==[]:
            bus_df.loc[actBus,'Qa'] = -CAP.kVar/nPh + bus_df.loc[actBus,'Qa']
        if '2' in phs or phs==[]:
            bus_df.loc[actBus,'Qb'] = -CAP.kVar/nPh + bus_df.loc[actBus,'Qb']
        if '3' in phs or phs==[]:
            bus_df.loc[actBus,'Qc'] = -CAP.kVar/nPh + bus_df.loc[actBus,'Qc']
    i = CAP.Next + nLds
    
    
if saveModel:
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    bus_df.to_csv(sn0+"_bus_df.csv")
    line_df.to_csv(sn0+"_line_df.csv")