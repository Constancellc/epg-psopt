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
    # WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\OxEMF\OxEMF_3ph_PF" # for files
sys.path.insert(0, WD0)
sys.path.insert(0, WD)

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

fn_ckt = get_ckt(WD,feeder)[1]

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

bus_df = pd.DataFrame(data=None, index=np.arange(nLds+nCap), columns=["name","number","load_type","connect","Pa","Pb","Pc","Qa","Qb","Qc"])
line_df = pd.DataFrame(data=None, index=np.arange(nTrn+nLns), columns=['busA','busB','Zaa','Zbb','Zcc','Zab','Zac','Zbc','Baa','Bbb','Bcc','Bab','Bac','Bbc'])

# Create Line DF from lines and transformers ====
i = LNS.First
while i:
    line_df.loc[i-1]['busA']=LNS.Bus1
    line_df.loc[i-1]['busB']=LNS.Bus2
    Zmat = tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix)
    Bmat = 2*np.pi*60*tp2mat(LNS.Cmatrix)
    SetACE('Line.'+LNS.Name)
    nPh = ACE.NumPhases
    line_df.loc[i-1]['Zaa']=Zmat[0,0]
    line_df.loc[i-1]['Baa']=Bmat[0,0]
    if nPh>1:
        line_df.loc[i-1]['Zbb']=Zmat[1,1]
        line_df.loc[i-1]['Zab']=Zmat[0,1]

        line_df.loc[i-1]['Bab']=Bmat[0,1]
        line_df.loc[i-1]['Bbb']=Bmat[1,1]
    if nPh==3:
        line_df.loc[i-1]['Zcc']=Zmat[2,2]
        line_df.loc[i-1]['Zac']=Zmat[0,2]
        line_df.loc[i-1]['Zbc']=Zmat[1,2]
    
        line_df.loc[i-1]['Bcc']=Bmat[2,2]
        line_df.loc[i-1]['Bac']=Bmat[0,2]
        line_df.loc[i-1]['Bbc']=Bmat[1,2]
    i = LNS.Next
    
i = TRN.First + nLns
while i-nLns:
    SetACE('Transformer.'+TRN.Name)
    line_df.loc[i-1]['busA']=ACE.BusNames[0]
    line_df.loc[i-1]['busB']=ACE.BusNames[1]
    
    YPrimLin = tp_2_ar(ACE.YPrim)
    # YPrimLin = np.reshape(YPrimLin,(8,8))
    
    # Zmat = tp2mat(LNS.Rmatrix) + 1j*tp2mat(LNS.Xmatrix)
    # Bmat = 2*np.pi*60*tp2mat(LNS.Cmatrix)
    
    # nPh = ACE.NumPhases
    # line_df.loc[i-1]['Zaa']=Zmat[0,0]
    # line_df.loc[i-1]['Baa']=Bmat[0,0]
    # if nPh>1:
        # line_df.loc[i-1]['Zbb']=Zmat[1,1]
        # line_df.loc[i-1]['Zab']=Zmat[0,1]

        # line_df.loc[i-1]['Bab']=Bmat[0,1]
        # line_df.loc[i-1]['Bbb']=Bmat[1,1]
    # if nPh==3:
        # line_df.loc[i-1]['Zcc']=Zmat[2,2]
        # line_df.loc[i-1]['Zac']=Zmat[0,2]
        # line_df.loc[i-1]['Zbc']=Zmat[1,2]
    
        # line_df.loc[i-1]['Bcc']=Bmat[2,2]
        # line_df.loc[i-1]['Bac']=Bmat[0,2]
        # line_df.loc[i-1]['Bbc']=Bmat[1,2]
    i = TRN.Next + nLns









# create bus_df from loads and capacitors ======
i = LDS.First
while i:
    bus_df.loc[i-1]['name'] = LDS.Name
    bus_df.loc[i-1]['number'] = i-1
    if LDS.Model==1:
        bus_df.loc[i-1]['load_type'] = 'PQ'
    elif LDS.Model==2:
        bus_df.loc[i-1]['load_type'] = 'Z'
    elif LDS.Model==5:
        bus_df.loc[i-1]['load_type'] = 'I'
    else:
        print('Warning! Load: ',LDS.Name,'Load model not a ZIP load. Setting as PQ.')
        
        bus_df.loc[i-1]['load_type'] = 'PQ'

    SetACE('Loads.'+LDS.Name)
    nPh = ACE.NumPhases
    phs = ACE.BusNames[0].split('.')[1:]
    if LDS.IsDelta:
        bus_df.loc[i-1]['connect'] = 'D'
        if nPh==1:
            if '1' in phs and '2' in phs:
                bus_df.loc[i-1]['Pa'] = LDS.kW
                bus_df.loc[i-1]['Qa'] = LDS.kVar
            if '2' in phs and '3' in phs:
                bus_df.loc[i-1]['Pb'] = LDS.kW
                bus_df.loc[i-1]['Qb'] = LDS.kVar
            if '3' in phs and '1' in phs:
                bus_df.loc[i-1]['Pc'] = LDS.kW
                bus_df.loc[i-1]['Qc'] = LDS.kVar
        if nPh==3:
            bus_df.loc[i-1]['Pa'] = LDS.kW/3
            bus_df.loc[i-1]['Pb'] = LDS.kW/3
            bus_df.loc[i-1]['Pc'] = LDS.kW/3
            bus_df.loc[i-1]['Qa'] = LDS.kVar/3
            bus_df.loc[i-1]['Qb'] = LDS.kVar/3
            bus_df.loc[i-1]['Qc'] = LDS.kVar/3
        if nPh==2:
            print('Warning! Load: ',LDS.Name,'2 phase Delta loads not yet implemented.')
    else:
        bus_df.loc[i-1]['connect'] = 'Y'
        if '1' in phs or phs==[]:
            bus_df.loc[i-1]['Pa'] = LDS.kW/nPh
            bus_df.loc[i-1]['Qa'] = LDS.kVar/nPh
        if '2' in phs or phs==[]:
            bus_df.loc[i-1]['Pb'] = LDS.kW/nPh
            bus_df.loc[i-1]['Qb'] = LDS.kVar/nPh
        if '3' in phs or phs==[]:
            bus_df.loc[i-1]['Pc'] = LDS.kW/nPh
            bus_df.loc[i-1]['Qc'] = LDS.kVar/nPh
    
    i = LDS.Next

i = CAP.First + nLds
while i-nLds:
    bus_df.loc[i-1]['name'] = CAP.Name
    bus_df.loc[i-1]['number'] = i-1
    
    bus_df.loc[i-1]['load_type'] = 'Z'
    SetACE('Capacitor.'+CAP.Name)
    nPh = ACE.NumPhases
    phs = ACE.BusNames[0].split('.')[1:]
    if CAP.IsDelta:
        bus_df.loc[i-1]['connect'] = 'D'
        if nPh==1:
            if '1' in phs and '2' in phs:
                bus_df.loc[i-1]['Qa'] = CAP.kVar
            if '2' in phs and '3' in phs:
                bus_df.loc[i-1]['Qb'] = CAP.kVar
            if '3' in phs and '1' in phs:
                bus_df.loc[i-1]['Qc'] = CAP.kVar
        if nPh==3:
            bus_df.loc[i-1]['Qa'] = CAP.kVar/3
            bus_df.loc[i-1]['Qb'] = CAP.kVar/3
            bus_df.loc[i-1]['Qc'] = CAP.kVar/3
        if nPh==2:
            print('Warning! Cap: ',CAP.Name,'2 phase Delta loads not yet implemented.')
    else:
        bus_df.loc[i-1]['connect'] = 'Y'
        if '1' in phs or phs==[]:
            bus_df.loc[i-1]['Qa'] = CAP.kVar/nPh
        if '2' in phs or phs==[]:
            bus_df.loc[i-1]['Qb'] = CAP.kVar/nPh
        if '3' in phs or phs==[]:
            bus_df.loc[i-1]['Qc'] = CAP.kVar/nPh
    i = CAP.Next + nLds