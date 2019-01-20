import win32com.client
import os
import numpy as np
import matplotlib.pyplot as plt
import getpass

# Cycle through each network
# - solve; get:
#   = total power in network (# loads)#
#   = max voltage pu
#   = min voltage pu
#   = losses

DSSObj = win32com.client.Dispatch('OpenDSSEngine.dss')
DSSText = DSSObj.Text
DSSC = DSSObj.ActiveCircuit

if getpass.getuser()=='Matt':
    WD=r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\manchester_models\batch_manc_ntwx"

TP = []; TL = []; MxV = []; MnV = []; NL = []; Ntwk = []; Fdr = []

os.chdir(WD)
Dir0 = os.listdir()
for dir in Dir0:
    if dir[0:8]=='network_':
        os.chdir(dir)
        Dir1 = os.listdir()
        for fdr in Dir1:
            if fdr[0:7]=='Feeder_':
                os.chdir(fdr)
                print('Running ' + dir + ', ' + fdr)
                DSSText.command="Compile (" + os.getcwd() + "\\Master.dss)"
                TP = TP + [DSSC.TotalPower[0] + 1j*DSSC.TotalPower[1]]
                TL = TL + [DSSC.Losses[0] + 1j*DSSC.Losses[1]]
                MxV = MxV + [max(DSSC.AllBusVmagPu)]
                MnV = MnV + [min(DSSC.AllBusVmagPu)]
                NL = NL + [DSSC.Loads.Count]
                Ntwk = Ntwk + [dir]
                Fdr = Fdr + [fdr]
                os.chdir('..')
        os.chdir('..')

TP = np.array(TP); TL = np.array(TL); MxV = np.array(MxV); MnV = np.array(MnV); NL = np.array(NL); Ntwk = np.array(Ntwk); Fdr = np.array(Fdr)

for ntwk in np.unique(Ntwk):
    plt.plot(NL[Ntwk==ntwk])
plt.show()

for ntwk in np.unique(Ntwk):
    plt.plot(MnV[Ntwk==ntwk])
    plt.plot(MxV[Ntwk==ntwk],'--')

plt.show()
    
# plt.plot(NL[Ntwk=='network_5']); plt.show()






