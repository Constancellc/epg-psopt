# similar to v1 except this only looks at the full scale networks.
import win32com.client
import os, sys
import numpy as np
import matplotlib.pyplot as plt

DSSObj = win32com.client.Dispatch('OpenDSSEngine.dss')
DSSText = DSSObj.Text
DSSC = DSSObj.ActiveCircuit

WD = os.path.dirname(sys.argv[0])

TP = []; TL = []; MxV = []; MnV = []; NL = []; Ntwk = []; Fdr = []; Nnodes = []

for i in range(1,26):
    fn = os.path.join(WD,'batch_manc_ntwx','network_'+str(i),'masterNetwork'+str(i)+'.dss')
    print('Running ' + fn)
    DSSText.Command="Compile (" + fn + ")"
    TP = TP + [DSSC.TotalPower[0] + 1j*DSSC.TotalPower[1]]
    TL = TL + [DSSC.Losses[0] + 1j*DSSC.Losses[1]]
    MxV = MxV + [max(DSSC.AllBusVmagPu)]
    MnV = MnV + [min(DSSC.AllBusVmagPu)]
    NL = NL + [DSSC.Loads.Count]
    Nnodes = Nnodes + [DSSC.NumNodes]
    
TP = np.array(TP); TL = np.array(TL); MxV = np.array(MxV); MnV = np.array(MnV); 
NL = np.array(NL); Ntwk = np.array(Ntwk); Fdr = np.array(Fdr); Nnodes = np.array(Nnodes)

m2perHousehold = [350,406,432,551,358,292,223,508,341,664,292,424,300,394,
                      440,431,317,427,540,104,398,924,344,297,429] #as counted by cc

# Note than Min/Max V are impacted by some dodgy feeders, as described in the 
# notes by Nando on the set of networks. (basically: 'users should rephase...')
fig,axes = plt.subplots(2,3,figsize=(9,10))
data = [MxV,MnV,NL,1e2*1e-3*TL.real/NL,Nnodes,m2perHousehold]
# titles = ['Max V','Min V','No. Loads','No. Nodes']
titles = ['Max V','Min V','No. Loads','TL per load, %','Nnodes','m2 per household']
i=0
for ax in axes.ravel():
    ax.bar(np.arange(1,26),data[i])
    ax.set_xlabel('Network no.')
    ax.set_title(titles[i])
    ax.grid(True)
    i+=1

plt.tight_layout()
plt.show()