import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

nSet = [1,2,4,10,20,40,100]
epsSet = [0.01,0.025,0.05,0.075,0.10]
betaSet = [0.01,0.02,0.05,0.1]

N = np.logspace(0,4,200,dtype=int)
fig,axs = plt.subplots(len(betaSet))
ii=0
for ax in axs:
    betaAct = betaSet[ii]
    for eps in epsSet:
        betaAll = []
        for n in nSet:
            BetaN = []
            for Nchosen in N:
                betaN = 0.0
                for i in range(n):
                    addtion = comb(Nchosen,i,exact=False)*(eps**i)*( (1-eps)**(Nchosen-i) )
                    betaN = betaN+addtion
                BetaN.append(betaN)
            betaAll.append(BetaN)
        N2008 = N[np.argmin(np.array(betaAll)>betaAct,axis=1)]
        ax.plot(nSet,N2008,'x',label='Eps: '+str(eps))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlabel('No. decision variables')
    ax.set_ylabel('No. req. scenarios')
    ax.legend(title='Beta: '+str(betaAct))
    ii+=1
plt.show()