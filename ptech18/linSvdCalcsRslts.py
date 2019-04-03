import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib import cm
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import linModel, calcVar, hcPdfs, plotCns, plotHcVltn
from scipy.stats.stats import pearsonr


WD = os.path.dirname(sys.argv[0])

fn0 = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190325\\"

nMc = 50
prmI = 0

pdfName = 'gammaWght'; prms=np.array([0.5]); prms=np.array([3.0])
# pdfName = 'gammaFlat'; prms=np.array([0.5]); prms=np.array([3.0])
pdfName = 'gammaFrac'; prms=np.arange(0.05,1.05,0.05)
# pdfName = 'gammaFrac'; prms=np.arange(0.025,0.625,0.025)
# pdfName = 'gammaFrac'; prms=np.array([0.25,0.25])
# pdfName = 'gammaXoff'; prms=(np.concatenate((0.33*np.ones((1,19)),np.array([30*np.arange(0.05,1.0,0.05)])),axis=0)).T

fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

# ============================== 1. plotting EU LV and EPRI K1 for CC 
fdr_i_set = [0,20]
for fdr_i in fdr_i_set:
    LM = linModel(fdr_i,WD,QgenPf=1.0)
    LM.loadNetModel()
    ax = LM.plotNetwork(pltShow=False)

    xlm = ax.get_xlim() 
    ylm = ax.get_ylim()
    dx = xlm[1] - xlm[0]; dy = ylm[1] - ylm[0] # these seem to be in feet for k1
    if fdr_i==0:
        # (2637175.474787638, 2653020.026654688) (2637175.474787638, 2653020.026654688)
        dist = 10
        x0 = xlm[0] + 0.8*dx
        y0 = ylm[0] + 0.05*dy
        ax.plot([x0,x0+dist],[y0,y0],'k-')
        ax.plot([x0,x0],[y0-0.005*dy,y0+0.005*dy],'k-')
        ax.plot([x0+dist,x0+dist],[y0-0.005*dy,y0+0.005*dy],'k-')
        ax.annotate('10 metres',(x0+(dist/2),y0+dy*0.02),ha='center')
        plt.savefig(WD+'\\hcResults\\eulvNetwork.pdf')
    if fdr_i==20:
        # (390860.71323843475, 391030.5357615654) (390860.71323843475, 391030.5357615654)
        dist = 5280
        x0 = xlm[0] + 0.6*dx
        y0 = ylm[0] + 0.05*dy
        ax.plot([x0,x0+dist],[y0,y0],'k-')
        ax.plot([x0,x0],[y0-0.005*dy,y0+0.005*dy],'k-')
        ax.plot([x0+dist,x0+dist],[y0-0.005*dy,y0+0.005*dy],'k-')
        ax.annotate('1 mile',(x0+(dist/2),y0+dy*0.02),ha='center')
        plt.savefig(WD+'\\hcResults\\epriK1Network.pdf')
    plt.show()

# ============================== 2. MAP of Nstd before

