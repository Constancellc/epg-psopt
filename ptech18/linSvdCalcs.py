import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
from dss_voltage_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from matplotlib import cm, patches
from sklearn.decomposition import TruncatedSVD
from matplotlib.collections import LineCollection
from random import sample, seed, shuffle

from cvxopt import matrix, solvers, msk
from mosek import iparam

def cnsBdsCalc(vLsMv,vLsLv,vHsMv,vHsLv,vDv,cns,DVmax=0.06):
    nMc = vLsMv.shape[0]
    
    vLsMv[vLsMv<0.5] = 1.0
    vLsLv[vLsLv<0.5] = 1.0
    vHsMv[vHsMv<0.5] = 1.0
    vHsLv[vHsLv<0.5] = 1.0
    
    if vLsMv.shape[1]==0:
        vLsMv = np.ones((nMc,1))
    if vLsLv.shape[1]==0:
        vLsLv = np.ones((nMc,1))
    if vHsMv.shape[1]==0:
        vHsMv = np.ones((nMc,1))
    if vHsLv.shape[1]==0:
        vHsLv = np.ones((nMc,1))
    if vDv.shape[1]==0:
        vDv = np.zeros((nMc,1))
    
    VpMv = cns.VpMv
    VmMv = cns.VmMv
    VpLv = cns.VpLv
    VmLv = cns.VmLv
    
    vMaxLsMv = np.max(vLsMv,axis=1)
    vMinLsMv = np.min(vLsMv,axis=1)
    vMaxHsMv = np.max(vHsMv,axis=1)
    vMinHsMv = np.min(vHsMv,axis=1)
    maxDv = np.max(vDv,axis=1)
    if vLsLv.shape[1]!=0: # some networks do not have low voltage sections e.g. 34 bus
        vMaxLsLv = np.max(vLsLv,axis=1)
        vMinLsLv = np.min(vLsLv,axis=1)
        vMaxHsLv = np.max(vHsLv,axis=1)
        vMinHsLv = np.min(vHsLv,axis=1)
    else:
        vMaxLsLv = np.ones(vMaxLsMv.shape)
        vMinLsLv = np.ones(vMaxLsMv.shape)
        vMaxHsLv = np.ones(vMaxLsMv.shape)
        vMinHsLv = np.ones(vMaxLsMv.shape)
    
    cnsPct = 100*np.array([sum(maxDv>DVmax),sum(vMaxLsMv>VpMv),sum(vMinLsMv<VmMv),sum(vMaxLsLv>VpLv),sum(vMinLsLv<VmLv),sum(vMaxHsMv>VpMv),sum(vMinHsMv<VmMv),sum(vMaxHsLv>VpLv),sum(vMinHsLv<VmLv)])/nMc
    inBounds = np.any(np.array([maxDv>DVmax,vMaxLsMv>VpMv,vMinLsMv<VmMv,vMaxLsLv>VpLv,vMinLsLv<VmLv,vMaxHsMv>VpMv,vMinHsMv<VmMv,vMaxHsLv>VpLv,vMinHsLv<VmLv]),axis=0)
    return cnsPct, inBounds


def linPrgCalc(dVpmv,dVplv,dVqmv,dVqlv,vDvP,vDvQ,bLsMv,bHsMv,bLsLv,bHsLv,KtMv,KtLv,cns,DVmax=0.06,tmax=0.1,qmax=0.2):
    # # Linear program based on https://cvxopt.org/examples/tutorial/lp.html
    # need the form A s(hat) d + b <V+, for example, to maximise d.
    # NB note that the inputs are from Ks, Kt rather than Ktot.
    # From Linearise regs: tap limits are in pu; qlims are as a fraction of P allowed.
    
    # c = matrix([-1.0]+[0]*KtMv.shape[1])
    
    # Amv = np.hstack((np.array([dVpmv]).T,KtMv))[bLsMv>0.5]
    # Alv = np.hstack((np.array([dVplv]).T,KtLv))[bLsLv>0.5]
    # Adv = np.hstack((np.array([vDvP]).T,np.zeros((len(vDvP),KtMv.shape[1]))))
    
    # Acns = np.zeros((len(c)-1,len(c)))
    # cnsIdx = (np.arange(len(c)-1),np.arange(len(c)-1)+1)
    # Acns[cnsIdx] = 1
    
    # bLsMv = bLsMv[bLsMv>0.5]
    # bHsMv = bHsMv[bHsMv>0.5]
    # bLsLv = bLsLv[bLsLv>0.5]
    # bHsLv = bHsLv[bHsLv>0.5]
    
    # # build the b-matrix:
    # VpMv = cns.VpMv - np.maximum(bLsMv,bHsMv)
    # VpLv = cns.VpLv - np.maximum(bLsLv,bHsLv)
    # VmMv = cns.VmMv - np.minimum(bLsMv,bHsMv)
    # VmLv = cns.VmLv - np.minimum(bLsLv,bHsLv)
    # vDvCns = DVmax*np.ones((len(vDvP)))
    
    # cnsCns = tmax*np.ones((KtMv.shape[1]))
    
    # A = matrix(np.concatenate((Amv,-Amv,Alv,-Alv,Adv,-Adv,Acns,-Acns)))
    # b = matrix(np.concatenate((VpMv,-VmMv,VpLv,-VmLv,vDvCns,vDvCns,cnsCns,cnsCns)))
    
    # A = matrix(np.concatenate((Amv,Adv,-Adv,Acns,-Acns)))
    # b = matrix(np.concatenate((VpMv,vDvCns,vDvCns,cnsCns,cnsCns)))
    
    c = matrix([-1.0]+[0]*2*(KtMv.shape[1] + 1)) # just one Q constraint
    
    Amv0 = np.hstack((np.array([dVpmv]).T,KtMv,np.zeros(KtMv.shape),np.array([dVqmv]).T,np.zeros((len(dVqmv),1))))[bLsMv>0.5]
    Amv1 = np.hstack((np.array([dVpmv]).T,np.zeros(KtMv.shape),KtMv,np.zeros((len(dVqmv),1)),np.array([dVqmv]).T))[bLsMv>0.5]
    Alv0 = np.hstack((np.array([dVplv]).T,KtLv,np.zeros(KtLv.shape),np.array([dVqlv]).T,np.zeros((len(dVqlv),1))))[bLsLv>0.5]
    Alv1 = np.hstack((np.array([dVplv]).T,np.zeros(KtLv.shape),KtLv,np.zeros((len(dVqlv),1)),np.array([dVqlv]).T))[bLsLv>0.5]
    
    Adv0 = np.hstack((np.array([vDvP]).T,np.zeros((len(vDvP),KtMv.shape[1]*2)),np.array([vDvQ]).T,np.zeros((len(vDvQ),1))))
    Adv1 = np.hstack((np.array([vDvP]).T,np.zeros((len(vDvP),KtMv.shape[1]*2)),np.zeros((len(vDvQ),1)),np.array([vDvQ]).T))
    
    Acns = np.zeros((len(c)-1,len(c)))
    cnsIdx = (np.arange(len(c)-1),np.arange(len(c)-1)+1)
    Acns[cnsIdx] = 1
    
    bLsMv = bLsMv[bLsMv>0.5]
    bHsMv = bHsMv[bHsMv>0.5]
    bLsLv = bLsLv[bLsLv>0.5]
    bHsLv = bHsLv[bHsLv>0.5]
    
    # build the b-matrix:
    VpMvLs = cns.VpMv - bLsMv
    VpLvLs = cns.VpLv - bLsLv
    VmMvLs = cns.VmMv - bLsMv
    VmLvLs = cns.VmLv - bLsLv
    VpMvHs = cns.VpMv - bHsMv
    VpLvHs = cns.VpLv - bHsLv
    VmMvHs = cns.VmMv - bHsMv
    VmLvHs = cns.VmLv - bHsLv
    vDvCns = DVmax*np.ones((len(vDvP)))
    
    cnsCns = np.concatenate((tmax*np.ones((KtMv.shape[1]*2)),np.array([qmax]*2) ))
    
    A = matrix(np.concatenate((Amv0,-Amv0,Alv0,-Alv0,Amv1,-Amv1,Alv1,-Alv1,Adv0,-Adv0,Adv1,-Adv1,Acns,-Acns)))
    b = matrix(np.concatenate((VpMvLs,-VmMvLs,VpLvLs,-VmLvLs,VpMvHs,-VmMvHs,VpLvHs,-VmLvHs,vDvCns,vDvCns,vDvCns,vDvCns,cnsCns,cnsCns)))
    
    # A = matrix(np.concatenate((Amv0,-Amv0,Alv0,-Alv0,Amv1,-Amv1,Alv1,-Alv1,Adv0,-Adv0,Adv1,-Adv1,Acns,-Acns)))
    # b = matrix(np.concatenate((VpMvLs,-VmMvLs,VpLvLs,-VmLvLs,VpMvHs,-VmMvHs,VpLvHs,-VmLvHs,vDvCns,vDvCns,vDvCns,vDvCns,cnsCns,cnsCns)))
    
    msk.options = {iparam.log: 0}
    sol = solvers.lp(c,A,b,solver='mosek')
    
    if sol['status']=='optimal':
        sts = 0
        sln = sol['x'][0]
        # sln = sum(sol['x'][1:])
    else:
        sts = 1
        sln = np.nan
    
    return sln,sts
    
def linSnsCalc(dVmv,dVlv,vDv,bLsMv,bHsMv,bLsLv,bHsLv,cns,DVmax=0.06):
    # as in the gm paper
    dVmv = dVmv[bLsMv>0.5]
    dVlv = dVlv[bLsLv>0.5]
    bLsMv = bLsMv[bLsMv>0.5]
    bHsMv = bHsMv[bHsMv>0.5]
    bLsLv = bLsLv[bLsLv>0.5]
    bHsLv = bHsLv[bHsLv>0.5]
    
    # build the b-matrix:
    VpMv = cns.VpMv - np.maximum(bLsMv,bHsMv)
    VpLv = cns.VpLv - np.maximum(bLsLv,bHsLv)
    VmMv = cns.VmMv - np.minimum(bLsMv,bHsMv)
    VmLv = cns.VmLv - np.minimum(bLsLv,bHsLv)
    
    vDvCnsU = DVmax*np.ones((len(vDv)))
    vDvCnsL = -DVmax*np.ones((len(vDv)))
    
    cUp = np.concatenate((VpMv,VpLv,vDvCnsU))
    cLo = np.concatenate((VmMv,VmLv,vDvCnsL))
    kMult = np.concatenate((dVmv,dVlv,vDv))
    
    ltcUp = (cUp/kMult)[kMult>=0]
    ltcLo = (cLo/kMult)[kMult<0]
    x = np.min(np.concatenate((ltcUp,ltcLo)))
    return x

def pos2str(tapPos):
    hexPos = ''
    for pos in tapPos:
        hexPos = hexPos + hex(pos)
    return hexPos

def calcVar(X):
    i=0
    var = np.zeros(len(X))
    for x in X:
        var[i] = x.dot(x)
        if var[i]==0:
            var[i]=1e-100
        i+=1
    return var
    
def calcDp1rSqrt(d,e,n):
    # Calculates the diagonal and off-diagonal elements of a 
    # matrix with diagonal elements d and off-diagonal elements
    # e, of dimension n x n. 
    # e.g. A = np.diag(np.ones(n))*(d-e) + np.ones((n,n))*e
    # X = la.sqrtm(A)
    # a2 = X[0,0]**2
    # b2 = X[1,0]**2
    
    a = 4*(n-1) + (n-2)**2
    b = -( 2*(n-2)*e + 4*d )
    c = e**2
    
    b2ii = (-b-np.sqrt( b**2 - 4*a*c ))/(2*a) # <- this seems to give the right solution
    a2ii = d - (n-1)*b2ii
    
    dSqrt = np.sqrt(a2ii)
    eSqrt = -np.sqrt(b2ii)
    # X = (d-e)*np.diag(np.ones(n)) + e*np.ones((n,n)) # remember during testing to do X.dot(X) (not X*X!)
    return dSqrt, eSqrt

def dp1rMv(M,d,e): # diagonal d plus rank-1 e right multiplication
    return M*(d-e) + np.outer(e*(np.sum(M,axis=1)),np.ones(M.shape[1]))

# =================================== PLOTTING FUNCTIONS

def plotCns(mu_k,prms,Cns_pct,ax=None,pltShow=True,feeder=None,lineStyle='-',clrs=None):
    if ax==None:
        fig, ax = plt.subplots()
    if clrs==None:
        clrs = cm.nipy_spectral(np.linspace(0,1,9))
        ax.set_prop_cycle(color=clrs)
    # plt.plot(pdfData['mu_k'],Cns_pct_lin[0],'--')
    if len(Cns_pct.shape)==2:
        x_vals = mu_k
        y_vals = Cns_pct
        # ax.plot(mu_k,Cns_pct,lineStyle)
    elif Cns_pct.shape[0]==1:
        x_vals = mu_k
        y_vals = Cns_pct[0]
        # ax.plot(mu_k,Cns_pct[0],lineStyle)
    else:
        x_vals = prms
        y_vals = Cns_pct[:,0,:]
        # ax.plot(mu_k,Cns_pct[:,0,:],lineStyle)
    ax.plot(x_vals,y_vals,lineStyle)
    
    plt.xlabel('Scale factor');
    plt.ylabel('P(.), %');
    if not feeder==None:
        plt.title('Constraints, '+feeder)
    else:
        plt.title('Constraints')
    plt.legend(('$\Delta V$','$V^{+}_{\mathrm{MV,LS}}$','$V^{-}_{\mathrm{MV,LS}}$','$V^{+}_{\mathrm{LV,LS}}$','$V^{-}_{\mathrm{LV,LS}}$','$V^{+}_{\mathrm{MV,HS}}$','$V^{-}_{\mathrm{MV,HS}}$','$V^{+}_{\mathrm{LV,HS}}$','$V^{-}_{\mathrm{LV,HS}}$'))
    if pltShow:
        plt.show()
        ax = None; fig = None
    return ax

def plotHcVltn(mu_k,prms,Vp_pct,ax=None,pltShow=True,feeder=None,lineStyle='.-',logScale=True):
    if len(mu_k)>1:
        x_vals = mu_k
        y_vals = Vp_pct[0]
    elif len(prms)>1:
        x_vals = prms
        y_vals = Vp_pct[:,0]
    
    if ax==None:
        fig, ax = plt.subplots()
    if logScale:
        ax.semilogy(x_vals,y_vals,lineStyle)
        ax.set_title('Prob. of violation (logscale), '+feeder);
        ax.set_ylabel('Log [ P(.), % ]')
    else:
        ax.plot(x_vals,y_vals,lineStyle)
        ax.set_title('Prob. of violation '+feeder);
        ax.set_ylabel('P(.), %')
    ax.set_xlabel('Scale factor');
    ax.grid(True)
    if pltShow:
        plt.show()
        ax = None; fig = None
    return ax
    
def plotPwrCdf(pp,ppPdf,ax=None,pltShow=True,feeder=None,lineStyle='-'):
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(pp,ppPdf,LineStyle=lineStyle)
    ax.set_xlabel('Power')
    ax.set_ylabel('Power')
    ax.grid(True)
    if pltShow:
        plt.show()
        ax = None; fig = None
    return ax
    
def plotHcGen(mu_k,prms,hcGenSet,lineColor,ax=None):
    if ax==None:
        fig, ax = plt.subplots()
    
    if len(prms)==1:
        x_vals = mu_k
        y_vals = hcGenSet[0,:,:]
    if len(mu_k)==1:
        x_vals = prms
        y_vals = hcGenSet[:,0,:]
    
    ax.plot(x_vals,y_vals[:,0],lineColor+'-')
    ax.plot(x_vals,y_vals[:,1],lineColor+'_')
    ax.plot(x_vals,y_vals[:,2],lineColor+'-')
    
    return ax

def plotBoxWhisk(ax,x,ddx,y,clr=cm.tab10(0),zOrder=10,lineWidth=1.0,bds=[None],transpose=False):
    if not transpose:
        if bds[0]!=None:
            ax.plot([x],bds[0],color=clr, zorder=zOrder,marker='^',markersize=5,markerfacecolor='none')
            ax.plot([x],bds[1],color=clr, zorder=zOrder,marker='v',markersize=5,markerfacecolor='none')
        ax.plot([x]*2,y[0:2],'--',color=clr,zorder=zOrder,linewidth=lineWidth)
        
        ax.plot([x-ddx,x+ddx],[y[0]]*2,color=clr,zorder=zOrder,linewidth=lineWidth)
        ax.plot([x-ddx,x+ddx],[y[4]]*2,color=clr,zorder=zOrder,linewidth=lineWidth)
        
        ddx2 = ddx/1.5
        box = patches.Rectangle(xy=(x-ddx2,y[1]),width=ddx2*2,height=y[3]-y[1],edgecolor=clr,facecolor='w',zorder=zOrder,linewidth=lineWidth)
        ax.add_patch(box)
        ax.plot([x-ddx2,x+ddx2],[y[2]]*2,color=clr,zorder=zOrder,linewidth=lineWidth)
        ax.plot([x]*2,y[3::],'--',color=clr,zorder=zOrder,linewidth=lineWidth)
    if transpose:
        if bds[0]!=None:
            ax.plot(bds[0]-1,[x],color=clr, zorder=zOrder,marker='>',markersize=5,markerfacecolor='none')
            ax.plot(bds[1]+1,[x],color=clr, zorder=zOrder,marker='<',markersize=5,markerfacecolor='none')
        ax.plot(y[0:2],[x]*2,'--',color=clr,zorder=zOrder,linewidth=lineWidth)
        
        ax.plot([y[0]]*2,[x-ddx,x+ddx],color=clr,zorder=zOrder,linewidth=lineWidth)
        ax.plot([y[4]]*2,[x-ddx,x+ddx],color=clr,zorder=zOrder,linewidth=lineWidth)
        
        ddx2 = ddx/1.5
        box = patches.Rectangle(xy=(y[1],x-ddx2),width=y[3]-y[1],height=ddx2*2,edgecolor=clr,facecolor='w',zorder=zOrder,linewidth=lineWidth)
        ax.add_patch(box)
        ax.plot([y[2]]*2,[x-ddx2,x+ddx2],color=clr,zorder=zOrder,linewidth=lineWidth)
        ax.plot(y[3::],[x]*2,'--',color=clr,zorder=zOrder,linewidth=lineWidth)
    return ax

def getKcdf(param,Vp_pct):
    if np.any(Vp_pct):
        kCdf = [param[np.argmax(Vp_pct!=0)]]
    else:
        kCdf = [np.nan]
    for centile in np.arange(5.,105.,5.): # NB in %, not probability
        if np.any(Vp_pct>=centile):
            kCdf.append(param[np.argmax(Vp_pct>=centile)])
        else:
            kCdf.append(np.nan)
    xCdf = [0.]+(np.arange(5.,105.,5.).tolist())
    return np.array(kCdf),np.array(xCdf)

# =================================== CLASS: linModel
class linModel:
    """Linear model class with a whole bunch of useful things that we can do with it."""
    def __init__(self,fdr_i,WD,QgenPf=1.00,setCapsModel='linModel',kDroop=None,aDroop=None):
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr']
        fdrNetModels = [0,0,0,0,0,1,1,0,1,2,-1,-1,-1,-1,0,-1,-1,0,0,2,2,2,2]
        feeder = fdrs[fdr_i]
        print('Loading linModel from linSvdCalcs. Feeder:'+feeder+', QgenPf:'+str(QgenPf))
        
        self.feeder = feeder
        self.WD = WD # for debugging
        self.QgenPf = QgenPf # nominally have a unity pf
        self.kDroopV = None # nominal values - these may get updated lated
        self.kDroopDV = None # nominal values - these may get updated lated
        self.aDroopV = None # nominal values - these may get updated lated
        
        self.netModelNom = fdrNetModels[fdr_i]
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
            lp0data = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','regBndwth.pkl'),'rb') as handle:
            self.regBandwidth = pickle.load(handle)
        
        self.linPoint = lp0data['k']
        self.VpMv = lp0data['VpMv']
        self.VmMv = lp0data['VmMv']
        self.VpLv = lp0data['VpLv']
        self.VmLv = lp0data['VmLv']
        self.nRegs = lp0data['nRegs']
        self.vSrcBus = lp0data['vSrcBus']
        self.srcReg = lp0data['srcReg']
        self.legLoc = lp0data['legLoc']
        self.DVmax = 0.06 # pu
        
        self.loadPointLo = lp0data['kLo']
        self.loadPointHi = lp0data['kHi']
        self.loadScaleNom = lp0data['kLo'] - lp0data['k']
        
        # with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','busCoords.pkl'),'rb') as handle:
            # self.busCoords = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','busCoordsAug.pkl'),'rb') as handle:
            self.busCoords = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','branches.pkl'),'rb') as handle:
            self.branches = pickle.load(handle)
        
        
        # load the fixed model, as this always exists
        LMfxd = loadLinMagModel(self.feeder,self.linPoint,WD,'Lpt',regModel=False)
        Kyfix=LMfxd['Ky'];Kdfix=LMfxd['Kd']
        dvBase = LMfxd['vKvbase'] # NB: this is different to vBase for ltc/regulator models!
        
        self.LMfxd = {'Ky':Kyfix,'Kd':Kdfix,'bV':LMfxd['bV'],'Kt':LMfxd['Kt'],'xhy0':LMfxd['xhy0'],'xhd0':LMfxd['xhd0'],'xhyCap0':LMfxd['xhyCap0'],'xhdCap0':LMfxd['xhdCap0'],'xhyLds0':LMfxd['xhyLds0'],'xhdLds0':LMfxd['xhdLds0']}
        self.dvBase = dvBase
        self.vFixYNodeOrder = LMfxd['vYNodeOrder']
        self.v_idx_fix = LMfxd['v_idx']
        
        
        self.updateFxdModel()
        self.nV, self.nS = self.KfixPu.shape
        self.nSy = Kyfix.shape[1];         self.nSd = Kdfix.shape[1]
        self.nT = self.LMfxd['Kt'].shape[1]
        self.KtPu = dsf.vmM(1/self.dvBase,self.LMfxd['Kt'])
        self.bVfixPu = self.LMfxd['bV']/self.dvBase
        
        self.pIs = np.concatenate( (np.arange(0,self.nSy//2),np.arange(self.nSy,self.nSy+self.nSd//2)) )
        self.qIs = np.concatenate( (np.arange(self.nSy//2,self.nSy),np.arange(self.nSy+self.nSd//2,self.nSy+self.nSd)) )
        
        self.setCapsModel=setCapsModel
        self.capPosLin = lp0data['capPosOut']
        
        if self.netModelNom>0:
            LM = loadNetModel(self.feeder,self.linPoint,self.WD,'Lpt',self.netModelNom)
            self.vTotBase = LM['Vbase']
            self.idxShf = LM['idxShf']
            self.regVreg0 = LM['regVreg']
            self.regVregPu = self.regVreg0/self.vTotBase[-self.nT:]
            
            v_idx_shf,s_idx_shf,sD_idx_shf = self.idxShf[:]
            KyR = self.LMfxd['Ky'][v_idx_shf]
            KdR = self.LMfxd['Kd'][v_idx_shf]
            bVR = self.LMfxd['bV'][v_idx_shf]
            KtR = self.LMfxd['Kt'][v_idx_shf]
            
            self.KfixPuShf = dsf.vmM(1/self.vTotBase,np.c_[KyR,KdR])
            self.bVPuShf = bVR/self.vTotBase
            self.KtPuShf = dsf.vmM(1/self.vTotBase,KtR)*(0.1/16)

        self.loadNetModel()
        
        
        
        # if kDroop!=None and aDroop!=None:
            # self.updateDroopCoeffs(kDroop,aDroop) # NOT YET FUNCTIONAL!
        # elif kDroop!=None:
            # self.updateDroopCoeffs(kDroop) # NOT YET FUNCTIONAL!

    # def updateDroopCoeffs(self,kDroop,aDroop=1.0):
        # # converts droop coefficients in VAr per pu to VAr per V (in the right base). NOT YET FUNCTIONAL!
        # self.kDroopDV = kDroop*self.dvBase # droop slope ( W per volt, e.g. -100 means 500 var at 1.05V )
        
        # self.updateFxdModel()
        
        # self.kDroopV = kDroop*self.vTotBase # droop slope ( W per volt, e.g. -100 means 500kVar at 1.05V )
        # self.aDroopV = aDroop*self.vTotBase # droop intercept (volts)
        
    def getTapPosition(self,pGen,seq=False):
        # 'fix' model is the load flow model; 'tot' is the kron redct. model.
        # sGen needs to be in watts (not kW)
        if self.netModelNom>0:
            pGenY = pGen[:self.nSy//2]
            pGenD = pGen[self.nSy//2:]
            k_Q = pf2kq(self.QgenPf)
            xGen = np.r_[pGenY,k_Q*pGenY,pGenD,k_Q*pGenD] # following Kfix0Pu     
            if self.netModelNom==1:
                VregPu = self.regVregPu
                
                VcompPuLo = self.regIdxMatVlts.dot(self.xTotLs + xGen)/self.vTotBase[-self.nT:]
                VcompPuHi = self.regIdxMatVlts.dot(self.xTotHs + xGen)/self.vTotBase[-self.nT:]
                
                VldcPuLo = VregPu + VcompPuLo
                VldcPuHi = VregPu + VcompPuHi
            if self.netModelNom==2:
                VldcPuLo = self.regVregPu
                VldcPuHi = VldcPuLo
            
            b0lsFxd = self.KfixPuShf.dot(self.xTotLs + xGen) + self.bVPuShf
            b0hsFxd = self.KfixPuShf.dot(self.xTotHs + xGen) + self.bVPuShf
            
            KtReg = self.KtPuShf[-self.nT:]
            
            if not seq:
                tSetLo = np.linalg.solve( KtReg, VldcPuLo - b0lsFxd[-self.nT:] ) # this is already 'per tap'.
                tSetHi = np.linalg.solve( KtReg, VldcPuHi - b0hsFxd[-self.nT:] )
            else:
                
                tSetLo = np.array([])
                for i in range(self.nT):
                    A = KtReg[i:,i:]
                    b = (VldcPuLo - b0lsFxd[-self.nT:])[i:] - KtReg[i:,:i].dot(tSetLo)
                    x = np.linalg.solve(A,b)
                    tSetLo = np.r_[tSetLo,np.round(x[0])]
                tSetHi = np.array([])
                for i in range(self.nT):
                    A = KtReg[i:,i:]
                    b = (VldcPuHi - b0hsFxd[-self.nT:])[i:] - KtReg[i:,:i].dot(tSetHi)
                    x = np.linalg.solve(A,b)
                    tSetHi = np.r_[tSetHi,np.round(x[0])]
        
            b0ls = self.KfixPuShf.dot(self.xTotLs + xGen) + self.KtPuShf.dot(tSetLo) + self.bVPuShf
            b0hs = self.KfixPuShf.dot(self.xTotHs + xGen) + self.KtPuShf.dot(tSetHi) + self.bVPuShf
            
            vHiHs = np.max(b0hs)
            vHiLs = np.max(b0ls)
            
            KtPuPt = self.KtPuShf
            
            tSetLoSns = np.zeros(self.nT)
            tSetHiSns = np.zeros(self.nT)
            for i in range(KtPuPt.shape[1]):
                vSnsLo = ((self.VpMv - b0ls)/KtPuPt[:,i]) + 1e100*(KtPuPt[:,i]<0.1*(0.1/16))
                tSetLoSns[i] = np.min(vSnsLo)
                vSnsHi = ((self.VpMv - b0hs)/KtPuPt[:,i]) + 1e100*(KtPuPt[:,i]<0.1*(0.1/16))
                tSetHiSns[i] = np.min(vSnsHi)
        else:
            tSetHi = [] 
            tSetLo = []
            tSetHiSns = []
            tSetLoSns = []
            vHiHs = np.nan
            vHiLs = np.nan
            
        return tSetHi,tSetLo,tSetHiSns,tSetLoSns,vHiHs,vHiLs
        
    def updateFxdModel(self):
        Ky = self.LMfxd['Ky']
        Kd = self.LMfxd['Kd']
        
        KyPfix = Ky[:,:Ky.shape[1]//2]
        KyQfix = Ky[:,Ky.shape[1]//2::]
        KdPfix = Kd[:,:Kd.shape[1]//2]
        KdQfix = Kd[:,Kd.shape[1]//2::]
        
        # if self.kDroopDV==None:
        k_Q = pf2kq(self.QgenPf)
        Kfix = np.concatenate((KyPfix+k_Q*KyQfix,KdPfix + k_Q*KdQfix),axis=1)
        # else:
            # KfixY = np.linalg.solve( np.eye(len(KyPfix)) - KyQfix*self.kDroopDV,KyPfix )
            # KfixD = np.linalg.solve( np.eye(len(KdPfix)) - KdQfix*self.kDroopDV,KdPfix )
            # Kfix = np.concatenate((KyPfix+k_Q*KyQfix,KdPfix + k_Q*KdQfix),axis=1)
            
        self.KfixPu = dsf.vmM(1/self.dvBase,Kfix)
        self.Kfix0Pu = dsf.vmM(1/self.dvBase,np.concatenate((Ky,Kd),axis=1))
        
        self.KpFix0Pu = dsf.vmM(1/self.dvBase,np.concatenate((KyPfix,KdPfix),axis=1))
        self.KqFix0Pu = dsf.vmM(1/self.dvBase,np.concatenate((KyQfix,KdQfix),axis=1)) # for gen reactive power control
        
    def loadNetModel(self,netModel=None):
        if netModel==None:
            netModel = self.netModelNom
        
        if netModel==0:
            # IF using the FIXED model:
            LM = loadLinMagModel(self.feeder,self.linPoint,self.WD,'Lpt',regModel=False)
            Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV']
            self.vTotBase = LM['vKvbase']
            A = np.concatenate((Ky,Kd),axis=1)
        else:
            # IF using the LTC of DCP model:
            LM = loadNetModel(self.feeder,self.linPoint,self.WD,'Lpt',netModel)
            A=LM['A'];bV=LM['B']
            self.vTotBase = LM['Vbase']
            
        xhyCap0=LM['xhyCap0'];xhdCap0=LM['xhdCap0'];xhyLds0=LM['xhyLds0'];xhdLds0=LM['xhdLds0']
        
        self.xTotCap = np.concatenate((xhyCap0,xhdCap0))
        
        self.xhyNtot = LM['xhy0']/self.linPoint # half deprecated - still useful for halfLoadMean.
        self.xhdNtot = LM['xhd0']/self.linPoint # half deprecated - still useful for halfLoadMean.
        self.xNomTot = np.concatenate((self.xhyNtot,self.xhdNtot))
        
        self.xTotLds = np.concatenate((xhyLds0,xhdLds0))/self.linPoint
        
        if self.setCapsModel==True:
            self.xTotLs = (self.xNomTot*self.loadPointLo)
            self.xTotHs = (self.xNomTot*self.loadPointHi)        
        else:
            self.xTotLs = (self.xTotLds*self.loadPointLo) + self.xTotCap
            self.xTotHs = (self.xTotLds*self.loadPointHi) + self.xTotCap
        
        self.vTotYNodeOrder = LM['vYNodeOrder']
        
        self.mvIdxTot = np.where(self.vTotBase>1000)[0]
        self.lvIdxTot = np.where(self.vTotBase<=1000)[0]
        
        self.mvIdxFxd = np.where(self.dvBase>1000)[0]
        self.lvIdxFxd = np.where(self.dvBase<=1000)[0]
        
        self.mlvIdx = np.concatenate((self.mvIdxTot,self.lvIdxTot))
        self.mlvUnIdx = np.argsort(self.mlvIdx)        
        
        self.nVmv = len(self.mvIdxTot)
        self.nVlv = len(self.lvIdxTot)
        
        self.updateTotModel(A,bV)
        
        if netModel==1 or netModel==2: # decoupling regulator model
            self.idxShf = LM['idxShf']
            self.regVreg0 = LM['regVreg']
        if netModel==1:
            self.regIdxMatVlts = LM['regIdxMatVlts']
        self.v_idx_tot = LM['v_idx']
        
        self.vTotBaseMv = self.vTotBase[self.mvIdxTot]
        self.vTotBaseLv = self.vTotBase[self.lvIdxTot]
        
        self.SyYNodeOrderTot = LM['SyYNodeOrder']
        self.SdYNodeOrderTot = LM['SdYNodeOrder']
        
    def runLinHc(self,pdf,model='nom',fast=False):
        nCnstr = 9
        
        pdfData = pdf.pdf
        nMc = pdfData['nMc']
        
        Vp_pct = np.zeros(pdfData['nP'])
        Cns_pct = np.zeros(list(pdfData['nP'])+[nCnstr])
        hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
        hcGenAll = np.array([])
        genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
        genTotAll = np.nan*np.zeros((pdfData['nP'][0],nMc*pdfData['nP'][1])) # NB this gets flattened later
        nV = self.KtotPu.shape[0]
        nS = self.KtotPu.shape[1]
        if fast:
            lpPct = np.nan
        else:
            lpPct = np.zeros(list(pdfData['nP'])+[nMc])
        
        
        tapPos = np.zeros((pdfData['nP'][0],nMc,2,self.nT)) # 2 load points. Only designed to work with some options...
        tapPosSeq = tapPos.copy()
        tapPosSns = tapPos.copy()
        vChecks = np.zeros((pdfData['nP'][0],nMc,2))
        
        if model=='nom':
            NSetTot = np.arange(nV)
            NSetFix = NSetTot
        elif model=='std' or model=='cor':
            if model=='std':
                vars = self.varKtotU.copy() # <--- not yet implemented 'full' model.
                varSortN = vars.argsort()[::-1]
                NSetTot = varSortN[0:self.NSetStd[0]]
                NSetFix = np.arange(nV)
            elif model=='cor':
                vars = self.varKfullU.copy()
                varSortN = vars.argsort()[::-1]
                NSet = np.array(varSortN[self.NSetCor[0]])
                NSetTot = NSet[NSet<nV]
                NSetFix = NSet[NSet>=nV] - nV
                
        # draw samples before the clock, as this proving quite slow.        
        tStartSample = time.time()
        Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot) # in W
        pdfMcAll = []
        for i in range(pdfData['nP'][0]):
            pdfMcAll.append(pdf.genPdfMcSet(nMc,Mu,i)[0]) # pdfMc in kW (Mu is in W)#
        tSampling = time.time() - tStartSample
        
        tStart = time.process_time()
        tStartClk = time.time()
        
        mvIdxNSet = np.where(self.vTotBase[NSetTot]>1000)[0] # as in mvIdxTot
        lvIdxNSet = np.where(self.vTotBase[NSetTot]<1000)[0] # as in lvIdxTot
        
        KtotPuCalc = self.KtotPu[NSetTot]
        KfixPuCalc = self.KfixPu[NSetFix]

        for i in range(pdfData['nP'][0]):
            # Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot) # in W
            # pdfMc = pdf.genPdfMcSet(nMc,Mu,i)[0] # pdfMc in kW (Mu is in W)
            pdfMc = pdfMcAll[i]
            
            genTot0 = np.sum(pdfMc,axis=0)
            genTotSort = genTot0.copy()
            genTotSort.sort()
            genTotAll0 = np.outer(genTot0,pdfData['mu_k'])
            genTotAll[i] = genTotAll0.flatten()
            
            # DvMu0 = self.KtotU.dot(np.sqrt(pdfData['prms'])*np.ones(nS))
            
            DelVout = (KtotPuCalc.dot(pdfMc).T)*1e3 # KtotPu in V per W
            ddVout = abs((KfixPuCalc.dot(pdfMc).T)*1e3) # just get abs change
            
            b0ls = self.b0ls[NSetTot]
            b0hs = self.b0hs[NSetTot]
            b0lsMv = b0ls[mvIdxNSet]
            b0lsLv = b0ls[lvIdxNSet]
            b0hsMv = b0hs[mvIdxNSet]
            b0hsLv = b0hs[lvIdxNSet]
                
            for jj in range(pdfData['nP'][-1]):
                genTot = genTot0*pdfData['mu_k'][jj]
                
                # vLsMv = ((DelVout*pdfData['mu_k'][jj]) + b0ls)[:,self.mvIdxTot]
                # vLsLv = ((DelVout*pdfData['mu_k'][jj]) + b0ls)[:,self.lvIdxTot]
                # vHsMv = ((DelVout*pdfData['mu_k'][jj]) + b0hs)[:,self.mvIdxTot]
                # vHsLv = ((DelVout*pdfData['mu_k'][jj]) + b0hs)[:,self.lvIdxTot]
                
                DVoP = DelVout*pdfData['mu_k'][jj]
                DVoPmv = DVoP[:,mvIdxNSet]
                DVoPlv = DVoP[:,lvIdxNSet]
                vLsMv = DVoPmv + b0lsMv
                vLsLv = DVoPlv + b0lsLv
                vHsMv = DVoPmv + b0hsMv
                vHsLv = DVoPlv + b0hsLv
                
                vDv = ddVout*pdfData['mu_k'][jj]
                
                # vDv = ddVout[:,self.mvIdxTot]*pdfData['mu_k'][jj]
                
                Cns_pct[i,jj], inBounds = cnsBdsCalc(vLsMv,vLsLv,vHsMv,vHsLv,vDv,self)
                
                Vp_pct[i,jj] = 100*sum(inBounds)/nMc
                hcGen = genTot[inBounds]
                
                if not fast:
                    # Running the sensitivity analysis
                    for kk in range(nMc):
                        lpPct[i,jj,kk] = linSnsCalc(DVoPmv[kk],DVoPlv[kk],vDv[kk],b0lsMv,b0hsMv,b0lsLv,b0hsLv,self)
                        gtp = self.getTapPosition(pdfMc[:,kk]*1e3*pdfData['mu_k'][jj],seq=False)
                        tapPos[i,kk] = gtp[:2]
                        tapPosSns[i,kk] = gtp[2:4]
                        vChecks[i,kk] = gtp[4:]
                        gtp = self.getTapPosition(pdfMc[:,kk]*1e3*pdfData['mu_k'][jj],seq=True)
                        tapPosSeq[i,kk] = gtp[:2]

                if len(hcGen)!=0:
                    hcGenAll = np.concatenate((hcGenAll,hcGen))
                    hcGen.sort()
                    hcGenSet[i,jj,0] = hcGen[0]
                    hcGenSet[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
                    hcGenSet[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
                    hcGenSet[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
                    hcGenSet[i,jj,4] = hcGen[-1]
                genTotSet[i,jj,0] = genTotSort[0]*pdfData['mu_k'][jj]
                genTotSet[i,jj,1] = genTotSort[np.floor(len(genTotSort)*1.0/4.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,2] = genTotSort[np.floor(len(genTotSort)*1.0/2.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,3] = genTotSort[np.floor(len(genTotSort)*3.0/4.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,4] = genTotSort[-1]*pdfData['mu_k'][jj]
        tEnd = time.process_time()
        tEndClk = time.time()
        
        binNo = max(pdf.pdf['nP'])//2
        genTotAll = genTotAll.flatten()
        hist1 = plt.hist(genTotAll,bins=max(binNo,1),range=(0,max(genTotAll)))
        hist2 = plt.hist(hcGenAll,bins=max(binNo,1),range=(0,max(genTotAll)))
        pp = hist1[1][1:]
        ppPdf = 100*hist2[0]/hist1[0] # <<<<< still to be tested!!!!
        plt.close()
        
        if pdfData['name']=='gammaWght':
            param = pdfData['mu_k']
        elif pdfData['name']=='gammaFrac':
            param = pdfData['prms']
        
        
        self.linHcRsl = {}
        
        self.linHcRsl['pp'] = pp
        self.linHcRsl['ppPdf'] = ppPdf
        self.linHcRsl['ppCdf'],self.linHcRsl['xCdf'] = getKcdf(pp,ppPdf)
        
        self.linHcRsl['hcGenSet'] = hcGenSet
        self.linHcRsl['Vp_pct'] = Vp_pct
        self.linHcRsl['Cns_pct'] = Cns_pct
        self.linHcRsl['Lp_pct'] = lpPct
        self.linHcRsl['hcGenAll'] = hcGenAll
        self.linHcRsl['genTotSet'] = genTotSet
        self.linHcRsl['runTime'] = tEnd - tStart
        self.linHcRsl['runTimeClk'] = tEndClk - tStartClk
        self.linHcRsl['runTimeSample'] = tSampling
        self.linHcRsl['kCdf'] = 100*getKcdf(param,Vp_pct)[0]
        self.linHcRsl['kCdf'][np.isnan(self.linHcRsl['kCdf'])] = 100.0000011111
        self.linHcRsl['tapPos'] = tapPos
        self.linHcRsl['tapPosSeq'] = tapPosSeq
        self.linHcRsl['tapPosSns'] = tapPosSns
        
        self.vChecks = vChecks # for debugging
        self.pdfMcAll = pdfMcAll
    
    def runDssHc(self,pdf,DSSObj,genNames,BB0,SS0,regBand=0,setCapsModel='linModel',runType='par',tapPosStart=None):
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSSolution = DSSCircuit.Solution
        
        
        pdfData = pdf.pdf
        nMc = pdfData['nMc']
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr']
        
        fdr_i = fdrs.index(self.feeder)
        
        nCnstr = 9
        nRcd = 5
        
        Vp_pct = np.zeros(pdfData['nP'])
        Cns_pct = np.zeros(list(pdfData['nP'])+[nCnstr])
        hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nRcd))
        genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nRcd))
        
        genTotAll = np.nan*np.zeros((pdfData['nP'][0],nMc*pdfData['nP'][1])) # NB this gets flattened later
        hcGenAll = np.array([])
        
        v_idx = self.v_idx_tot
        
        regVI = np.zeros((nMc,pdfData['nP'][0],2,self.nT))
        
        R,X = getRx(DSSCircuit)
        zReg = np.diag(R) + 1j*np.diag(X)
        Ic,Vr = getRegIcVr(DSSCircuit)
        BW,Vrto = getRegBwVrto(DSSCircuit)
        
        if fdr_i != 6:
            DSSText.Command='Batchedit load..* vmin=0.33 vmax=3.0 model=1'
        DSSText.Command='Batchedit generator..* vmin=0.33 vmax=3.0'
        
        if regBand!=0:
            DSSText.Command='Batchedit regcontrol..* band='+str(regBand) # 1v seems to be as low as we can set without bit problems
        
        # draw samples before the clock, as this proving quite slow.        
        tStartSample = time.time()
        Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot) # in W
        pdfMcAll = []
        for i in range(pdfData['nP'][0]):
            pdfMcAll.append(pdf.genPdfMcSet(nMc,Mu,i)[0]) # pdfMc in kW (Mu is in W)#
        tSampling = time.time() - tStartSample
        
        tStart = time.process_time()
        tStartClk = time.time()
        
        tapPosSet = np.zeros(tapPosStart.shape,dtype=int)
        
        capLoMc = []; capHiMc = []; capDvMc = []
        for i in range(pdfData['nP'][0]):
            if i%(max(pdfData['nP'])//4)==0:
                print('---- Start MC ----',time.process_time(),i+1,'/',pdfData['nP'][0])
            # Mu0 = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot)
            # pdfGen = pdf.genPdfMcSet(nMc,Mu0,i)[0]
            pdfGen = pdfMcAll[i]
            
            genTot0 = np.sum(pdfGen,axis=0)
            genTotSort = genTot0.copy()
            genTotSort.sort()
            genTotAll0 = np.outer(genTot0,pdfData['mu_k'])
            genTotAll[i] = genTotAll0.flatten()
            
            for jj in range(pdfData['nP'][-1]):
                genTot = genTot0*pdfData['mu_k'][jj]
                
                vLsMv = np.ones((nMc,self.nVmv))
                vHsMv = np.ones((nMc,self.nVmv))
                vLsLv = np.ones((nMc,self.nVlv))
                vHsLv = np.ones((nMc,self.nVlv))
                vDv = np.ones((nMc,self.nV))
                convLo = []; convDv = []; convHi = []
                capLo = []; capDv = []; capHi = [];
                print('\nDSS MC Run:',jj,'/',pdfData['nP'][-1])
                
                nYVV = len(DSSCircuit.YNodeVarray)//2
                vLsDss0 = np.zeros((nMc,nYVV),dtype=complex)
                vHsDss0 = np.zeros((nMc,nYVV),dtype=complex)
                vDv0 = np.zeros((nMc,nYVV),dtype=complex)
                
                cpf_set_loads(DSSCircuit,BB0,SS0,1.0,setCaps=True) # <--- reset any previous settings
                if runType=='seq': # nominal version
                    for j in range(nMc):
                        if j%(nMc//4)==0:
                            print(j,'/',nMc)
                        set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                        
                        # first solve for the high load point [NB: This order seems best!]
                        cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointHi,setCaps=False)
                        
                        DSSSolution.Solve()
                        convHi = convHi+[DSSSolution.Converged]
                        capHi.append(getCapPstns(DSSCircuit))
                        vHsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        # then low load point
                        cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointLo,setCaps=False)
                        
                        DSSSolution.Solve()
                        convLo = convLo+[DSSSolution.Converged]
                        capLo.append(getCapPstns(DSSCircuit))
                        vLsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        # finally solve for voltage deviation. 
                        DSSText.Command='Batchedit generator..* kW=0.001'
                        DSSText.Command='set controlmode=off'
                        DSSSolution.Solve()

                        convDv = convDv+[DSSSolution.Converged]
                        capDv.append(getCapPstns(DSSCircuit))
                        vDv0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        DSSText.Command='set controlmode=static'
                        
                        if convLo and convHi and convDv:
                            vLsMv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vLsLv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vHsMv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vHsLv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vDv[j,:] = abs(abs(vLsDss0[j,:]) - abs(vDv0[j,:]))[3:][v_idx]/self.vTotBase
                if runType=='par':
                    # first solve for the high load point
                    # cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointHi,setCaps=setCapsModel,capPos=self.capPosLin)
                    cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointHi,setCaps=False) # <--- caps should be at nominal values
                    for j in range(nMc):
                        set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                        DSSSolution.Solve()
                        convHi = convHi+[DSSSolution.Converged]
                        capHi.append(getCapPstns(DSSCircuit))
                        vHsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        
                        v0reg = (vHsDss0[j,:][3:][v_idx][-self.nT:])/Vrto
                        iv0reg = zReg.dot(self.getRegI(DSSCircuit)/np.array(Ic))
                        regVI[j,i,0,:] = (abs(v0reg + iv0reg) - Vr)/(np.array(BW)/2)
                        
                    # then low load point
                    # cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointLo,setCaps=setCapsModel,capPos=self.capPosLin)
                    cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointLo,setCaps=False) # <--- caps should be at nominal values
                    DSSSolution.Solve()
                    tapSlns = {}
                    for j in range(nMc):
                        set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                        DSSSolution.Solve()
                        convLo = convLo+[DSSSolution.Converged]
                        capLo.append(getCapPstns(DSSCircuit))
                        vLsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        v0reg = (vLsDss0[j,:][3:][v_idx][-self.nT:])/Vrto
                        iv0reg = zReg.dot(self.getRegI(DSSCircuit)/np.array(Ic))
                        regVI[j,i,1,:] = (abs(v0reg + iv0reg) - Vr)/(np.array(BW)/2)
                        
                        tapPos = find_tap_pos(DSSCircuit)
                        tapHex = pos2str(tapPos)
                        
                        try:
                            vDv0[j,:] = tapSlns[tapHex]
                            convDv = convDv+[DSSSolution.Converged]
                        except:    
                            # solve for voltage deviation. 
                            DSSText.Command='Batchedit generator..* kW=0.001'
                            DSSText.Command='set controlmode=off'
                            DSSSolution.Solve()
                            convDv = convDv+[DSSSolution.Converged]
                            tapSlns[tapHex] = tp_2_ar(DSSCircuit.YNodeVarray)
                            vDv0[j,:] = tapSlns[tapHex]
                            DSSText.Command='set controlmode=static'
                        
                    for j in range(nMc):
                        if convLo and convHi and convDv:
                            vLsMv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vLsLv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vHsMv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vHsLv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vDv[j,:] = abs(abs(vLsDss0[j,:]) - abs(vDv0[j,:]))[3:][v_idx]/self.vTotBase
                    
                if runType=='tapSetLock' or runType=='tapSet':
                    if runType=='tapSetLock':
                        DSSText.Command='batchedit regcontrol..* maxtapchange=0'
                    else:
                        DSSText.Command='batchedit regcontrol..* maxtapchange=1'
                    # first solve for the high load point
                    cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointHi,setCaps=False) # <--- caps should be at nominal values
                    for j in range(nMc):
                        set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                        self.set_taps( DSSCircuit, tapPosStart[i,j,0] ) # set the tap positions and turn off tap changes
                        
                        DSSSolution.Solve()
                        convHi = convHi+[DSSSolution.Converged]
                        capHi.append(getCapPstns(DSSCircuit))
                        vHsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        tapPosSet[i,j,0] = find_tap_pos(DSSCircuit)
                        
                        # get regulator voltage and current
                        if self.netModelNom>0:
                            v0reg = (vHsDss0[j,:][3:][v_idx][-self.nT:])/Vrto
                            iv0reg = zReg.dot(self.getRegI(DSSCircuit)/np.array(Ic))
                            regVI[j,i,0,:] = (abs(v0reg + iv0reg) - Vr)/(np.array(BW)/2)
                    
                    # then low load point
                    # cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointLo,setCaps=setCapsModel,capPos=self.capPosLin)
                    cpf_set_loads(DSSCircuit,BB0,SS0,self.loadPointLo,setCaps=False) # <--- caps should be at nominal values
                    DSSSolution.Solve()
                    tapSlns = {}
                    for j in range(nMc):
                        set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                        self.set_taps( DSSCircuit, tapPosStart[i,j,1] ) # set the tap positions and turn off tap changes
                        DSSSolution.Solve()
                        convLo = convLo+[DSSSolution.Converged]
                        capLo.append(getCapPstns(DSSCircuit))
                        vLsDss0[j,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                        
                        # print(find_tap_pos(DSSCircuit))
                        if self.netModelNom>0:
                            v0reg = (vLsDss0[j,:][3:][v_idx][-self.nT:])/Vrto
                            iv0reg = zReg.dot(self.getRegI(DSSCircuit)/np.array(Ic))
                            regVI[j,i,1,:] = (abs(v0reg + iv0reg) - Vr)/(np.array(BW)/2)
                        
                        tapPosSet[i,j,1] = find_tap_pos(DSSCircuit)
                        tapHex = pos2str(tapPosSet[i,j,1])
                        
                        try:
                            vDv0[j,:] = tapSlns[tapHex]
                            convDv = convDv+[DSSSolution.Converged]
                        except:
                            # solve for voltage deviation. 
                            DSSText.Command='Batchedit generator..* kW=0.001'
                            DSSText.Command='set controlmode=off'
                            DSSSolution.Solve()
                            convDv = convDv+[DSSSolution.Converged]
                            tapSlns[tapHex] = tp_2_ar(DSSCircuit.YNodeVarray)
                            vDv0[j,:] = tapSlns[tapHex]
                            DSSText.Command='set controlmode=static'
                        
                    for j in range(nMc):
                        if convLo and convHi and convDv:
                            vLsMv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vLsLv[j,:] = abs(vLsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vHsMv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.mvIdxTot]/self.vTotBaseMv
                            vHsLv[j,:] = abs(vHsDss0[j,:])[3:][v_idx][self.lvIdxTot]/self.vTotBaseLv
                            vDv[j,:] = abs(abs(vLsDss0[j,:]) - abs(vDv0[j,:]))[3:][v_idx]/self.vTotBase
                            
                            
                            
                    DSSText.Command='batchedit regcontrol..* maxtapchange=16'
                    
                    
                if sum(convLo+convHi+convDv)!=len(convLo+convHi+convDv):
                    print('\nNo. Converged:',sum(convLo+convHi+convDv),'/',nMc*3)
                
                # NOW: calculate the HC value:
                Cns_pct[i,jj], inBoundsDss = cnsBdsCalc(vLsMv,vLsLv,vHsMv,vHsLv,vDv,self)
                Vp_pct[i,jj] = 100*sum(inBoundsDss)/nMc
                hcGen = genTot[inBoundsDss]

                if len(hcGen)!=0:
                    hcGenAll = np.concatenate((hcGenAll,hcGen))
                    hcGen.sort()
                    hcGenSet[i,jj,0] = hcGen[0]
                    hcGenSet[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
                    hcGenSet[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
                    hcGenSet[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
                    hcGenSet[i,jj,4] = hcGen[-1]
                genTotSet[i,jj,0] = genTotSort[0]*pdfData['mu_k'][jj]
                genTotSet[i,jj,1] = genTotSort[np.floor(len(genTotSort)*1.0/4.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,2] = genTotSort[np.floor(len(genTotSort)*1.0/2.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,3] = genTotSort[np.floor(len(genTotSort)*3.0/4.0).astype(int)]*pdfData['mu_k'][jj]
                genTotSet[i,jj,4] = genTotSort[-1]*pdfData['mu_k'][jj]
            if i%(max(pdfData['nP'])//4)==0:
                print('MC complete.',time.process_time())
            capLoMc.append(capLo);capHiMc.append(capHi);capDvMc.append(capDv)
        tEnd = time.process_time()
        tEndClk = time.time()
        binNo = max(pdf.pdf['nP'])//2
        genTotAll = genTotAll.flatten()
        hist1 = plt.hist(genTotAll,bins=binNo,range=(0,max(genTotAll)))
        hist2 = plt.hist(hcGenAll,bins=binNo,range=(0,max(genTotAll)))
        pp = hist1[1][1:]
        ppPdf = 100*hist2[0]/hist1[0] # <<<<< still to be tested!!!!
        plt.close()
        
        if pdfData['name']=='gammaWght':
            param = pdfData['mu_k']
        elif pdfData['name']=='gammaFrac':
            param = pdfData['prms']
        
        self.dssHcRsl = {}
        self.dssHcRsl['pp'] = pp # NB ppDss==ppLin if both at the same time
        self.dssHcRsl['ppPdf'] = ppPdf 
        self.dssHcRsl['ppCdf'],self.dssHcRsl['xCdf'] = getKcdf(pp,ppPdf)
        self.dssHcRsl['genTotAll'] = genTotAll
        self.dssHcRsl['hcGenSet'] = hcGenSet
        self.dssHcRsl['Vp_pct'] = Vp_pct
        self.dssHcRsl['Cns_pct'] = Cns_pct
        self.dssHcRsl['hcGenAll'] = hcGenAll
        self.dssHcRsl['genTotSet'] = genTotSet
        self.dssHcRsl['runTime'] = tEnd - tStart
        self.dssHcRsl['runTimeClk'] = tEndClk - tStartClk
        self.dssHcRsl['kCdf'] = 100*getKcdf(param,Vp_pct)[0]
        self.dssHcRsl['kCdf'][np.isnan(self.dssHcRsl['kCdf'])] = 100.0000011111
        self.dssHcRsl['capLo'] = capLoMc
        self.dssHcRsl['capHi'] = capHiMc
        
        self.dssHcRsl['regVI'] = regVI
        self.dssHcRsl['tapPosSet'] = tapPosSet
        self.tapPosSet = tapPosSet
        
        
    def set_taps( self,DSSCircuit,dTapPos ):
        newTaps = (np.array(self.TC_No0) + np.round(dTapPos).astype(int)).tolist()
        # print(newTaps)
        # fix_tap_pos(DSSCircuit, self.TC_No0)
        fix_tap_pos(DSSCircuit, newTaps)
    
    def getRegI(self,DSSCircuit):
        RGC = DSSCircuit.RegControls
        i = RGC.First
        currents = []
        while i:
            DSSCircuit.SetActiveElement('Transformer.'+RGC.Transformer)
            i0 = tp_2_ar(DSSCircuit.ActiveElement.Currents)
            iChoose = len(i0)//2
            currents.append(i0[iChoose])
            i = RGC.Next
        return currents
        
        
    def runLinLp(self,pdf,model='nom',tmax=0.1,qmax=0.2):
        pdfData = pdf.pdf
        nMc = pdfData['nMc']
        
        Vp_pct = np.zeros(pdfData['nP'])
        lpPct = np.zeros(list(pdfData['nP'])+[nMc])
        lpSts = np.zeros(list(pdfData['nP'])+[nMc])
        
        nV = self.KtotPu.shape[0]
        nS = self.KtotPu.shape[1]

        if model=='nom':
            NSetTot = np.arange(nV)
            NSetFix = NSetTot
        elif model=='std' or model=='cor':
            if model=='std':
                vars = self.varKtotU.copy() # <--- not yet implemented 'full' model.
                varSortN = vars.argsort()[::-1]
                NSetTot = varSortN[0:self.NSetStd[0]]
                NSetFix = np.arange(nV)
            elif model=='cor':
                vars = self.varKfullU.copy()
                varSortN = vars.argsort()[::-1]
                NSet = np.array(varSortN[self.NSetCor[0]])
                NSetTot = NSet[NSet<nV]
                NSetFix = NSet[NSet>=nV] - nV
        tStart = time.process_time()
        tStartClk = time.time()
        
        mvIdxNSet = np.where(self.vTotBase[NSetTot]>1000)[0] # as in mvIdxTot
        lvIdxNSet = np.where(self.vTotBase[NSetTot]<1000)[0] # as in lvIdxTot
        
        KtotPuCalc = self.KtotPu[NSetTot]
        
        Kfix0PuCalc = self.Kfix0Pu[NSetFix] # for loads
        KfixPuCalc = self.KfixPu[NSetFix] # for generators
        
        # for the LP stuff
        KtPuCalcMv = self.KtPu[NSetFix][mvIdxNSet]
        KtPuCalcLv = self.KtPu[NSetFix][lvIdxNSet]
        
        KpPu = self.KpFix0Pu[NSetFix]
        KqPu = self.KqFix0Pu[NSetFix]
        
        b0lpLs = Kfix0PuCalc.dot(self.xTotLs) + self.bVfixPu.T
        b0lpHs = Kfix0PuCalc.dot(self.xTotHs) + self.bVfixPu.T
        b0lpLsMv = b0lpLs[mvIdxNSet]
        b0lpLsLv = b0lpLs[lvIdxNSet]
        b0lpHsMv = b0lpHs[mvIdxNSet]
        b0lpHsLv = b0lpHs[lvIdxNSet]
        
        self.b0lsLp = b0lpLs[np.argsort(NSetFix)]
        self.b0hsLp = b0lpHs[np.argsort(NSetFix)]

        for i in range(pdfData['nP'][0]):
            print(i)
            
            Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot) # in W
            pdfMc = pdf.genPdfMcSet(nMc,Mu,i)[0] # pdfMc in kW (Mu is in W)
            
            # DelVout = (KtotPuCalc.dot(pdfMc).T)*1e3 # KtotPu in V per W #
            # ddVout = abs((KfixPuCalc.dot(pdfMc).T)*1e3) # just get abs change
            
            DelVlp = (KfixPuCalc.dot(pdfMc).T)*1e3
            
            DelVlp = (KpPu.dot(pdfMc).T)*1e3
            DelVlq = (KqPu.dot(pdfMc).T)*1e3
            
            for jj in range(pdfData['nP'][-1]):
                # DVlP = DelVlp*pdfData['mu_k'][jj]
                # DVlPmv = DVlP[:,mvIdxNSet]
                # DVlPlv = DVlP[:,lvIdxNSet]
                DVlP = DelVlp*pdfData['mu_k'][jj]
                DVlPmv = DVlP[:,mvIdxNSet]
                DVlPlv = DVlP[:,lvIdxNSet]
                DVlQ = DelVlq*pdfData['mu_k'][jj]
                DVlQmv = DVlQ[:,mvIdxNSet]
                DVlQlv = DVlQ[:,lvIdxNSet]
                
                for kk in range(nMc):
                    lpPct[i,jj,kk],lpSts[i,jj,kk] = linPrgCalc(DVlPmv[kk],DVlPlv[kk],DVlQmv[kk],DVlQlv[kk],DVlP[kk],DVlQ[kk],b0lpLsMv,b0lpHsMv,b0lpLsLv,b0lpHsLv,KtPuCalcMv,KtPuCalcLv,self,tmax=tmax,qmax=qmax)
                
                Vp_pct[i,jj] = 100*np.sum(np.maximum((lpPct[i,jj]<1),np.isnan(lpPct[i,jj])))/nMc
                
        tEnd = time.process_time()
        tEndClk = time.time()
        
        if pdfData['name']=='gammaWght':
            param = pdfData['mu_k']
        elif pdfData['name']=='gammaFrac':
            param = pdfData['prms']
        
        self.linLpRsl = {}
        self.linLpRsl['Vp_pct'] = Vp_pct
        self.linLpRsl['Lp_pct'] = lpPct
        self.linLpRsl['runTime'] = tEnd - tStart
        self.linLpRsl['runTimeClk'] = tEndClk - tStartClk
        self.linLpRsl['kCdf'] = 100*getKcdf(param,Vp_pct)[0]
        self.linLpRsl['kCdf'][np.isnan(self.linLpRsl['kCdf'])] = 100.0000011111
        
        
    def getCovMat(self,getTotCov=True,getFixCov=False,getFullCov=False):
        if getTotCov:
            self.KtotUcov = self.KtotU.dot(self.KtotU.T)
            covScaling = np.sqrt(np.diag(self.KtotUcov))
            covScaling[covScaling==0] = np.inf # avoid divide by zero complaint
            self.KtotUcorr = dsf.vmvM(1/covScaling,self.KtotUcov,1/covScaling)        
        if getFixCov:
            self.KfixUcov = self.KfixU.dot(self.KfixU.T)
            covScaling = np.sqrt(np.diag(self.KfixUcov))
            covScaling[covScaling==0] = np.inf # avoid divide by zero complaint
            self.KfixUcorr = dsf.vmvM(1/covScaling,self.KfixUcov,1/covScaling)
        if getFullCov:
            KfullU = np.concatenate((self.KtotU,self.KfixU),axis=0)
            self.KfullUcov = KfullU.dot(KfullU.T)
            covScaling = np.sqrt(np.diag(self.KfullUcov))
            covScaling[covScaling==0] = np.inf # avoid divide by zero complaint
            self.KfullUcorr = dsf.vmvM(1/covScaling,self.KfullUcov,1/covScaling)
    
    def busViolationVar(self,Sgm,lim='all',Mu=np.array([None]),calcSrsVals=False):
        if lim=='all':
            limAct, limDV = self.updateKlims(Mu)
        
        # if lim=='VpLvLs': # <<< these have not been tested for quite a while.
            # if Mu[0]==None:
                # limAct = self.VpLv - self.b0ls
            # else:
                # limAct = self.VpLv - self.b0ls - self.KtotPu.dot(Mu)
            # limAct[self.b0ls<0.5] = np.inf
            # limAct[self.mvIdxTot] = np.inf
        # elif lim=='VpMvLs':
            # if Mu[0]==None:
                # limAct = self.VpMv - self.b0ls
            # else:
                # limAct = self.VpMv - self.b0ls - self.KtotPu.dot(Mu)
            # limAct[self.b0ls<0.5] = np.inf
            # limAct[self.lvIdxTot] = np.inf
            
        
        
        # VVV These uused to be used a lot so may need reinstating at some point...!
        # if Sgm.shape==(1,):
            # self.KtotU = dsf.vmM(1/limAct,self.KtotPu)*Sgm
            # self.KfixU = dsf.vmM(1/limDV,self.KfixPu)*Sgm
        # else:
            # self.KtotU = dsf.vmvM(1/limAct,self.KtotPu,Sgm)
            # self.KfixU = dsf.vmvM(1/limDV,self.KfixPu,Sgm)
        self.KtotU = dp1rMv( dsf.vmM(1/limAct,self.KtotPu),self.CovSqrt[0],self.CovSqrt[1] )
        self.KfixU = dp1rMv( dsf.vmM(1/limDV,self.KfixPu),self.CovSqrt[0],self.CovSqrt[1] )
        
        self.varKtotU = calcVar(self.KtotU)
        self.varKfixU = calcVar(self.KfixU)
        self.varKfullU = np.concatenate((self.varKtotU,self.varKfixU))
        self.svdLim = limAct
        self.svdLimDv = limDV
        self.nStdKtotU = np.sign(self.svdLim)/np.sqrt(self.varKtotU)
        self.nStdKfixU = np.sign(self.svdLimDv)/np.sqrt(self.varKfixU)
        
        self.nStdU = np.minimum( self.nStdKtotU,self.nStdKfixU )
        
        if calcSrsVals: # for speedy 'updateNormCalc'
            self.KtotUsrs = self.varKtotU.copy()
            self.KfixUsrs = self.varKfixU.copy()
            self.svdLim2 = self.svdLim**2
            self.svdLimDv2 = self.svdLimDv**2
        
    def updateNormCalc(self,Mu,inclSmallMu=True):
        if inclSmallMu:
            limAct00, limDV00 = self.updateKlims(Mu)
            limAct01, limDV01 = self.updateKlims(Mu*1e-6)
            limAct = np.minimum(limAct00,limAct01)
            limDV = np.minimum(limDV00,limDV01)
        else:
            limAct, limDV = self.updateKlims(Mu)
        
        limActScl = self.svdLim2/(limAct**2)
        limDvScl = self.svdLimDv2/(limDV**2)
        
        nSgmsTot = self.KtotUsrs*limActScl
        nSgmsFix = self.KfixUsrs*limDvScl
        
        nStdsTot = np.sign(limAct)/np.sqrt(nSgmsTot)
        nStdsFix = np.sign(limDV)/np.sqrt(nSgmsFix)
        
        Kfro = np.sqrt(np.sum( nSgmsTot+nSgmsFix ))
        Knstd = np.min(np.minimum(nStdsTot,nStdsFix))
        
        return Kfro, Knstd

        
    def updateKlims(self,Mu):
        if Mu[0]==None:
            dMu = 0
            dvMu = np.zeros(self.b0ls.shape)
            dMuMv = np.zeros(self.b0MvMax.shape)
            dMuLv = np.zeros(self.b0LvMax.shape)
        else:
            if Mu.shape==(1,):
                Mu = np.ones((self.KtotPu.shape[1]))*Mu
            dMu = self.KtotPu.dot(Mu)
            dMuMv = self.KtotPuMv.dot(Mu)
            dMuLv = self.KtotPuLv.dot(Mu)
            dvMu = self.KfixPu.dot(Mu)        
        # NEW VERSION
        limMvHi = self.VpMv - self.b0MvMax - dMuMv
        limMvLo = -(self.VmMv - self.b0MvMin - dMuMv)
        limLvHi = self.VpLv - self.b0LvMax - dMuLv
        limLvLo = -(self.VmLv - self.b0LvMin - dMuLv)
        
        # # Brickwall to stop there being undervoltages at no load
        # limMvHi[self.VpMv - self.b0MvMax<0] = -np.inf
        # limMvLo[self.VmMv - self.b0MvMin>0] = -np.inf
        # limLvHi[self.VpLv - self.b0LvMax<0] = -np.inf
        # limLvLo[self.VmLv - self.b0LvMin>0] = -np.inf
        limMvHi[self.VpMv - self.b0MvMax<0] = -1e100
        limMvLo[self.VmMv - self.b0MvMin>0] = -1e100
        limLvHi[self.VpLv - self.b0LvMax<0] = -1e100
        limLvLo[self.VmLv - self.b0LvMin>0] = -1e100
        
        limDvA = self.DVmax - dvMu # required so that min puts out a sensible answer?
        limDvB = -(-self.DVmax - dvMu)
        limLo = np.concatenate((limMvLo,limLvLo))[self.mlvUnIdx]
        limHi = np.concatenate((limMvHi,limLvHi))[self.mlvUnIdx]
        
        limAct = np.minimum(limLo,limHi)
        limDV = np.minimum(limDvA,limDvB)
        
        # limAct[self.b0ls<0.5] = np.inf # required still unfortunately
        # limAct[self.b0hs<0.5] = np.inf # required still unfortunately
        limAct[self.b0ls<0.5] = 1e100 # required still unfortunately
        limAct[self.b0hs<0.5] = 1e100 # required still unfortunately
        return limAct, limDV
    
    
    def makeSvdModel(self,Sgm,evSvdLim=[0.99],nMax=300):
        # method:
        # 1. take in voltage that can occur
        # 2. take in the mean and variance of the input
        # 3. pre and postmultiply to normalise
        # 4. calculate variance of the matrix
        # 
        # we generally use the ZERO MEAN version for simplicity
        # 
        # example:
        # print('Start Svd calcs...',time.process_time())
        # LM.makeSvdModel(Sgm,evSvdLim=[0.95,0.98,0.99,0.995,0.999],nMax=3500)
        
        self.busViolationVar(Sgm)
        
        nSvdMax = min([nMax,min(self.KtotPu.shape)]) - 1
        
        svd = TruncatedSVD(n_components=nSvdMax,algorithm='arpack') # see: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
        
        Us,Ss,Vhs,evS = dsf.trunc_svd(svd,self.KtotU)
        
        if len(evSvdLim)>1:
            NSvd = []
            for evsvdlim in evSvdLim:
                NSvd = NSvd + [np.argmax(evS>evsvdlim)]
            print('evSvdLim values:',evSvdLim)
            print('NSvd values:',NSvd)
            nSvd = NSvd[-1]
        else:
            nSvd = np.argmax(evS>evSvdLim)
        
        if nSvd==0:
            print('SVD failed, final evS:',evS[-1])
        
        print('Number of Components:',nSvd)
        print('Computational effort saved (%):',100*(1-(nSvd*(self.KtotU.shape[0] + self.KtotU.shape[1])/(self.KtotU.shape[0]*self.KtotU.shape[1]) )))
        
        UsSvd = Us[:,:nSvd]
        self.KtotUsSvd = UsSvd
        self.KtotSvd = UsSvd.T.dot(self.KtotU)
        
    def makeVarLinModel(self,stdLim = [0.90,0.95,0.98,0.99,0.995,0.999]):
        # run LM.busViolationVar(Sgm) before running this
        vars = self.varKtotU.copy()
        
        varSortN = vars.argsort()[::-1]
        stds = np.sqrt(vars)
        stds = stds[varSortN]
        
        stds = stds/stds[0]
        
        NSetStd = []
        N0 = len(stds)
        
        for stdlim in stdLim:
            NSetStd = NSetStd + [np.argmin(abs(stds - (1-stdlim)))]
        print('\nStdLim:',stdLim)
        print('NSetStd:',NSetStd,', out of ', N0)
        self.NSetStd = NSetStd
        
    def makeCorrModel(self,stdLim=0.70,corrLim=[0.95]): # as chosen running linSvdCalcs_run.py.
        # self.getCovMat(getFixCov=False,getTotCov=False,getFullCov=True)
        vars = self.varKfullU.copy()
        
        varSortN = vars.argsort()[::-1]
        
        stds = np.sqrt(vars)
        stds = stds[varSortN]
        stdNorm = stds/stds[0]
        
        # first find which buses should be in. from https://stackoverflow.com/questions/9868653/find-first-sequence-item-that-matches-a-criterion
        nOut = next(n for n in range(len(stdNorm)) if stdNorm[n]<(1-stdLim))
        stdOut = stds[:nOut]
        varSortNout = varSortN[:nOut]
        
        # KfullU = np.concatenate((self.KtotU,self.KfixU),axis=0)
        # self.KfullUcov = KfullU.dot(KfullU.T)
        # covScaling = np.sqrt(np.diag(self.KfullUcov))
        # covScaling[covScaling==0] = np.inf # avoid divide by zero complaint
        # self.KfullUcorr = dsf.vmvM(1/covScaling,self.KfullUcov,1/covScaling)
        
        # corr = abs(self.KfullUcorr)
        # corr = corr - np.diag(np.ones(len(corr)))
        # corr = corr[varSortN][:,varSortN]
        
        KfullUoutNorm = dsf.vmM(1/stdOut,(np.concatenate((self.KtotU,self.KfixU),axis=0)[varSortNout]))
        corr = abs(KfullUoutNorm.dot(KfullUoutNorm.T))
        for i in range(len(corr)):
            corr[i,i] = 0
        # print(corr[0])
        
        NsetLen = []
        Nset = []
        for corrlim in corrLim:
            nset = [0]
            i=1
            # while (stds[i] > (1-stdLim)) and i<(len(stds)-1):
            while i<(len(stdOut)-1):
                maxCorr = max(corr[i,nset])
                if maxCorr < corrlim:
                    nset = nset + [i]
                i+=1
            Nset = Nset + [nset]
            NsetLen = NsetLen + [len(nset)]
        print('\nCorr model stdLim/corrLim:',stdLim,'/',corrLim)
        print('Corr model nset:',NsetLen,' out of ',len(vars))
        
        self.NSetCor = Nset
        # self.getCovMat(getFixCov=False,getTotCov=False,getFullCov=True)
        # vars = self.varKtotU.copy()
        
        # varSortN = vars.argsort()[::-1]

        # stds = np.sqrt(vars)
        # stds = stds[varSortN]
        
        
        
        # stds = stds/stds[0]

        # corr = abs(self.KtotUcorr)
        # corr = corr - np.diag(np.ones(len(corr)))

        # corr = corr[varSortN][:,varSortN]
        
        # NsetLen = []
        # Nset = []
        # for corrlim in corrLim:
            # nset = [0]
            # i=1
            # while (stds[i] > (1-stdLim)) and i<(len(stds)-1):
                # maxCorr = max(corr[i,nset])
                # if maxCorr < corrlim:
                    # nset = nset + [i]
                # i+=1
            # Nset = Nset + [nset]
            # NsetLen = NsetLen + [len(nset)]
        # print('\nCorr model stdLim/corrLim:',stdLim,'/',corrLim)
        # print('Corr model nset:',NsetLen)
        # self.NSetCor = NsetFix
        
        # # NOTE: THIS is literally a copy of the above code with 'fix' changing 'tot'.
        # varsFix = self.varKfixU.copy()
        
        # varSortNfix = varsFix.argsort()[::-1]

        # stds = np.sqrt(varsFix)
        # stds = stds[varSortNfix]

        # stds = stds/stds[0]

        # corr = abs(self.KfixUcorr)
        # corr = corr - np.diag(np.ones(len(corr)))

        # corr = corr[varSortNfix][:,varSortNfix]
        
        # NsetLen = []
        # NsetFix = []
        # for corrlim in corrLim:
            # nset = [0]
            # i=1
            # while (stds[i] > (1-stdLim)) and i<(len(stds)-1):
                # maxCorr = max(corr[i,nset])
                # if maxCorr < corrlim:
                    # nset = nset + [i]
                # i+=1
            # NsetFix = NsetFix + [nset]
            # NsetLen = NsetLen + [len(nset)]
        # print('\nCorr model stdLim/corrLim (fix):',stdLim,'/',corrLim)
        # print('Corr model nset (fix):',NsetLen)
        # self.NSetFixCor = NsetFix
        
        
    def updateDcpleModel(self,regVreg):
        # this only requires the update of b0.
        bV = lmKronRedVregUpdate(self.LMfxd,self.idxShf,regVreg)
        try:
            A = self.A0
        except:
            stt = self.WD+'\\lin_models\\'+self.feeder+'\\fxd_model\\'+self.feeder+'Lpt'+'Fxd'
            end = str(np.round(self.linPoint*100).astype(int)).zfill(3)+'.npy'
            A = np.load(stt+'A'+end)
            self.A0 = A
        # NB based on self.updateTotModel(A,Bkron)
        # self.b0ls = (A.dot(self.xNomTot*self.loadPointLo) + bV)/self.vTotBase # in pu
        # self.b0hs = (A.dot(self.xNomTot*self.loadPointHi) + bV)/self.vTotBase # in pu
        self.b0ls = (A.dot(self.xTotLs) + bV)/self.vTotBase # in pu
        self.b0hs = (A.dot(self.xTotHs) + bV)/self.vTotBase # in pu
        
        self.updateMvLvB0Models()
        
        # Akron, Bkron = lmKronRed(self.LMfxd,self.idxShf,regVreg)
        # self.updateTotModel(Akron,Bkron)
        
    
    def updateTotModel(self,A,bV):
        KyP = A[:,:len(self.xhyNtot)//2] # these might be zero if there is no injection (e.g. only Q)
        KyQ = A[:,len(self.xhyNtot)//2:len(self.xhyNtot)]
        KdP = A[:,len(self.xhyNtot):len(self.xhyNtot) + (len(self.xhdNtot)//2)]
        KdQ = A[:,len(self.xhyNtot) + (len(self.xhdNtot)//2)::]
        
        # self.b0ls = (A.dot(self.xNomTot*self.loadPointLo) + bV)/self.vTotBase # in pu
        # self.b0hs = (A.dot(self.xNomTot*self.loadPointHi) + bV)/self.vTotBase # in pu
        self.b0ls = (A.dot(self.xTotLs) + bV)/self.vTotBase # in pu
        self.b0hs = (A.dot(self.xTotHs) + bV)/self.vTotBase # in pu
        
        # # Check these match up with explicit tap cals.
        # b0 = self.KfixPuShf.dot(self.xTotLs) + self.bVPuShf
        # tSet = np.linalg.solve( self.KtPuShf[-self.nT:], self.regVregPu - b0[-self.nT:] )
        # vOut = self.KfixPuShf.dot(self.xTotLs) + self.KtPuShf.dot(tSet) + self.bVPuShf
        # self.vOutTapSet = vOut
        
        self.b0ls[self.b0ls<0.5] = 1.0 # get rid of outliers
        self.b0hs[self.b0hs<0.5] = 1.0 # get rid of outliers
        
        k_Q = pf2kq(self.QgenPf)
        Ktot = np.concatenate((KyP + k_Q*KyQ,KdP + k_Q*KdQ),axis=1)   
        
        self.KtotPu = dsf.vmM(1/self.vTotBase,Ktot) # scale to be in pu per W
        
        self.KtotPuMv = self.KtotPu[self.mvIdxTot]
        self.KtotPuLv = self.KtotPu[self.lvIdxTot]
        self.updateMvLvB0Models()
        
    def updateMvLvB0Models(self):
        self.b0lsMv = self.b0ls[self.mvIdxTot]
        self.b0hsMv = self.b0hs[self.mvIdxTot]
        self.b0lsLv = self.b0ls[self.lvIdxTot]
        self.b0hsLv = self.b0hs[self.lvIdxTot]
        
        self.b0MvMax = np.maximum(self.b0lsMv,self.b0hsMv)
        self.b0MvMin = np.minimum(self.b0lsMv,self.b0hsMv)
        self.b0LvMax = np.maximum(self.b0lsLv,self.b0hsLv)
        self.b0LvMin = np.minimum(self.b0lsLv,self.b0hsLv) 
        
    def calcLinPdfError(self,otherRslt,type='MAE',model='lin'):
        if model=='lin':
            Vp0 = 0.01*self.linHcRsl['Vp_pct']
        if model=='dss':
            Vp0 = 0.01*self.dssHcRsl['Vp_pct']
        
        Vp1 = 0.01*otherRslt['Vp_pct']
        if type=='reg':
            Error = np.linalg.norm(Vp0-Vp1,ord=1)/(np.linalg.norm(Vp1,ord=1)+1) # this equivalent to: sum of errors/(regularised sum of function)
        elif type=='MAE':
            Error = 100*np.mean(np.abs(Vp1 - Vp0)) # in %
        return Error
    
    # --------------------------------- PLOTTING FUNCTIONS FROM HERE
    def corrPlot(self):
        # vars = self.varKtotU.copy()
        vars = self.varKfullU.copy()
        varSortN = vars.argsort()[::-1]
        
        # corrLogAbs = np.log10(abs((1-self.KtotUcorr)) + np.diag(np.ones(len(self.KtotPu))) +1e-14 )
        corrLogAbs = np.log10(abs((1-self.KfullUcorr)) + np.diag(np.ones(len(self.KfullUcorr))) +1e-14 )
        corrLogAbs = corrLogAbs[varSortN][:,varSortN]

        plt.spy(corrLogAbs<-1.0,color=cm.viridis(0.),markersize=1,marker='.') # 90%
        plt.spy(corrLogAbs<-1.3,color=cm.viridis(0.33),markersize=1,marker='.') # 95%
        plt.spy(corrLogAbs<-1.7,color=cm.viridis(0.66),markersize=1,marker='.') # 98%
        plt.spy(corrLogAbs<-2.0,color=cm.viridis(0.99),markersize=1,marker='.') # 99%
        # plt.spy(asd,color=cm.viridis(0.5),markersize=0.6,marker='.')
        # plt.spy(qwe,color=cm.viridis(1.),markersize=0.4,marker='.')
        plt
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def getBusPhs(self):
        vYZ = self.vTotYNodeOrder
        sYZ = np.concatenate((self.SyYNodeOrderTot,self.SdYNodeOrderTot))

        bus0v = []; bus0s = []
        phs0v = []; phs0s = []
        for yz in vYZ:
            fullBus = yz.split('.')
            bus0v = bus0v+[fullBus[0].lower()]
            if len(fullBus)>1:
                phs0v = phs0v+[fullBus[1::]]
            else:
                phs0v = phs0v+[['1','2','3']]
        for yz in sYZ:
            fullBus = yz.split('.')
            bus0s = bus0s+[fullBus[0].lower()]
            if len(fullBus)>1:
                phs0s = phs0s+[fullBus[1::]]
            else:
                phs0s = phs0s+[['1','2','3']]
        
        self.bus0v = np.array(bus0v)
        self.phs0v = np.array(phs0v)
        self.bus0s = np.array(bus0s)
        self.phs0s = np.array(phs0s)
    
    def plotBranches(self,ax,scores=None):
        # branchCoords = self.branchCoords
        branches = self.branches
        busCoords = self.busCoords
        print('Plotting branches...')
        segments = []
        for branch,buses in branches.items():
            bus1 = buses[0].split('.')[0]
            bus2 = buses[1].split('.')[0]
            segments = segments + [[busCoords[bus1],busCoords[bus2]]]
            # if branch.split('.')[0]=='Transformer':
                # ax.plot(points0[-1],points1[-1],'--',Color='#777777')
        if scores==None:    
            coll = LineCollection(segments, Color='#cccccc')
        else:
            coll = LineCollection(segments, cmap=plt.cm.viridis)
            coll.set_array(scores)
        ax.add_collection(coll)
        ax.autoscale_view()
        self.segments = segments
        
    def plotBuses(self,ax,scores,minMax,colorInvert=False,modMarkerSze=False,cmap=plt.cm.viridis):
        busCoords = self.busCoords
        print('Plotting buses...')
        x0scr = []
        y0scr = []
        xyClr = []
        x0nne = []
        y0nne = []
        mrkSze = []
        for bus,coord in busCoords.items():
            if not np.isnan(busCoords[bus][0]):
                if np.isnan(scores[bus]):
                    x0nne = x0nne + [coord[0]]
                    y0nne = y0nne + [coord[1]]
                else:
                    x0scr = x0scr + [coord[0]]
                    y0scr = y0scr + [coord[1]]
                    if minMax==None:
                        score=scores[bus]
                    else:
                        score = (scores[bus]-minMax[0])/(minMax[1]-minMax[0])
                    if colorInvert:
                        xyClr = xyClr + [cmap(1-score)]
                        if modMarkerSze:
                            mrkSze.append(150.0*(1-score))
                    else:
                        xyClr = xyClr + [cmap(score)]
                        if modMarkerSze:
                            mrkSze.append(150.0*score)
                    if not modMarkerSze:
                        mrkSze.append(20.0)
        
        plt.scatter(x0scr,y0scr,Color=xyClr,marker='.',zorder=+10,s=mrkSze,alpha=0.8)
        plt.scatter(x0nne,y0nne,Color='#cccccc',marker='.',zorder=+5,s=20/np.sqrt(2))
    
    def plotRegs(self,ax):
        if self.nRegs>0:
            regBuses = self.vTotYNodeOrder[-self.nRegs:]; i=0
            for regBus in regBuses:
                regCoord = self.busCoords[regBus.split('.')[0].lower()]
                if not np.isnan(regCoord[0]):
                    ax.plot(regCoord[0],regCoord[1],'r',marker=(6,1,0),zorder=+15)
                    # ax.annotate(str(i),(regCoord[0],regCoord[1]),zorder=+40)
                else:
                    print('Could not plot regulator bus'+regBus+', no coordinate')
                i+=1
        else:
            print('No regulators to plot.')
    
    def plotSub(self,ax,pltSrcReg=True):
        srcCoord = self.busCoords[self.vSrcBus]
        if not np.isnan(srcCoord[0]):
            ax.plot(srcCoord[0],srcCoord[1],'k',marker='H',markersize=8,zorder=+20)
            if self.srcReg and pltSrcReg:
                ax.plot(srcCoord[0],srcCoord[1],'r',marker='H',markersize=3,zorder=+21)
            else:
                ax.plot(srcCoord[0],srcCoord[1],'w',marker='H',markersize=3,zorder=+21)
        else:
            print('Could not plot source bus'+self.vSrcBus+', no coordinate')
        
        
    def getSetVals(self,Set,type='mean',busType='v'):
        busCoords = self.busCoords
        if busType=='v':
            bus0 = self.bus0v
            # phs0 = self.phs0v
        elif busType=='s':
            bus0 = self.bus0s
            # phs0 = self.phs0s
        
        setVals = {}
        setMin = 1e100
        setMax = -1e100
        
        for bus in busCoords:
            if not np.isnan(busCoords[bus][0]):
                vals = Set[bus0==bus.lower()]
                vals = vals[~np.isnan(vals)]
                # phses = phs0v[bus0v==bus.lower()].flatten()
                if not len(vals):
                    setVals[bus] = np.nan
                else:
                    if type=='mean':    
                        setVals[bus] = np.mean(vals)
                        setMax = max(setMax,np.mean(vals))
                        setMin = min(setMin,np.mean(vals))
                    elif type=='max':
                        setVals[bus] = np.max(vals)
                        setMax = max(setMax,np.max(vals))
                        setMin = min(setMin,np.max(vals))
                    elif type=='min':
                        setVals[bus] = np.min(vals)
                        setMax = max(setMax,np.min(vals))
                        setMin = min(setMin,np.min(vals))
            else:
                setVals[bus] = np.nan
        if setMin==setMax:
            setMinMax=None
        else:
            setMinMax = [setMin,setMax]
        return setVals, setMinMax
        
    def ccColorbar(self,ax,minMax,roundNo=2,units='',loc='NorthEast',colorInvert=False,cmap=plt.cm.viridis):
        xlm = ax.get_xlim()
        ylm = ax.get_ylim()
        dx = np.diff(xlm)
        dy = np.diff(ylm)
        if loc=='NorthEast':
            top = ylm[1] - dy*0.025
            btm = ylm[1] - dy*0.2
            xcrd = xlm[1] - dx*0.2
        elif loc=='NorthWest':
            top = ylm[1]
            btm = ylm[1] - dy*0.25
            xcrd = xlm[1] - dx*0.9
        elif loc=='SouthEast':
            top = ylm[1] - dy*0.75
            btm = ylm[0] 
            xcrd = xlm[1] - dx*0.25
        elif loc=='resPlot24': # used in linSvdCalcsRslts.py
            top = ylm[1] - dy*0.15
            btm = ylm[1] - dy*0.4
            xcrd = xlm[1] - dx*0.2

        for i in range(100):
            y1 = btm+0.96*(top-btm)*(i/100)
            y2 = btm+0.96*(top-btm)*((i+1)/100)
            ax.plot([xcrd,xcrd],[y1,y2],lw=6,c=cmap(i/100))
            
        if colorInvert:
            tcks = [str(round(minMax[1],roundNo)),str(round(np.mean(minMax),roundNo)),str(round(minMax[0],roundNo))]
        else:
            tcks = [str(round(minMax[0],roundNo)),str(round(np.mean(minMax),roundNo)),str(round(minMax[1],roundNo))]
        
        for i in range(3):
            y_ = btm+(top-btm)*(i/2)-((top-btm)*0.075)
            ax.annotate('  '+tcks[i]+units,(xcrd+dx*0.02,y_),fontsize='small')
        if loc=='resPlot24':
            t = ax.annotate('# St. Dev.',(xcrd-dx*0.065,top + (top-btm)*0.145),fontsize='small')
            width_scale = 0.2 # epri K1
            width_scale = 0.285 # epri 24
            btmm_y = btm - 0.165*(top-btm)
            xcrd_0 = xcrd - 0.09*dx
            hghtScale = 1.49
            box = patches.Rectangle(xy=(xcrd_0,btmm_y),width=dx*width_scale,height=(top-btm)*hghtScale,edgecolor='k',facecolor='w',zorder=-10)
            ax.add_patch(box)
            
        
    def plotNetBuses(self,type,regsOn=True,pltShow=True,minMax=None,pltType='mean',varMax=10,cmap=plt.cm.viridis):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.getBusPhs()

        self.plotBranches(ax)
        if type=='vLo':
            scoreNom = self.b0ls
            colorInvert = False
        elif type=='vHi':
            scoreNom = self.b0hs
            colorInvert = False
        elif type=='logVar':
            if self.nRegs > 0:
                scoreNom = np.log10(self.varKtotU + min(self.varKtotU[:-self.nRegs]))
            else:
                scoreNom = np.log10(self.varKtotU)
            # scoreNom[(scoreNom - np.mean(scoreNom))/np.std(scoreNom) < -3] = np.nan
            colorInvert = False
        elif type=='nStd':
            scoreNom = self.nStdU
            scoreNom[scoreNom>varMax] = np.nan
            colorInvert = True
        scores, minMax0 = self.getSetVals(scoreNom,pltType)

        if minMax!=None:
            minMax0 = minMax
        
        self.plotBuses(ax,scores,minMax0,colorInvert=colorInvert,cmap=cmap)
        self.plotSub(ax)
        
        
        self.plotRegs(ax)
        ax.axis('off')
        # plt.title(self.feeder)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if type=='vLo' or type=='vHi':
            self.ccColorbar(ax,minMax0,loc=self.legLoc,units=' pu',roundNo=3,colorInvert=colorInvert,cmap=cmap)
        elif type=='logVar' or type=='nStd':
            # if self.legLog!=None:
            if self.legLoc!=None:
                self.ccColorbar(ax,minMax0,loc=self.legLoc,colorInvert=colorInvert,cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        print('Complete')
        if pltShow:
            plt.show()
        else:
            self.currentAx = ax
    def plotNetwork(self,pltShow=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.getBusPhs()
        self.plotBranches(ax)
        
        scoreNom = np.ones((self.nS))
        scores, minMax0 = self.getSetVals(scoreNom,busType='s')
        self.plotBuses(ax,scores,minMax0,modMarkerSze=False,cmap=cm.Blues)
        self.plotSub(ax,pltSrcReg=False)

        xlm = ax.get_xlim() 
        ylm = ax.get_xlim()
        dx = xlm[1] - xlm[0]; dy = ylm[1] - ylm[0] # these seem to be in feet for k1
        
        srcCoord = self.busCoords[self.vSrcBus]
        ax.annotate('Substation',(srcCoord[0]+0.01*dx,srcCoord[1]+0.01*dy))
        ax.axis('off')
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(ticks=[],labels=[])
        plt.yticks(ticks=[],labels=[])
        plt.tight_layout()

        if pltShow:
            plt.show()
        return ax


# =================================== CLASS: hcPdfs
class hcPdfs:
    def __init__(self,feeder,WD=None,netModel=0,dMu=None,pdfName=None,prms=np.array([]),clfnSolar=None,nMc=100,rndSeed=0):
        self.rndSeed=rndSeed # 32 bit unsigned integer (up to 2**32)
        if pdfName==None or pdfName=='gammaWght' or pdfName=='gammaFlat':
            if dMu==None:
                dMu = 0.02
            if netModel==0:
                circuitK = {'eulv':1.8,'usLv':5.0,'13bus':4.8,'34bus':5.4,'123bus':3.0,'8500node':1.2,'epri5':2.4,'epri7':2.0,'epriJ1':1.2,'epriK1':1.2,'epriM1':1.5,'epri24':1.5}
            elif netModel==1: # LDC model
                circuitK = {'13bus':6.0,'34bus':8.0,'123bus':3.6}
            elif netModel==2: # decoupled model
                circuitK = {'8500node':2.5,'epriJ1':6.0,'epriK1':1.5,'epriM1':1.8,'epri24':1.5}
            
            self.dMu = dMu
            mu_k = circuitK[feeder]*np.arange(dMu,1.0,dMu) # NB this is as a PERCENTAGE of the chosen nominal powers.
        else:
            if dMu==None: # ! NOTE THAT THIS NEEDS SETTING to avoid preloading the expanded values.
                if WD==None:
                    print('Working directory needs to be passed to __init__ of hcPdfs for this mode.')
                else:
                    SD = os.path.join(WD,'hcResults','th_kW_mult.pkl')
                    with open(SD,'rb') as handle:
                        circuitK = pickle.load(handle)
                self.dMu = circuitK[feeder]
            else:
                self.dMu = 1.0
            mu_k = np.array([self.dMu])
        
        if clfnSolar==None:
            clfnSolar = {'k':4.21423544,'th_kW':1.2306995} # from plot_california_pv.py
        
        if pdfName==None:
            pdfName = 'gammaWght'
            prms = np.array([clfnSolar['k']]) # if none, initialised with clfnSolar shape parameter value
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k)),'clfnSolar':None,'nMc':nMc}
        elif pdfName=='gammaWght':
            # parameters: np.array([k0,k1,...])
            if len(prms)==0:
                prms = np.array([clfnSolar['k']])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k)),'clfnSolar':None,'nMc':nMc}
        elif pdfName=='gammaFlat':
            # parameters: np.array([k0,k1,...])
            if len(prms)==0:
                prms = np.array([clfnSolar['k']])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k)),'clfnSolar':None,'nMc':nMc}
        elif pdfName=='gammaFrac':
            # parameters: np.array([frac0,frac1,...])
            if len(prms)==0:
                prms=np.arange(0.02,1.02,0.02) # if none, use values from santoso paper
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k)),'clfnSolar':clfnSolar,'nMc':nMc}
        elif pdfName=='gammaXoff':
            # parameters: np.array([[frac0,xOff0],[frac1,xOff1],...])
            if len(prms)==0:
                prms=np.array([[0.50,8]])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k)),'clfnSolar':clfnSolar,'nMc':nMc}
        
    def halfLoadMean(self,scale,xhyN,xhdN):
        # scale suggested as: LM.scaleNom = lp0data['kLo'] - lp0data['k']
        
        roundI = 1e0
        Mu0_y = -scale*roundI*np.round(xhyN[:xhyN.shape[0]//2]/roundI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
        Mu0_d = -scale*roundI*np.round(xhdN[:xhdN.shape[0]//2]/roundI - 1e6*np.finfo(np.float64).eps)
        Mu0 = np.concatenate((Mu0_y,Mu0_d))
        
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        return Mu0
        
    
    def getMuStd(self,LM=None,prmI=None):
        pdfName = self.pdf['name']
        if pdfName=='gammaWght':
            if LM==None:
                print('\nError: no linear model loaded into ---getMuStd--- for model gammaWght\n')
            else:
                Mu = self.halfLoadMean(LM.loadScaleNom,LM.xhyNtot,LM.xhdNtot) # in W
                Sgm = Mu/np.sqrt(self.pdf['prms'][0]) # in W
        if pdfName=='gammaFrac':
            if prmI==None:
                print('\nError: no prmI loaded into ---getMuStd--- for model gammaFrac\n')
            else:
                frac = self.pdf['prms'][prmI] # fraction of load this is zero (as part of mixture)
                k,th = self.pdf['clfnSolar'].values()
                Mu0 = k*th*1e3
                Sgm0 = np.sqrt(k)*th*1e3
                Mu = np.array([frac*Mu0])
                Sgm = np.sqrt(frac*(Mu0**2 + Sgm0**2) - Mu**2) # see FrÃ¼hwirth-Schnatter chapter 1
                
                var = Sgm**2
                covar = (Mu0**2)*frac*(frac-1)/(LM.nS - 1)
                
                LM.Cov = [var,covar] # only represent the diagonal and off-diagonal elements
                LM.CovSqrt = calcDp1rSqrt(var,covar,LM.nS) # NEW! :D
                
        return Mu,Sgm # both in WATTS
        
    def genPdfMcSet(self,nMc,Mu0,prmI,getMcU=False):
        # NB: Mu0 used in gammaFrac/gammaXoff
        np.random.seed(self.rndSeed)
        seed(self.rndSeed) # used in the gammaFrac draw
        if self.pdf['name']=='gammaWght':
            k = self.pdf['prms'][prmI]
            pdfMc0 = np.random.gamma(k,1/np.sqrt(k),(len(Mu0),nMc))
            pdfMc = dsf.vmM( 1e-3*Mu0/np.sqrt(k),pdfMc0 ) # in kW
            if getMcU:
                pdfMcU = pdfMc0 - np.sqrt(k) # zero mean, unit variance
            
        elif self.pdf['name']=='gammaFlat':
            k = self.pdf['prms'][prmI]
            Mu0mean = Mu0.mean()
            # pdfMc0 = np.random.gamma(shape=k,scale=1/np.sqrt(k),size=(len(Mu0),nMc))
            # pdfMc = (1e-3*Mu0mean/np.sqrt(k))*pdfMc0
            pdfMc0 = np.random.gamma(shape=k,scale=(1e-3*Mu0mean/np.sqrt(k))/np.sqrt(k),size=(len(Mu0),nMc))
            if getMcU:
                # pdfMcU = pdfMc0 - np.sqrt(k) # zero mean, unit variance
                pdfMcU = (pdfMc0/(1e-3*Mu0mean/np.sqrt(k))) - np.sqrt(k) # zero mean, unit variance
        
        elif self.pdf['name']=='gammaFrac':
            clfnSolar = self.pdf['clfnSolar']
            frac = self.pdf['prms'][prmI]
            
            genIn = np.zeros((len(Mu0),nMc))
            nDraw = np.ceil(frac*len(Mu0)).astype(int)
            # rangeMu0 = np.arange(len(Mu0),dtype=int)
            for i in range(nMc):
                # idxs = shuffle(rangeMu0)
                # idxs = np.random.choice(rangeMu0,nDraw,replace=False)
                idxs = sample(range(len(Mu0)),nDraw)
                genIn[idxs,i] = 1
                
            # genIn = np.random.binomial(1,frac,(len(Mu0),nMc))
            
            pdfGen = np.random.gamma(shape=clfnSolar['k'],scale=clfnSolar['th_kW'],size=(len(Mu0),nMc))
            pdfMc = pdfGen*genIn
            pdfMeans = np.mean(pdfMc) # NB these are uniformly distributed
            pdfStd = np.std(pdfMc) # NB these are uniformly distributed
            if getMcU:
                pdfMcU = (pdfMc - pdfMeans)/pdfStd
        elif self.pdf['name']=='gammaXoff':
            clfnSolar = self.clfnSolar
            
            frac = self.pdf['prms'][prmI][0]
            xOff = self.pdf['prms'][prmI][1]
            
            genIn = np.random.binomial(1,frac,(len(Mu0),nMc))
            pdfGen = np.random.gamma(shape=clfnSolar['k'],scale=clfnSolar['th_kW'],size=(len(Mu0),nMc))
            pdfGen = np.minimum(pdfGen,xOff*np.ones(pdfGen.shape))
            pdfMc = pdfGen*genIn
            pdfMeans = np.mean(pdfMc) # NB these are uniformly distributed
            pdfStd = np.std(pdfMc) # NB these are uniformly distributed
            if getMcU:
                pdfMcU = (pdfMc - pdfMeans)/pdfStd
        if not getMcU:
            pdfMcU = []
        return pdfMc, pdfMcU
        
        