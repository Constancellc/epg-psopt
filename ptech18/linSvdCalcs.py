import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
from dss_voltage_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD



def cnsBdsCalc(vLsMv,vLsLv,vHsMv,vHsLv,vDv,lp0data,DVmax=0.06):
    nMc = vLsMv.shape[0]
    
    vLsMv[vLsMv<0.5] = 1.0
    vLsLv[vLsLv<0.5] = 1.0
    vHsMv[vHsMv<0.5] = 1.0
    vHsLv[vHsLv<0.5] = 1.0
    
    VpMv = lp0data['VpMv']
    VmMv = lp0data['VmMv']
    VpLv = lp0data['VpLv']
    VmLv = lp0data['VmLv']
    
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


def calcVar(X):
    i=0
    var = np.zeros(len(X))
    for x in X:
        var[i] = x.dot(x)
        i+=1
    return var
    
# =================================== PLOTTING FUNCTIONS

def plotCns(mu_k,Cns_pct,ax=None,pltShow=True,feeder=None,lineStyle='-'):
    if ax==None:
        fig, ax = plt.subplots()
        clrs = cm.nipy_spectral(np.linspace(0,1,9))
        ax.set_prop_cycle(color=clrs)
    # plt.plot(pdfData['mu_k'],Cns_pct_lin[0],'--')
    ax.plot(mu_k,Cns_pct[0],lineStyle)
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

def plotHcVltn(mu_k,Vp_pct,ax=None,pltShow=True,feeder=None,lineStyle='.-',logScale=True):
    if logScale:
        ax.semilogy(mu_k,Vp_pct,lineStyle)
        ax.set_title('Prob. of violation (logscale), '+feeder);
        ax.set_ylabel('Log [ P(.), % ]')
    else:
        ax.plot(mu_k,Vp_pct,lineStyle)
        ax.set_title('Prob. of violation '+feeder);
        ax.set_ylabel('P(.), %')
    ax.set_xlabel('Scale factor');
    ax.grid(True)
    

    
# =================================== CLASS: linModel
class linModel:
    """Linear model class with a whole bunch of useful things that we can do with it."""
    
    def __init__(self,fdr_i,WD):
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
        fdrNetModels = [0,0,0,0,0,1,1,0,1,2,-1,-1,-1,-1,0,-1,-1,0,0,2,2,2,2]
        
        feeder = fdrs[fdr_i]
        
        self.feeder = feeder
        self.WD = WD # for debugging
        
        self.netModelNom = fdrNetModels[fdr_i]
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
            lp0data = pickle.load(handle)
        
        self.linPoint = lp0data['k']
        self.loadPointLo = lp0data['kLo']
        self.loadPointHi = lp0data['kHi']
        self.loadScaleNom = lp0data['kLo'] - lp0data['k']
        self.VpMv = lp0data['VpMv']
        self.VmMv = lp0data['VmMv']
        self.VpLv = lp0data['VpLv']
        self.VmLv = lp0data['VmLv']
        self.nRegs = lp0data['nRegs']
        self.vSrcBus = lp0data['vSrcBus']
        self.srcReg = lp0data['srcReg']
        self.legLoc = lp0data['legLoc']
        self.DVmax = 0.06 # pu
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','busCoords.pkl'),'rb') as handle:
            self.busCoords = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','branches.pkl'),'rb') as handle:
            self.branches = pickle.load(handle)
        
        # load the fixed model, as this always exists
        LMfxd = loadLinMagModel(self.feeder,self.linPoint,WD,'Lpt',regModel=False)
        Kyfix=LMfxd['Ky'];Kdfix=LMfxd['Kd']
        dvBase = LMfxd['vKvbase'] # NB: this is different to vBase for ltc/regulator models!

        KyPfix = Kyfix[:,:Kyfix.shape[1]//2]
        KdPfix = Kdfix[:,:Kdfix.shape[1]//2]
        Kfix = np.concatenate((KyPfix,KdPfix),axis=1)
        KfixCheck = np.sum(Kfix==0,axis=1)!=Kfix.shape[1] # [can't remember what this is for...]
        Kfix = Kfix[KfixCheck]
        dvBase = dvBase[KfixCheck]
        
        self.LMfxd = {'Ky':Kyfix,'Kd':Kdfix,'bV':LMfxd['bV'],'Kt':LMfxd['Kt']}
        
        self.dvBase = dvBase
        self.KfixPu = dsf.vmM(1/dvBase,Kfix)
        self.vFixYNodeOrder = LMfxd['vYNodeOrder']
        self.v_idx_fix = LMfxd['v_idx']
    
    def loadNetModel(self,netModel=None):
        if netModel==None:
            netModel = self.netModelNom
        
        if netModel==0:
            # IF using the FIXED model:
            LM = loadLinMagModel(self.feeder,self.linPoint,self.WD,'Lpt',regModel=False)
            Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV']
            self.vTotBase = LM['vKvbase']
        else:
            # IF using the LTC of DCP model:
            LM = loadNetModel(self.feeder,self.linPoint,self.WD,'Lpt',netModel)
            A=LM['A'];bV=LM['B']
            self.vTotBase = LM['Vbase']
            
        xhy0=LM['xhy0'];xhd0=LM['xhd0']
        self.xhyNtot = xhy0/self.linPoint
        self.xhdNtot = xhd0/self.linPoint
        self.xNomTot = np.concatenate((self.xhyNtot,self.xhdNtot))
        self.vTotYNodeOrder = LM['vYNodeOrder']
        
        if not netModel: # needed seperately because of Y/D concatenation of powers.
            self.b0ls = (Ky.dot(self.xhyNtot*self.loadPointLo) + Kd.dot(self.xhdNtot*self.loadPointLo) + bV)/self.vTotBase # in pu
            self.b0hs = (Ky.dot(self.xhyNtot*self.loadPointHi) + Kd.dot(self.xhdNtot*self.loadPointHi) + bV)/self.vTotBase # in pu

            KyP = Ky[:,:Ky.shape[1]//2]
            KdP = Kd[:,:Kd.shape[1]//2]
        elif netModel>0:
            self.updateTotModel(A,bV)
            # self.b0ls = (A.dot(self.xNomTot*self.loadPointLo) + bV)/self.vTotBase # in pu
            # self.b0hs = (A.dot(self.xNomTot*self.loadPointHi) + bV)/self.vTotBase # in pu
            
            # KyP = A[:,0:len(xhy0)//2] # these might be zero if there is no injection (e.g. only Q)
            # KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
            
        # Ktot = np.concatenate((KyP,KdP),axis=1)
        # self.KtotPu = dsf.vmM(1/self.vTotBase,Ktot) # scale to be in pu
        
        
        if netModel==2: # decoupling regulator model
            self.idxShf = LM['idxShf']
            self.regVreg0 = LM['regVreg']
        
        self.v_idx_tot = LM['v_idx']
        
        self.mvIdx = np.where(self.vTotBase>1000)[0]
        self.lvIdx = np.where(self.vTotBase<=1000)[0]
        
        self.SyYNodeOrderTot = LM['SyYNodeOrder']
        self.SdYNodeOrderTot = LM['SdYNodeOrder']
        
    def getBusPhs(self):
        vYZ = self.vTotYNodeOrder
        
        bus0 = []
        phs0 = []
        for yz in vYZ:
            fullBus = yz.split('.')
            bus0 = bus0+[fullBus[0].lower()]
            if len(fullBus)>1:
                phs0 = phs0+[fullBus[1::]]
            else:
                phs0 = phs0+[['1','2','3']]
        self.bus0 = np.array(bus0)
        self.phs0 = np.array(phs0)
        
    def plotBranches(self,ax,scores=None):
        branches = self.branches
        busCoords = self.busCoords
        print('Plotting branches...')
        for branch in branches:
            bus1 = branches[branch][0].split('.')[0]
            bus2 = branches[branch][1].split('.')[0]
            if branch.split('.')[0]=='Transformer':
                ax.plot((busCoords[bus1][0],busCoords[bus2][0]),(busCoords[bus1][1],busCoords[bus2][1]),'--',Color='#777777')
            else:
                ax.plot((busCoords[bus1][0],busCoords[bus2][0]),(busCoords[bus1][1],busCoords[bus2][1]),Color='#cccccc')
        
    def plotBuses(self,ax,scores,minMax):
        busCoords = self.busCoords
        print('Plotting buses...')
        for bus in busCoords:
            if not np.isnan(busCoords[bus][0]):
                if np.isnan(scores[bus]):
                    ax.plot(busCoords[bus][0],busCoords[bus][1],'.',Color='#cccccc')
                else:
                    score = (scores[bus]-minMax[0])/(minMax[1]-minMax[0])
                    ax.plot(busCoords[bus][0],busCoords[bus][1],'.',Color=cm.viridis(score),zorder=+10)
    
    def plotRegs(self,ax):
        if self.nRegs>0:
            regBuses = self.vTotYNodeOrder[-self.nRegs:]
            for regBus in regBuses:
                regCoord = self.busCoords[regBus.split('.')[0].lower()]
                if not np.isnan(regCoord[0]):
                    # ax.plot(regCoord[0],regCoord[1],'r',marker='o',zorder=+20)
                    ax.plot(regCoord[0],regCoord[1],'r',marker=(6,1,0),zorder=+20)
                else:
                    print('Could not plot regulator bus'+regBus+', no coordinate')
        else:
            print('No regulators to plot.')
    
    def plotSub(self,ax):
        srcCoord = self.busCoords[self.vSrcBus]
        if not np.isnan(srcCoord[0]):
            ax.plot(srcCoord[0],srcCoord[1],'k',marker='H',markersize=8,zorder=+20)
            if self.srcReg:
                ax.plot(srcCoord[0],srcCoord[1],'r',marker='H',markersize=3,zorder=+21)
            else:
                ax.plot(srcCoord[0],srcCoord[1],'w',marker='H',markersize=3,zorder=+21)
        else:
            print('Could not plot source bus'+self.vSrcBus+', no coordinate')
        
        
    def getSetMean(self,Set):
        busCoords = self.busCoords
        phs0 = self.phs0
        bus0 = self.bus0
        
        setMean = {}
        setMin = 1e100
        setMax = -1e100
        
        for bus in busCoords:
            if not np.isnan(busCoords[bus][0]):
                vals = Set[bus0==bus.lower()]
                phses = phs0[bus0==bus.lower()].flatten()
                
                if not len(vals):
                    setMean[bus] = np.nan
                else:
                    setMean[bus] = np.mean(vals)
                    setMax = max(setMax,np.mean(vals))
                    setMin = min(setMin,np.mean(vals))
            else:
                setMean[bus] = np.nan
        
        setMinMax = [setMin,setMax]
        return setMean, setMinMax
        
    def ccColorbar(self,plt,minMax,roundNo=2,units='',loc='NorthEast'):
        xlm = plt.xlim()
        ylm = plt.ylim()
        if loc=='NorthEast':
            top = ylm[1] - np.diff(ylm)*0.025
            btm = ylm[1] - np.diff(ylm)*0.2
            xcrd = xlm[1] - np.diff(xlm)*0.2
        elif loc=='NorthWest':
            top = ylm[1]
            btm = ylm[1] - np.diff(ylm)*0.25
            xcrd = xlm[1] - np.diff(xlm)*0.9
        elif loc=='SouthEast':
            top = ylm[1] - np.diff(ylm)*0.75
            btm = ylm[0] 
            xcrd = xlm[1] - np.diff(xlm)*0.25
            

        for i in range(100):
            y1 = btm+(top-btm)*(i/100)
            y2 = btm+(top-btm)*((i+1)/100)
            plt.plot([xcrd,xcrd],[y1,y2],lw=6,c=cm.viridis(i/100))
        tcks = [str(round(minMax[0],roundNo)),str(round(np.mean(minMax),roundNo)),str(round(minMax[1],roundNo))]
        for i in range(3):
            y_ = btm+(top-btm)*(i/2)-((top-btm)*0.075)
            plt.annotate('  '+tcks[i]+units,(xcrd,y_))
        
    def plotNetBuses(self,type,regsOn=True,pltShow=True,minMax=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.getBusPhs()

        self.plotBranches(ax)
        if type=='vLo':
            setMean, setMeanMinMax = self.getSetMean(self.b0ls)
        elif type=='vHi':
            setMean, setMeanMinMax = self.getSetMean(self.b0hs)
        elif type=='logVar':
            if self.nRegs > 0:
                logVar = np.log10(self.KtotUvar + min(self.KtotUvar[:-self.nRegs]))
            else:
                logVar = np.log10(self.KtotUvar)
            logVar[(logVar - np.mean(logVar))/np.std(logVar) < -3] = np.nan
            setMean, setMeanMinMax = self.getSetMean(logVar)
        
        if minMax!=None:
            setMeanMinMax = minMax
        
        self.plotBuses(ax,setMean,setMeanMinMax)
        self.plotRegs(ax)
        self.plotSub(ax)

        plt.title(self.feeder)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if type=='vLo' or type=='vHi':
            self.ccColorbar(plt,setMeanMinMax,loc=self.legLoc,units=' pu',roundNo=3)
        else:
            self.ccColorbar(plt,setMeanMinMax,loc=self.legLoc)
            
        print('Complete')
        if pltShow:
            plt.show()
        
    
    def runLinHc(self,nMc,pdfData,model='nom'):
        nCnstr = 9
        
        Vp_pct = np.zeros(pdfData['nP'])
        Cns_pct = np.zeros(list(pdfData['nP'])+[nCnstr])
        hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
        hcGenAll = np.array([])
        genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
        genTotAll = {}
        PP = []
        
        nV = self.KtotPu.shape[0]
        nS = self.KtotPu.shape[1]
        
        for i in range(pdfData['nP'][0]):
            pdf = hcPdfs(self.feeder,netModel=self.netModelNom,pdfName=pdfData['name'],prms=pdfData['prms'])
            Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot)
            pdfMc, pdfMcU = pdf.genPdfMcSet(nMc,Mu,i)
            
            genTot0 = np.sum(pdfMc,axis=0)
            genTotSort = genTot0.copy()
            genTotSort.sort()
            genTotAll0 = np.outer(genTot0,pdfData['mu_k'])
            genTotAll[i] = genTotAll0.flatten()
            
            if model=='nom':
                NSet = np.arange(nV)
                
            elif model=='std' or model=='cor':
                vars = self.KtotUvar.copy()
                varSortN = vars.argsort()[::-1]
                if model=='std':
                    NSet = varSortN[0:self.NSetStd[0]]
                elif model=='cor':
                    NSet = varSortN[self.NSetCor[0]]
            # DvMu0 = self.KtotU.dot(np.sqrt(pdfData['prms'])*np.ones(nS))
            
            DelVout = (self.KtotPu[NSet].dot(pdfMc).T)*1e3
            ddVout = abs((self.KfixPu.dot(pdfMc).T)*1e3) # just get abs change
            
            b0ls = self.b0ls[NSet]
            b0hs = self.b0hs[NSet]
                
            for jj in range(pdfData['nP'][-1]):
                genTot = genTot0*pdfData['mu_k'][jj]

                vLsMv = ((DelVout*pdfData['mu_k'][jj]) + b0ls)[:,self.mvIdx]
                vLsLv = ((DelVout*pdfData['mu_k'][jj]) + b0ls)[:,self.lvIdx]
                vHsMv = ((DelVout*pdfData['mu_k'][jj]) + b0hs)[:,self.mvIdx]
                vHsLv = ((DelVout*pdfData['mu_k'][jj]) + b0hs)[:,self.lvIdx]
                vDv = ddVout*pdfData['mu_k'][jj]
                
                lp0data = {}
                lp0data['VpMv'] = self.VpMv
                lp0data['VpLv'] = self.VpLv
                lp0data['VmMv'] = self.VmMv
                lp0data['VmLv'] = self.VmLv
                
                Cns_pct[i,jj], inBounds = cnsBdsCalc(vLsMv,vLsLv,vHsMv,vHsLv,vDv,lp0data)
                
                Vp_pct[i,jj] = 100*sum(inBounds)/nMc
                hcGen = genTot[inBounds]
                
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
        
            binNo = int(np.round(0.5*len(pdf.pdf['mu_k'])))
            hist1 = plt.hist(genTotAll[i],bins=binNo,range=(0,max(genTotAll[i])))
            PP = PP + [hist1[1][1:]]
            plt.close()
        
        self.linHcRsl = {}
        self.linHcRsl['PP'] = PP;
        self.linHcRsl['hcGenSet'] = hcGenSet
        self.linHcRsl['Vp_pct'] = Vp_pct
        self.linHcRsl['Cns_pct'] = Cns_pct
        self.linHcRsl['hcGenAll'] = hcGenAll
        self.linHcRsl['genTotSet'] = genTotSet
    
    def getCovMat(self):
        self.KtotUcov = self.KtotU.dot(self.KtotU.T)
        covScaling = np.sqrt(np.diag(self.KtotUcov))
        covScaling[covScaling==0] = np.inf # avoid divide by zero complaint
        self.KtotUcorr = dsf.vmvM(1/covScaling,self.KtotUcov,1/covScaling)
        
    
    def busViolationVar(self,Sgm,lim='all'):
        if lim=='all':
            limA =   self.VpMv - self.b0ls
            limB =   self.VpLv - self.b0ls
            limC =   -(self.VmMv - self.b0ls)
            limD =   -(self.VmLv - self.b0ls)
            limE =   self.VpMv - self.b0hs
            limF =   self.VpLv - self.b0hs
            limG =   -(self.VmMv - self.b0hs)
            limH =   -(self.VmLv - self.b0hs)
            
            limA[self.lvIdx] = 1.0
            limB[self.mvIdx] = 1.0
            limC[self.lvIdx] = 1.0
            limD[self.mvIdx] = 1.0
            limE[self.lvIdx] = 1.0
            limF[self.mvIdx] = 1.0
            limG[self.lvIdx] = 1.0
            limH[self.mvIdx] = 1.0
            
            # limA0 =   self.Vmax - self.b0ls
            # limB0 =   self.Vmax - self.b0hs
            # limC0 = -(self.Vmin - self.b0ls)
            # limD0 = -(self.Vmin - self.b0hs)
            lim = np.min(np.array([limA,limB,limC,limD,limE,limF,limG,limH]),axis=0)
        
        self.KtotU = dsf.vmvM(lim,self.KtotPu,Sgm)
        self.KtotUvar = calcVar(self.KtotU)
        self.svdLim = lim
    
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
        
    def makeStdModel(self,stdLim = [0.90,0.95,0.98,0.99,0.995,0.999]):
        # run LM.busViolationVar(Sgm) before running this
        vars = self.KtotUvar.copy()
        
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
        # LM.plotNetBuses('logVar') # this is the plot to call to visualize this
        
    def makeCorrModel(self,stdLim=0.99,corrLim=[0.95,0.98,0.99]):
        vars = self.KtotUvar.copy()
        varSortN = vars.argsort()[::-1]

        stds = np.sqrt(vars)
        stds = stds[varSortN]

        stds = stds/stds[0]

        corr = abs(self.KtotUcorr)
        corr = corr - np.diag(np.ones(len(corr)))

        corr = corr[varSortN][:,varSortN]

        
        NsetLen = []
        Nset = []
        for corrlim in corrLim:
            nset = [0]
            i=1
            while stds[i] > (1-stdLim):
                maxCorr = max(corr[i,nset])
                if maxCorr < corrlim:
                    nset = nset + [i]
                i+=1
            Nset = Nset + [nset]
            NsetLen = NsetLen + [len(nset)]
        print('\nCorr model stdLim/corrLim:',stdLim,'/',corrLim)
        print('Corr model nset:',NsetLen)
        self.NSetCor = Nset

    def corrPlot(self):
        vars = self.KtotUvar.copy()
        varSortN = vars.argsort()[::-1]
        
        corrLogAbs = np.log10(abs((1-self.KtotUcorr)) + np.diag(np.ones(len(self.KtotPu))) +1e-14 )
        corrLogAbs = corrLogAbs[varSortN][:,varSortN]

        wer = corrLogAbs<-1.3 # 95%
        asd = corrLogAbs<-1.7 # 98%
        qwe = corrLogAbs<-2. # 99%

        plt.spy(wer,color=cm.viridis(0.),markersize=1,marker='.')
        plt.spy(asd,color=cm.viridis(0.5),markersize=0.6,marker='.')
        plt.spy(qwe,color=cm.viridis(1.),markersize=0.4,marker='.')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def updateDcpleModel(self,regVreg):
        Akron, Bkron = lmKronRed(self.LMfxd,self.idxShf,regVreg)
        self.updateTotModel(Akron,Bkron)
        
    def updateTotModel(self,A,bV):
        self.b0ls = (A.dot(self.xNomTot*self.loadPointLo) + bV)/self.vTotBase # in pu
        self.b0hs = (A.dot(self.xNomTot*self.loadPointHi) + bV)/self.vTotBase # in pu
        
        KyP = A[:,0:len(self.xhyNtot)//2] # these might be zero if there is no injection (e.g. only Q)
        KdP = A[:,len(self.xhyNtot):len(self.xhyNtot) + (len(self.xhdNtot)//2)]
        
        Ktot = np.concatenate((KyP,KdP),axis=1)
        self.KtotPu = dsf.vmM(1/self.vTotBase,Ktot) # scale to be in pu
        
        
class hcPdfs:
    def __init__(self,feeder,netModel=0,dMu=0.01,pdfName=None,prms=np.array([])):
        
        if netModel==0:
            circuitK = {'eulv':1.8,'usLv':5.0,'13bus':4.8,'34bus':5.4,'123bus':3.0,'8500node':1.2,'epri5':2.4,'epri7':2.0,'epriJ1':1.2,'epriK1':1.2,'epriM1':1.5,'epri24':1.5}
        elif netModel==1:
            circuitK = {'13bus':6.0,'34bus':8.0,'123bus':3.6}
        elif netModel==2:
            circuitK = {'8500node':2.5,'epriJ1':6.0,'epriK1':1.5,'epriM1':1.8,'epri24':1.5}
        
        if pdfName==None or pdfName=='gammaWght' or pdfName=='gammaFlat':
            self.dMu = dMu
            mu_k = circuitK[feeder]*np.arange(dMu,1.0,dMu) # NB this is as a PERCENTAGE of the chosen nominal powers.
        else:
            self.dMu=1
            mu_k = circuitK[feeder]*np.array([self.dMu])
        
        self.clfnSolar = {'k':4.21423544,'th_kW':1.2306995} # from plot_california_pv.py
        
        if pdfName==None:
            pdfName = 'gammaWght'
            prms = np.array([3.0])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k))}
        elif pdfName=='gammaWght':
            # parameters: np.array([k0,k1,...])
            if len(prms)==0:
                prms = np.array([3.0])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k))}
        elif pdfName=='gammaFlat':
            # parameters: np.array([k0,k1,...])
            if len(prms)==0:
                prms = np.array([3.0])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k))}
        elif pdfName=='gammaFrac':
            # parameters: np.array([frac0,frac1,...])
            if len(prms)==0:
                prms=np.array([0.50])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k))}
        elif pdfName=='gammaXoff':
            # parameters: np.array([[frac0,xOff0],[frac1,xOff1],...])
            if len(prms)==0:
                prms=np.array([[0.50,8]])
            self.pdf = {'name':pdfName,'prms':prms,'mu_k':mu_k,'nP':(len(prms),len(mu_k))}
    
    def halfLoadMean(self,scale,xhyN,xhdN):
        # scale suggested as: LM.scaleNom = lp0data['kLo'] - lp0data['k']
        
        roundI = 1e0
        Mu0_y = -scale*roundI*np.round(xhyN[:xhyN.shape[0]//2]/roundI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
        Mu0_d = -scale*roundI*np.round(xhdN[:xhdN.shape[0]//2]/roundI - 1e6*np.finfo(np.float64).eps)
        Mu0 = np.concatenate((Mu0_y,Mu0_d))
        
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        return Mu0
        
    def genPdfMcSet(self,nMc,Mu0,prmI):
        np.random.seed(0)
        if self.pdf['name']=='gammaWght':
            k = self.pdf['prms'][prmI]
            pdfMc0 = np.random.gamma(k,1/np.sqrt(k),(len(Mu0),nMc))
            pdfMc = dsf.vmM( 1e-3*Mu0/np.sqrt(k),pdfMc0 )
            pdfMcU = pdfMc0 - np.sqrt(k) # zero mean, unit variance
            
        elif self.pdf['name']=='gammaFlat':
            k = self.pdf['prms'][prmI]
            pdfMc0 = np.random.gamma(k,1/np.sqrt(k),(len(Mu0),nMc))
            pdfMcU = pdfMc0 - np.sqrt(k) # zero mean, unit variance
            Mu0mean = Mu0.mean()
            pdfMc = (1e-3*Mu0mean/np.sqrt(k))*pdfMc0
        
        elif self.pdf['name']=='gammaFrac':
            clfnSolar = self.clfnSolar
            frac = self.pdf['prms'][prmI]
            
            genIn = np.random.binomial(1,frac,(len(Mu0),nMc))
            pdfGen = np.random.gamma(shape=clfnSolar['k'],scale=clfnSolar['th_kW'],size=(len(Mu0),nMc))
            pdfMc = pdfGen*genIn
            pdfMeans = np.mean(pdfMc) # NB these are uniformly distributed
            pdfStd = np.std(pdfMc) # NB these are uniformly distributed
            
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
            
            pdfMcU = (pdfMc - pdfMeans)/pdfStd
            
        return pdfMc, pdfMcU
        
        