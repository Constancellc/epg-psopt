import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from matplotlib import cm
from sklearn.decomposition import TruncatedSVD


def calcVar(X):
    i=0
    var = np.zeros(len(X))
    for x in X:
        var[i] = x.dot(x)
        i+=1
    return var

class exampleClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'

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
        self.Vmax = lp0data['Vp']
        self.Vmin = lp0data['Vm']
        self.DVmax = 0.06 # pu
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','busCoords.pkl'),'rb') as handle:
            self.busCoords = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','branches.pkl'),'rb') as handle:
            self.branches = pickle.load(handle)
        
        # load the fixed model, as this always exists
        LMfix = loadLinMagModel(self.feeder,self.linPoint,WD,'Lpt')
        Kyfix=LMfix['Ky'];Kdfix=LMfix['Kd']
        dvBase = LMfix['vKvbase'] # NB: this is different to vBase for ltc/regulator models!

        KyPfix = Kyfix[:,:Kyfix.shape[1]//2]
        KdPfix = Kdfix[:,:Kdfix.shape[1]//2]
        Kfix = np.concatenate((KyPfix,KdPfix),axis=1)
        KfixCheck = np.sum(Kfix==0,axis=1)!=Kfix.shape[1] # [can't remember what this is for...]
        Kfix = Kfix[KfixCheck]
        dvBase = dvBase[KfixCheck]
        
        self.dvBase = dvBase
        self.KfixPu = dsf.vmM(1/dvBase,Kfix)
        self.vFixYNodeOrder = LMfix['vYNodeOrder']
        self.v_idx_fix = LMfix['v_idx']
    
    def loadNetModel(self,netModel=None):
        if netModel==None:
            netModel = self.netModelNom
        
        if not netModel:
            # IF using the FIXED model:
            LM = loadLinMagModel(self.feeder,self.linPoint,self.WD,'Lpt')
            Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
            vBase = LM['vKvbase']

            xhyN = xhy0/self.linPoint # needed seperately for lower down
            xhdN = xhd0/self.linPoint
            xNom = np.concatenate((xhyN,xhdN))
            b0lo = (Ky.dot(xhyN*self.loadPointLo) + Kd.dot(xhdN*self.loadPointLo) + bV)/vBase # in pu
            b0hi = (Ky.dot(xhyN*self.loadPointHi) + Kd.dot(xhdN*self.loadPointHi) + bV)/vBase # in pu

            KyP = Ky[:,:Ky.shape[1]//2]
            KdP = Kd[:,:Kd.shape[1]//2]
            Ktot = np.concatenate((KyP,KdP),axis=1)
            vYZ = LM['vYNodeOrder']
        elif netModel>0:
            # IF using the LTC model:
            LM = loadNetModel(self.feeder,self.linPoint,self.WD,'Lpt',netModel)
            A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
            vBase = LM['Vbase']
            
            xhyN = xhy0/self.linPoint # needed seperately for lower down
            xhdN = xhd0/self.linPoint
            xNom = np.concatenate((xhyN,xhdN))
            b0lo = (A.dot(xNom*self.loadPointLo) + bV)/vBase # in pu
            b0hi = (A.dot(xNom*self.loadPointHi) + bV)/vBase # in pu
            
            KyP = A[:,0:len(xhy0)//2] # these might be zero if there is no injection (e.g. only Q)
            KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
            
            Ktot = np.concatenate((KyP,KdP),axis=1)
            
            vYZ = LM['vYNodeOrder']
            
        KtotCheck = np.sum(Ktot==0,axis=1)!=Ktot.shape[1] # [this seems to get rid of fixed regulated buses]
        Ktot = Ktot[KtotCheck]
        
        self.xhyNtot = xhyN
        self.xhdNtot = xhdN
        self.xNom = xNom
        self.b0lo = b0lo[KtotCheck]
        self.b0hi = b0hi[KtotCheck]
        vBase = vBase[KtotCheck]
        
        self.vTotBase = vBase
        self.KtotPu = dsf.vmM(1/vBase,Ktot) # scale to be in pu
        self.vTotYNodeOrder = vYZ[KtotCheck]
        self.v_idx_tot = LM['v_idx']
        
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
                ax.plot((busCoords[bus1][0],busCoords[bus2][0]),(busCoords[bus1][1],busCoords[bus2][1]),Color='#777777')
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
        
    def ccColorbar(self,plt,minMax):
        xlm = plt.xlim()
        ylm = plt.ylim()
        top = ylm[1]
        btm = ylm[1] - np.diff(ylm)*0.25
        xcrd = xlm[1] - np.diff(xlm)*0.25

        for i in range(100):
            y1 = btm+(top-btm)*(i/100)
            y2 = btm+(top-btm)*((i+1)/100)
            plt.plot([xcrd,xcrd],[y1,y2],lw=6,c=cm.viridis(i/100))
        tcks = [str(round(minMax[0],3)),str(round(np.mean(minMax),3)),str(round(minMax[1],3))]
        for i in range(3):
            y_ = btm+(top-btm)*(i/2)-2
            plt.annotate('  '+tcks[i]+' pu',(xcrd,y_))
        
    def plotNetBuses(self,type):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.getBusPhs()

        self.plotBranches(ax)
        if type=='vLo':
            setMean, setMeanMinMax = self.getSetMean(self.b0lo)
        elif type=='vHi':
            setMean, setMeanMinMax = self.getSetMean(self.b0hi)
        elif type=='logVar':
            setMean, setMeanMinMax = self.getSetMean(np.log10(self.KtotUvar))
        
        self.plotBuses(ax,setMean,setMeanMinMax)

        plt.title(self.feeder)
        self.ccColorbar(plt,setMeanMinMax)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        print('Complete')
        plt.show()
        
    
    def runLinHc(self,nMc,pdfData,model='nom'):
        Vp_pct = np.zeros(pdfData['nP'])
        Cns_pct = np.zeros(list(pdfData['nP'])+[5])
        hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
        hcGenAll = np.array([])
        genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
        
        nV = self.KtotPu.shape[0]
        nS = self.KtotPu.shape[1]
        
        for i in range(pdfData['nP'][0]):
            pdf = hcPdfs(self.feeder,netModel=self.netModelNom)
            Mu = pdf.halfLoadMean(self.loadScaleNom,self.xhyNtot,self.xhdNtot)
            pdfMc, pdfMcU = pdf.genPdfMcSet(nMc,Mu)
            
            genTot0 = np.sum(pdfMc,axis=0)
            genTotSort = genTot0.copy()
            genTotSort.sort()
            
            if model=='nom':
                DelVout = (self.KtotPu.dot(pdfMc).T)*1e3
                ddVout = abs((self.KfixPu.dot(pdfMc).T)*1e3) # just get abs change
                Vmax = np.ones(len(DelVout))
                
            elif model=='svd':
                
                DvMu0 = self.KtotU.dot(np.sqrt(pdfData['prms'])*np.ones(nS))
                
                DelVout = self.KtotSvd.dot(pdfMcU)
                ddVout = abs((self.KfixPu.dot(pdfMc).T)*1e3) # just get abs change
                
                # vbMu0 = 
                
            
            for jj in range(pdfData['nP'][-1]):
                if model=='nom':
                    VminKlo = np.ones(nV)*self.Vmin - self.b0lo
                    VmaxKlo = np.ones(nV)*self.Vmax - self.b0lo
                    VminKhi = np.ones(nV)*self.Vmin - self.b0hi
                    VmaxKhi = np.ones(nV)*self.Vmax - self.b0hi
                    DVmaxK = np.ones(nV)*self.DVmax    
                elif model=='svd':
                    dMu0 = DvMu0*pdfData['mu_k'][jj]
                    
                    VminKlo = self.KtotUsvd.dot((np.ones(nV)*self.Vmin - self.b0lo - dMu0)/self.svdLim)
                    VmaxKlo = self.KtotUsvd.dot((np.ones(nV)*self.Vmax - self.b0lo - dMu0)/self.svdLim)
                    VminKhi = self.KtotUsvd.dot((np.ones(nV)*self.Vmin - self.b0hi - dMu0)/self.svdLim)
                    VmaxKhi = self.KtotUsvd.dot((np.ones(nV)*self.Vmax - self.b0hi - dMu0)/self.svdLim)
                    

                    DVmaxK = np.ones(nV)*self.DVmax
                    
                genTot = genTot0*pdfData['mu_k'][jj]

                
                vDv = ddVout*pdfData['mu_k'][jj]
                vV = (DelVout*pdfData['mu_k'][jj])
                
                # vLo[vLo<0.5] = 1.0
                # vHi[vHi<0.5] = 1.0

                vLoMax = dsf.mvM(vV,1/VmaxKlo)
                vLoMin = dsf.mvM(vV,1/VminKlo)
                vHiMax = dsf.mvM(vV,1/VmaxKhi)
                vHiMin = dsf.mvM(vV,1/VminKhi)
                vDvMax = dsf.mvM(vDv,1/DVmaxK)
                
                minVlo = np.max(vLoMin,axis=1)
                minVhi = np.max(vHiMin,axis=1)
                maxVlo = np.max(vLoMax,axis=1)
                maxVhi = np.max(vHiMax,axis=1)
                maxDv = np.max(vDvMax,axis=1)
                
                Cns_pct[i,jj] = 100*np.array([sum(maxDv>1),sum(maxVhi>1),sum(minVhi>1),sum(maxVlo>1),sum(minVlo>1)])/nMc
                inBounds = np.any(np.array([maxVhi>1,minVhi>1,maxVlo>1,minVlo>1,maxDv>1]),axis=0)
                
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
        self.linHcRsl = {}
        self.linHcRsl['hcGenSet'] = hcGenSet
        self.linHcRsl['Vp_pct'] = Vp_pct
        self.linHcRsl['Cns_pct'] = Cns_pct
        self.linHcRsl['hcGenAll'] = hcGenAll
        self.linHcRsl['genTotSet'] = genTotSet
    
    def getCovMat(self):
        self.KtotUcov = self.KtotU.dot(self.KtotU.T)
        self.KtotUcorr = dsf.vmvM(1/np.sqrt(np.diag(self.KtotUcov)),self.KtotUcov,1/np.sqrt(np.diag(self.KtotUcov)))
        
    
    def busViolationVar(self,Sgm):
        limA0 =   self.Vmax - self.b0lo
        limB0 =   self.Vmax - self.b0hi
        limC0 = -(self.Vmin - self.b0lo)
        limD0 = -(self.Vmin - self.b0hi)
        lim = np.min(np.array([limA0,limB0,limC0,limD0]),axis=0)
        
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
        
        vars.sort()
        stds = np.sqrt(vars)
        stds = stds/stds[-1]
        
        NStd = []
        N0 = len(stds)
        
        for stdlim in stdLim:
            NStd = NStd + [N0 - np.argmin(abs(stds - (1-stdlim)))]
        print('\nStdLim:',stdLim)
        print('NStd:',NStd,', out of ', N0)
        # LM.plotNetBuses('logVar')
        
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

        
class hcPdfs:
    def __init__(self,feeder,netModel=0,dMu=0.01,pdf=None):
        if netModel==0:
            circuitK = {'eulv':1.8,'usLv':5.0,'13bus':4.8,'34bus':5.4,'123bus':3.0,'8500node':1.2,'epri5':2.4,'epri7':2.0,'epriJ1':1.2,'epriK1':1.2,'epriM1':1.5,'epri24':1.5}
        elif netModel==1:
            circuitK = {'13bus':6.0,'34bus':8.0,'123bus':3.6}
        elif netModel==2:
            circuitK = {'8500node':2.5,'epriJ1':6.0,'epriK1':1.5,'epriM1':1.8,'epri24':1.5}
        
        self.dMu = dMu
        mu_k = circuitK[feeder]*np.arange(dMu,1.0,dMu) # NB this is as a PERCENTAGE of the chosen nominal powers.
        if pdf==None:
            pdfName = 'gammaFlat'
            k = np.array([3.0]) # we do not know th, sigma until we know the scaling from mu0.
            params = k
            self.pdf = {'name':pdfName,'prms':params,'mu_k':mu_k,'nP':(len(params),len(mu_k))}
    
    
    def halfLoadMean(self,scale,xhyN,xhdN):
        # scale suggested as: LM.scaleNom = lp0data['kLo'] - lp0data['k']
        
        roundI = 1e0
        Mu0_y = -scale*roundI*np.round(xhyN[:xhyN.shape[0]//2]/roundI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
        Mu0_d = -scale*roundI*np.round(xhdN[:xhdN.shape[0]//2]/roundI - 1e6*np.finfo(np.float64).eps)
        Mu0 = np.concatenate((Mu0_y,Mu0_d))
        
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
        return Mu0
        
    def genPdfMcSet(self,nMc,Mu0):
        if self.pdf['name']=='gammaFlat':
            k = self.pdf['prms']
            np.random.seed(0)
            pdfMc0 = (np.random.gamma(k,1/np.sqrt(k),(len(Mu0),nMc)))
            pdfMc = dsf.vmM( 1e-3*Mu0/np.sqrt(k),pdfMc0 )
            pdfMcU = pdfMc0 - np.sqrt(k) # zero mean, unit variance
        
        return pdfMc, pdfMcU
        
        