# This script is a script to build all of the linear models that we want to use for building the QP model in Q, t, P, etc. It borrows heavily from linearise_manc_py.m; the plan is to try and get rid of as much detritus as possible for this case now we understand a bit better what is going on...

import numpy as np
import os, sys, win32com.client, time, pickle, logging
from scipy import sparse
from scipy.linalg import block_diag
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
from dss_voltage_funcs import *
import dss_stats_funcs as dsf
from importlib import reload
from scipy.linalg import toeplitz

plt.style.use('tidySettings')
from matplotlib.collections import LineCollection
from matplotlib import cm, patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from cvxopt import matrix, solvers
from win32com.client import makepy
from scipy.linalg import ldl
from mosek.fusion import *


# MISC funcs: ==================== 
def equalMat(n):
    return toeplitz([1]+[0]*(n-2),[1,-1]+[0]*(n-2))

def dirPrint(obj):
    print(*dir(obj),sep='\n')

def QtoH(Q):
    L,D,P = ldl(Q) #NB not implemented here, but could also use spectral decomposition.
    Pinv = np.argsort(P)
    if min(np.diag(D))<0:
        print('Warning: not PSD, removing negative D elements')
        D[D<0]=0
    
    H = dsf.mvM(L[P],np.sqrt(np.diag(D))) # get rid of the smallest eigenvalue,
    print('Q error norm:',np.linalg.norm( H[Pinv].dot(H[Pinv].T) - Q ))
    return H,Pinv

def aMulBsp(a,b):
    # returns a.dot(b) if b is sparse, as a numpy array.
    val = (b.T.dot(a.T)).T
    if sparse.issparse(val):
        val = val.toarray()
    return val

def getTrilIdxs(n):
    idxI = []
    idxJ = []
    for i in range(n):
        for j in range(i+1):
            idxI.append(i)
            idxJ.append(j)
    
    return idxI, idxJ


# CLASS from here ==================

class buildLinModel:
    def __init__(self,fdr_i=6,linPoints=[None],pCvr = 0.75,saveModel=False,setCapsModel='linPoint',
                                    FD=sys.argv[0],modelType=None,method='fpl',SD=[],pltSave=False):
        self.WD = os.path.dirname(FD)
        self.setCapsModel = setCapsModel
        self.SD = SD
        
        logging.basicConfig(filename=os.path.join(self.WD,'example.log'),filemode='w',level=logging.INFO)
        self.log = logging.getLogger()
        self.log.info('Feeder: '+str(fdr_i))
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']
        # fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']
        
        if type(fdr_i) is int:
            self.feeder=fdrs[fdr_i]
        else:
            self.feeder=fdr_i
        
        self.fn = get_ckt(self.WD,self.feeder)[1]
        
        try:
            with open(os.path.join(self.WD,'lin_models',self.feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
                lp0data = pickle.load(handle)
            if linPoints[0]==None:
                linPoints=[lp0data['k']]
            # if self.setCapsModel=='linPoint':
                # self.capPosLin=lp0data['capPosOut']
            # else:
            self.capPosLin=None
        except:
            if linPoints[0]==None:
                linPoints = np.array([1.0])
            self.capPosLin=None
        
        if modelType not in [None,'loadModel','loadAndRun','loadOnly']:
            self.initialiseOpenDss()
        
        self.pCvr = pCvr
        self.qCvr = 0.0
        self.linPoint = linPoints[0]
        self.method=method
        self.fixFot = 1 # nominally have the FOT change according to the load voltage sensitivity if req.
        
        # BUILD MODELS
        if modelType in ['buildSave','buildOnly']:
            self.makeCvrQp()
            self.initialiseOpenDss()
            self.setupPlots()
            self.slnF0 = self.runQp(np.zeros(self.nCtrl))
            delattr(self,'qpHlss')
            self.slnD0 = self.qpDssValidation(np.zeros(self.nCtrl))
            if modelType=='buildSave': self.saveLinModel()
        elif modelType == 'linOnly':
            self.createNrelModel(linPoints[0])
            self.testVoltageModel()
        elif modelType == 'fxd':
            self.createNrelModel(linPoints[0])
            self.createTapModel(linPoints[0],cvrModel=True) # this seems to be ok
            self.createFxdModel()
        elif modelType == 'ltc':
            self.createNrelModel(linPoints[0])
            self.createTapModel(linPoints[0],cvrModel=True) # this seems to be ok
            self.createWmodel() # this seems to be ok
            self.createLtcModel()
        elif modelType=='cvr':
            self.createCvrModel()
        elif modelType == 'plotOnly':
            self.barePlotSetup()
            self.setupPlots()
            self.plotNetwork(pltSave=pltSave)
        elif modelType in [None,'loadModel','loadAndRun','loadOnly']:
            self.loadLinModel()
            self.WD = os.path.dirname(FD) # WARNING! some paths may be a bit dodge...?
            self.SD = SD
            self.fn = get_ckt(self.WD,self.feeder)[1]
        
        if modelType in [None,'loadModel','loadAndRun']:
            self.initialiseOpenDss()
        
        # if modelType not in [None,'loadOnly','loadAndRun']:
            # self.setupPlots()

        
        if modelType in [None,'loadModel']:
            # self.testCvrQp()
            modesAll = ['full','part','maxTap','minTap','nomTap','loss','load']
            # modesAll = ['minTap']
            modesAll = ['full']
            optType = ['mosekFull']
            obj = 'hcGen'
            # obj = 'hcLds'
            # obj = 'opCst'
            for modeSet in modesAll:
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',modeSet)
                self.runCvrQp(modeSet,optType=optType,obj=obj)
                self.plotNetBuses('qSln')
                # self.printQpSln()
                # self.showQpSln()
        
        if modelType in ['loadAndRun']:
            self.setupConstraints() # this reloads the constraints in.
            self.runQpSet()
            # self.runQp4test()
            
        # if modelType==None:
            # self.
        # self.plotNetwork()
        # self.plotNetBuses('p0')
        # self.plotNetBuses('q0')
        # self.plotNetBuses('v0')
        
        # self.runCvrQp()
        # self.plotNetBuses('qSln')
        # self.plotNetBuses('vSln')
        
        # self.showQpSln(self.slnX,self.slnF)
        # self.testCvrQp()
        # self.snapQpComparison()
        # self.loadLoadProfiles()
        if saveModel:
            self.saveLinModel()
            # self.saveNrelModel()
    
    def barePlotSetup(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile ('+self.fn+'.dss)'
        
        YNodeOrder = DSSCircuit.YNodeOrder
        self.YZ = DSSCircuit.YNodeOrder[3:]
        
        self.xY, self.xD, self.pyIdx, self.pdIdx  = ldValsOnly( DSSCircuit ) # NB these do not change with the circuit!
        self.vYNodeOrder = YNodeOrder[3:]
        self.vInYNodeOrder = self.vYNodeOrder # for arguments sake...
        self.SyYNodeOrder = vecSlc(self.vYNodeOrder,self.pyIdx)
        self.SdYNodeOrder = vecSlc(self.vYNodeOrder,self.pdIdx)
        
        self.nPy = len(self.pyIdx[0])
        self.nPd = len(self.pdIdx[0])
        
        self.nPctrl = self.nPy + self.nPd
        
    def runQp4test(self,strategy='full',obj='opCst'):
        solvers.options['show_progress']=False
        optTypes = ['cvxopt','cvxMosek','mosekNom','mosekInt']
        for optType in optTypes:
            try:
                self.runCvrQp(strategy=strategy,obj=obj,optType=optType)
                self.printQpSln()
            except:
                print('Solution failed.')
        
    def runQpSet(self,saveQpSln=True,objs=None,strategySet=None,invLossTypes=None):
        
        if objs is None: objs = ['opCst','hcGen','hcLds']
        if strategySet is None: strategySet = { 'opCst':['full','phase','nomTap','load','loss'],'hcGen':['full','phase','nomTap','maxTap'],'hcLds':['full','phase','nomTap','minTap'] }
        if invLossTypes is None: invLossTypes = ['None','Low','Hi']
        
        for invType in invLossTypes:
            kQlossC,kQlossQ = self.getInvLossCoeffs(type=invType)
            self.setQlossOfs(kQlossQ=kQlossQ,kQlossC=0) # nominally does NOT include turn on losses!
            
            optType = ['mosekFull']
            qpSolutions = {}
            for obj in objs:
                for strategy in strategySet[obj]:
                    self.runCvrQp(strategy=strategy,obj=obj,optType=optType)
                    if 'slnD' in dir(self):
                        qpSolutions[strategy+'_'+obj] = [self.slnX,self.slnF,self.slnS,self.slnD]
                    else:
                        qpSolutions[strategy+'_'+obj] = [self.slnX,self.slnF,self.slnS]
            
            self.qpSolutions = qpSolutions
            
            if saveQpSln:
                SD = os.path.join( os.path.dirname(self.getSaveDirectory()),'results',self.feeder+'_runQpSet_out')
                SN = os.path.join(SD,self.getFilename()+'i'+self.invLossType+'_sln.pkl')
                if not os.path.exists(SD):
                    os.mkdir(SD)
                with open(SN,'wb') as outFile:
                    print('Results saved to '+ SN)
                    pickle.dump(qpSolutions,outFile)
    
    def loadQpSet(self,invType=None):
        if invType is None:
            invType = self.invLossType
        # SN = os.path.join(self.getSaveDirectory(),self.feeder+'_runQpSet_out',self.getFilename()+'_sln.pkl')
        SN = os.path.join(os.path.dirname(self.getSaveDirectory()),'results',self.feeder+'_runQpSet_out',self.getFilename()+'i'+invType+'_sln.pkl')
        with open(SN,'rb') as outFile:
            self.qpSolutions = pickle.load(outFile)
    
    def loadQpSln(self,strategy='full',obj='opCst'):
        key = strategy+'_'+obj
        self.slnX = self.qpSolutions[key][0]
        self.slnF = self.qpSolutions[key][1]
        self.slnS = self.qpSolutions[key][2]
        # if len(self.qpSolutions[key])>3:
        self.slnD = self.qpSolutions[key][3]
            
    def getSaveDirectory(self):
        return os.path.join(self.WD,'lin_models','cvr_models',self.feeder)
    
    def getFilename(self):
        power = str(np.round(self.linPoint*100).astype(int)).zfill(3)
        aCvr = str(np.round(self.pCvr*100).astype(int)).zfill(3)
        if self.method=='fot':
            aCvr = aCvr+self.method
        return self.feeder+'P'+power+'A'+aCvr
    
    def saveLinModel(self):
        SD = self.getSaveDirectory()
        SN = os.path.join(SD,self.getFilename()+'.pkl')
        if not os.path.exists(SD):
            os.makedirs(SD)
        self.dssStuff = [] # can't be saved
        with open(SN,'wb') as outFile:
            pickle.dump(self,outFile)
    
    def loadLinModel(self):
        SN = os.path.join(self.getSaveDirectory(),self.getFilename()+'.pkl')
        with open(SN,'rb') as outFile:
            savedSelf = pickle.load(outFile)
        self.__dict__ = savedSelf.__dict__.copy()
    
    def initialiseOpenDss(self):
        # sys.argv=["makepy","OpenDSSEngine.DSS"]
        # makepy.main()
        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.dssStuff = [DSSObj,DSSObj.Text,DSSObj.ActiveCircuit,DSSObj.ActiveCircuit.Solution]
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile '+self.fn
        
    def makeCvrQp(self):
        # things to make.
        # Control variables:
        # 1. reactive power of generators individually
        # 2. taps
        # 3. generation
        # 
        # CONSTRAINTS
        # 1. voltage magnitudes
        # 2. current magnitudes in the transformer
        # 3. (tap + Q constraints + curtailment)
        # 
        # COST FUNCTION
        # 1. real power losses
        # 2. total power load
        # 3. generated power.
        
        lin_point = self.linPoint
        self.currentLinPoint = self.linPoint
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(self.pCvr)+' cvrvars='+str(self.qCvr)
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = lin_point
        DSSSolution.Solve()
        print('\nNominally converged (makeCvrQp):',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        self.Cap_No0 = getCapPos(DSSCircuit)
        if self.capPosLin==None:
            self.capPosLin = self.Cap_No0
        print(self.Cap_No0)
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        self.loadCvrDssModel(self.pCvr,self.qCvr,loadMult=lin_point)
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        
        self.xY, self.xD, self.pyIdx, self.pdIdx  = ldValsOnly( DSSCircuit ) # NB these do not change with the circuit!
        
        self.qyIdx = [self.pyIdx[0]+DSSCircuit.NumNodes-3] # NB: note that this is wrt M, not YNodeOrder.
        self.syIdx = np.concatenate((self.pyIdx[0],self.qyIdx[0]))
        self.nPy = len(self.pyIdx[0])
        self.nPd = len(self.pdIdx[0])

        
        self.vKvbase = get_Yvbase(DSSCircuit)[3:]
        self.vKvbaseY = self.vKvbase[self.pyIdx[0]]
        self.vKvbaseD = self.vKvbase[self.pdIdx[0]]*np.sqrt(3)
        self.vKvbaseX = np.concatenate( (self.vKvbaseY,self.vKvbaseY,self.vKvbaseD,self.vKvbaseD) )
        
        if len(self.pdIdx[0])>0:
            H = create_Hmat(DSSCircuit)[self.pdIdx[0]+3]
        else:
            H = np.zeros((0,Ybus.shape[0]))
        
        # # Only consider single phase loads for now.
        VhYpu = abs(YNodeV[3:]/self.vKvbase)
        VhDpu = abs(H.dot(YNodeV)/self.vKvbaseD)
        self.kYcvr0 = np.concatenate((VhYpu**self.pCvr,VhYpu**self.qCvr))
        self.kDcvr0 = np.concatenate((VhDpu**self.pCvr,VhDpu**self.qCvr))
        
        xYcvr = lin_point*self.xY*self.kYcvr0
        xYcvr0 = xYcvr[self.syIdx]
        xDcvr = lin_point*self.xD*self.kDcvr0
        
        self.YZ = DSSCircuit.YNodeOrder[3:]
        
        # >>> 3. Create linear models for voltage in S
        V0 = YNodeV[0:3]
        Vh = YNodeV[3:]
        dVh = H.dot(YNodeV)
        DSSSolution.LoadMult=0.0
        DSSSolution.Solve()
        VoffLoad = tp_2_ar(DSSCircuit.YNodeVarray) 
        VnoLoad = VoffLoad[3:]
        dVnoLoad = H.dot(VoffLoad)
        
        self.nV = len(Vh)
        self.createTapModel(lin_point,cvrModel=True) # this seems to be ok. creates Kt and Mt matrices.
        self.nT = self.Mt.shape[1]
        
        # control (c) variables (in order): Pgen(Y then D) (kW),Qgen(Y then D) (kvar),t (no.).
        # notation as 'departure2arrival'
        self.nPctrl = self.nPy+self.nPd
        self.nSctrl = self.nPctrl*2
        self.nCtrl = self.nSctrl + self.nT
        
        self.lScale = 1e-3 # to convert to kW
        self.xSscale = 1e-3 # these values seem about correct, it seems.
        self.xTscale = 160*1e0
        self.xScale = np.r_[self.xSscale*np.ones(self.nSctrl),self.xTscale*np.ones(self.nT)] # change the scale factor of matrices here
        
        self.X0ctrl = self.xScale*np.concatenate( (xYcvr0[:self.nPy],xDcvr[:self.nPd],xYcvr0[self.nPy:],xDcvr[self.nPd::],np.zeros(self.nT)) )
        
        print('Create linear models:\n',time.process_time()); t = time.time()
        
        if self.method=='fot':
            # My,Md,a,dMy,dMd,da = firstOrderTaylor( Ybus,Vh,V0,xYcvr,xDcvr,H[:,3:] ); \
                            # print('===== Using the FOT method =====')
            My,Md,a,dMy,dMd,da = firstOrderTaylorQuick( Ybus,Vh,V0,xYcvr,xDcvr,H[:,3:] ); \
                            print('===== Using the FOT method =====')
            My = My[:,self.syIdx]
            dMy = dMy[:,self.syIdx]
            
            if 'fixFot' in dir(self):
                # FIRST recreate the equations:
                Ky,Kd,b = nrelLinK( My,Md,Vh,xYcvr0,xDcvr )
                dKy,dKd,self.dKt,db = lineariseMfull( dMy,dMd,H[:,3:].dot(self.Mt),dVh,xYcvr0,xDcvr,np.zeros(self.nT) )
                Kc2d = dsf.mvM( np.concatenate( (dKy[:,:self.nPy],dKd[:,:self.nPd],
                            dKy[:,self.nPy:],dKd[:,self.nPd::],
                            self.dKt),axis=1), 1/self.xScale )
                self.Kc2v = dsf.mvM( np.concatenate( (Ky[:,:self.nPy],Kd[:,:self.nPd],
                                        Ky[:,self.nPy::],Kd[:,self.nPd::],
                                        self.Kt),axis=1), 1/self.xScale )
                Kc2vloadpu = dsf.mvM( np.concatenate( (  dsf.vmM( 1/VhYpu[self.pyIdx[0]], dsf.vmM( 1/self.vKvbaseY,self.Kc2v[self.pyIdx[0]] ) ), 
                                             dsf.vmM( 1/VhDpu, dsf.vmM( 1/self.vKvbaseD, Kc2d[:self.nPd] ) ) ), axis=0), 1/self.xScale )
                Kc2pC = np.concatenate((xYcvr0[:self.nPy],xDcvr[:self.nPd]))
                Kc2p =  dsf.vmM(Kc2pC*self.pCvr,Kc2vloadpu)
                self.ploadL = -self.lScale*self.xScale*np.sum(Kc2p,axis=0)
                self.ploadC = -self.lScale*sum(Kc2pC)
                
                self.Kc2p = self.lScale*self.xScale*Kc2p
                
                # THEN MODIFY MY MD MT AS APPROPRIATE.
                Mc2v = dsf.mvM( np.concatenate((My[:,:self.nPy],Md[:,:self.nPd],
                            My[:,self.nPy::],Md[:,self.nPd::],
                            self.Mt),axis=1), 1/self.xScale )
                
                a = a - Mc2v.dot(np.r_[ self.Kc2p.dot(self.X0ctrl), np.zeros(self.nPctrl+self.nT)] )
                Mc2v = dsf.mvM( Mc2v.dot( np.eye(self.nCtrl) + np.r_[self.Kc2p,np.zeros((self.nPctrl+self.nT,self.nCtrl))] ) , self.xScale )
                
                My = np.c_[ Mc2v[:,:self.nPy],Mc2v[:,self.nPctrl:self.nPctrl+self.nPy] ]
                Md = np.c_[ Mc2v[:,self.nPy:self.nPctrl],Mc2v[:,self.nPctrl+self.nPy:self.nSctrl] ]
        elif self.method=='fpl':
            My,Md,a,dMy,dMd,da = cvrLinearization( Ybus,Vh,V0,H,0,0,self.vKvbase,self.vKvbaseD ); \
                            print('Using the FLP method')
            My = My[:,self.syIdx]
            dMy = dMy[:,self.syIdx]
        
        Ky,Kd,b = nrelLinK( My,Md,Vh,xYcvr0,xDcvr )
        dKy,dKd,self.dKt,db = lineariseMfull( dMy,dMd,H[:,3:].dot(self.Mt),dVh,xYcvr0,xDcvr,np.zeros(self.nT) )
        
        print('Linear models created.:',time.time()-t)
        
        Vh0 = (My.dot(xYcvr0) + Md.dot(xDcvr)) + a # for validation
        Va0 = (Ky.dot(xYcvr0) + Kd.dot(xDcvr)) + b # for validation
        dVh0 = (dMy.dot(xYcvr0) + dMd.dot(xDcvr)) + da # for validation
        dVa0 = (dKy.dot(xYcvr0) + dKd.dot(xDcvr)) + db # for validation
        
        self.log.info('\nVoltage clx error (lin point), Volts:'+str(np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh)))
        self.log.info('Voltage clx error (no load point), Volts:'+str(np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad)))
        self.log.info('\nVoltage abs error (lin point), Volts:'+str(np.linalg.norm(Va0-abs(Vh))/np.linalg.norm(abs(Vh))))
        self.log.info('Voltage abs error (no load point), Volts:'+str(np.linalg.norm(abs(b)-abs(VnoLoad))/np.linalg.norm(abs(VnoLoad))))
        if len(dVh)>0:
            self.log.info('\n Delta voltage clx error (lin point), Volts:'+str(np.linalg.norm(dVh0-dVh)/np.linalg.norm(dVh)))
            self.log.info('Delta voltage clx error (no load point), Volts:'+str(np.linalg.norm(da-dVnoLoad)/np.linalg.norm(dVnoLoad)))
            self.log.info('\nDelta voltage abs error (lin point), Volts:'+str(np.linalg.norm(dVa0-abs(dVh))/np.linalg.norm(abs(dVh))))
            self.log.info('Delta voltage abs error (no load point), Volts:'+str(np.linalg.norm(abs(db)-abs(dVnoLoad))/np.linalg.norm(abs(dVnoLoad))))
            

        self.My = My
        self.Md = Md
        self.aV = a
        self.bV = b
        self.H = H
        self.V0 = V0
        
        
        Kc2d = dsf.mvM( np.concatenate( (dKy[:,:self.nPy],dKd[:,:self.nPd],
                                dKy[:,self.nPy:],dKd[:,self.nPd::],
                                self.dKt),axis=1), 1/self.xScale )
        del(dKy); del(dKd); delattr(self,'dKt')
        self.Kc2v = dsf.mvM( np.concatenate( (Ky[:,:self.nPy],Kd[:,:self.nPd],
                                            Ky[:,self.nPy::],Kd[:,self.nPd::],
                                            self.Kt),axis=1), 1/self.xScale )
        del(Ky); del(Kd); delattr(self,'Kt')# kT comes from createTapModel it seems
        
        # # see e.g. WB 14/5/19
        # NB NB NB! PYIDX IS CORRECT as we are drawing out the voltages at the loads from the voltage index!
        Kc2vloadpu = dsf.mvM( np.concatenate( (  dsf.vmM( 1/VhYpu[self.pyIdx[0]], dsf.vmM( 1/self.vKvbaseY,self.Kc2v[self.pyIdx[0]] ) ), 
                                                 dsf.vmM( 1/VhDpu, dsf.vmM( 1/self.vKvbaseD, Kc2d[:self.nPd] ) ) ), axis=0), 1/self.xScale )
        
        Kc2pC = np.concatenate((xYcvr0[:self.nPy],xDcvr[:self.nPd]))
        Kc2p =  dsf.vmM(Kc2pC*self.pCvr,Kc2vloadpu)
        
        self.ploadL = -self.lScale*self.xScale*np.sum(Kc2p,axis=0)
        self.ploadC = -self.lScale*sum(Kc2pC)
        del(Kc2pC); del(Kc2p); del(Kc2vloadpu); del(Kc2d)
        
        
        
        self.vYNodeOrder = YNodeOrder[3:]
        self.SyYNodeOrder = vecSlc(self.vYNodeOrder,self.pyIdx)
        self.SdYNodeOrder = vecSlc(self.vYNodeOrder,self.pdIdx)
        self.currentLinPoint = lin_point
        
        del(My); del(Md); del(dMy); del(dMd)
        self.createWmodel(lin_point) # this seems to be ok
        
        f0 = self.v2iBrYxfmr.dot(YNodeV)[self.iXfmrModelled] # voltages to currents through xfmrs
        fNoload = self.v2iBrYxfmr.dot(VoffLoad)[self.iXfmrModelled] # voltages to currents through xfmrs - NB this should be the same as 'self.aIxfmr'.
        
        if np.any(f0==0): # occurs sometimes, e.g. Ckt 7, which have floating transformers with no loads on the other side.
            df = min( np.min(f0[f0!=0]),np.min(fNoload[fNoload!=0]) )*1e-10 # maybe this is doing us in?
            f0 = f0 + df*np.ones(len(f0))
            fNoload = fNoload + df*np.ones(len(fNoload))
        
        
        f0cpx = self.WyXfmr.dot(xYcvr0) + self.WdXfmr.dot(xDcvr) + self.WtXfmr.dot(np.zeros(self.WtXfmr.shape[1])) + self.aIxfmr
        self.log.info('\nCurrent cpx error (lin point):'+str(np.linalg.norm(f0cpx-f0)/np.linalg.norm(f0)))
        self.log.info('Current cpx error (no load point):'+str(np.linalg.norm(self.aIxfmr-fNoload)/np.linalg.norm(fNoload)))
        
        
        
        if 'recreateKc2i' in dir(self):
            KyW, KdW, KtW, self.bW = lineariseMfull(self.WyXfmr,self.WdXfmr,self.WtXfmr,f0,xYcvr0,xDcvr,np.zeros(self.WtXfmr.shape[1]))
            f0lin = KyW.dot(xYcvr0) + KdW.dot(xDcvr) + KtW.dot(np.zeros(self.WtXfmr.shape[1])) + self.bW
            self.log.info('\nCurrent abs error (lin point):'+str(np.linalg.norm(f0lin-abs(f0))/np.linalg.norm(abs(f0))))
            self.log.info('Current abs error (no load point):'+str(np.linalg.norm(self.bW-abs(fNoload))/np.linalg.norm(abs(fNoload)))+'( note that this is usually not accurate, it seems.)')
            self.Kc2i = dsf.mvM( np.concatenate( (KyW[:,:self.nPy],KdW[:,:self.nPd],
                                                KyW[:,self.nPy::],KdW[:,self.nPd::],
                                                KtW),axis=1), 1/self.xScale ) # limits for these are in self.iXfmrLims.
            del(KyW); del(KdW); del(KtW)
            
        # self.Mc2i = dsf.mvM( np.concatenate( (self.WyXfmr[:,:self.nPy],self.WdXfmr[:,:self.nPd],
        # self.Mc2i = sparse.csc_matrix( dsf.mvM( np.concatenate( (self.WyXfmr[:,:self.nPy],self.WdXfmr[:,:self.nPd],
                                            # self.WyXfmr[:,self.nPy::],self.WdXfmr[:,self.nPd::],
                                            # self.WtXfmr),axis=1), 1/self.xScale ) ) # limits for these are in self.iXfmrLims.
        Mc2i = dsf.mvM( np.concatenate( (self.WyXfmr[:,:self.nPy],self.WdXfmr[:,:self.nPd],
                                            self.WyXfmr[:,self.nPy::],self.WdXfmr[:,self.nPd::],
                                            self.WtXfmr),axis=1), 1/self.xScale ) # limits for these are in self.iXfmrLims.
        mc2iSmall = dsf.vmM(1/(self.iXfmrLims - np.abs(self.aIxfmr)),Mc2i)
        mc2iNorm = np.abs( dsf.vmM( 1/np.max(np.abs(mc2iSmall),axis=1),mc2iSmall ) )
        iFactor = 1e-5
        nonZero = np.where(mc2iNorm>iFactor)
        mc2iSel = np.zeros(Mc2i.shape,dtype=complex)
        mc2iSel[nonZero] = Mc2i[nonZero]
        self.Mc2i = sparse.csc_matrix(mc2iSel)
        del(Mc2i); del(mc2iSmall); del(mc2iNorm); del(mc2iSel)
        delattr(self,'WyXfmr'); delattr(self,'WdXfmr'); delattr(self,'WtXfmr')
        
        
        
        # self.Mc2v = dsf.mvM( np.concatenate((self.My[:,:self.nPy],self.Md[:,:self.nPd],
        Mc2v = dsf.mvM( np.concatenate((self.My[:,:self.nPy],self.Md[:,:self.nPd],
                                        self.My[:,self.nPy::],self.Md[:,self.nPd::],
                                        self.Mt),axis=1), 1/self.xScale )
        delattr(self,'My'); delattr(self,'Md'); delattr(self,'Mt')
        if 'recreateMc2v' in dir(self):
            self.Mc2v = Mc2v
        # M = np.block([[np.zeros((3,self.nCtrl),dtype=complex)],[self.Mc2v],[np.zeros((1,self.nCtrl),dtype=complex)]])
        M = np.block([[np.zeros((3,self.nCtrl),dtype=complex)],[Mc2v],[np.zeros((1,self.nCtrl),dtype=complex)]])
        del(Mc2v)
        
        Wcnj = dsf.mvM(np.concatenate((self.Wy[:,:self.nPy],self.Wd[:,:self.nPd],
                                self.Wy[:,self.nPy::],self.Wd[:,self.nPd::],self.Wt),axis=1).conj(), 1/self.xScale )
        delattr(self,'Wy'); delattr(self,'Wd'); delattr(self,'Wt')
        aIcnj = (self.aI).conj()
        
        Wcnj = np.delete(Wcnj,self.wregIdxs,axis=0)
        aIcnj = np.delete(aIcnj,self.wregIdxs,axis=0)
        
        aV = np.concatenate((self.V0,self.aV,np.array([0])))
        
        P = np.zeros((len(self.yzW2V),len(M)))
        P[range(len(self.yzW2V)),self.yzW2V] = 1
        P = np.delete(P,self.wregIdxs,axis=0) # don't forget we take out the regulator models!
        # PT = P.T

        # eyeDel = np.eye(len(self.Wy))
        # eyeDel = np.delete(eyeDel,self.wregIdxs,axis=0)
        # middleMat = PT.dot( eyeDel.dot( 
                # np.c_[np.zeros((len(self.Wy),3)),self.v2iBrY[:,3:].toarray().conj(),np.zeros((len(self.Wy),1) ) ] ) )
        # middleMat = 0.5*(middleMat + middleMat.conj().T)
        # # [wMm,vMm] = eigh(middleMat) # this gives negative eigenvalues (!?!?)
        # Lmm = dsf.mvM(vMm,np.sqrt(wMm))
        
        # I think that this could be factorised as H = np.r_[a,b] if we can get to aaT + bbT?
        # I don't know if we can do this efficiently though becase middleMat appears to have 
        # negative eigenvalues...?
        # qpQlss = self.lScale*np.real( (M.T).dot(middleMat.dot(M.conj())) )11
        
        
        # The input to the models are all in kW, t, and should output in kW too!
        qpQlss = self.lScale*np.real( (M.T).dot(P.T.dot(Wcnj)) )
        qpQlss = 0.5*(qpQlss + qpQlss.T) # make symmetric.
        qpHlss,self.qpHlssPinv = QtoH(qpQlss)
        self.qpHlssLin = qpHlss[getTrilIdxs(self.nCtrl)]
        # self.qpQlss = qpQlss # for debugging
        
        qpQlss0 = qpQlss.dot(self.X0ctrl);
        del(qpQlss)
        qpLlss0 = self.lScale*np.real( aV.dot(P.T.dot(Wcnj)) + aIcnj.dot(P.dot(M)) )
        
        self.qpLlss = qpLlss0 + 2*qpQlss0
        qpClss0 = self.lScale*np.real( aV.dot(P.T.dot(aIcnj)) )
        self.qpClss = qpClss0 + self.X0ctrl.dot( qpLlss0 + qpQlss0)
        
        
        pcurtL = np.zeros(self.nCtrl)
        pcurtL[:self.nPctrl] = -self.lScale/self.xSscale
        self.pcurtL = pcurtL
        
        self.setupConstraints()
        
        kQlossC,kQlossQ = self.getInvLossCoeffs()
        self.setQlossOfs(kQlossQ=kQlossQ,kQlossC=0) # nominall do NOT include turn on losses!
        
        self.loadCvrDssModel(self.pCvr,self.qCvr,loadMult=lin_point)
        self.log.info('Actual losses:'+str(DSSCircuit.Losses))
        self.log.info('Model losses'+str(self.qpClss))
        self.log.info('TLoad:'+str(-DSSCircuit.TotalPower[0] - self.lScale*DSSCircuit.Losses[0]))
        # self.log.info('TLoadEst:'+str(-self.lScale*sum(Kc2pC)))
        
        return
    
    def getHmat(self):
        n = self.nCtrl
        H = np.zeros((n,n))
        H[getTrilIdxs(n)] = self.qpHlssLin
        # self.qpHlss = H[self.qpHlssPinv]
        return H[self.qpHlssPinv]
    
    
    def getInvLossCoeffs(self,sRated=4.0,type='Low'):
        lossFracS0s = 1e-2*np.array([1.45,0.72,0.88]) #*(sRated**1) # from paper by Notton et al
        lossFracSmaxs = 1e-2*np.array([4.37,3.45,11.49]) #*(sRated**-1) # from paper by Notton et al
    
        lossSettings = {'None':[0.0,0.0],'Low':[lossFracS0s[1],lossFracSmaxs[1]],'Med':[lossFracS0s[2],lossFracSmaxs[2]], 'Hi':[lossFracS0s[0],lossFracSmaxs[0]] }
        lossSetting = 'Hi'
        
        kQlossC = lossSettings[type][0]*(sRated**1)
        kQlossQ = lossSettings[type][1]*(sRated**-1)
        self.invLossSrated = sRated
        self.invLossType = type
        return kQlossC,kQlossQ
    
    
    def setQlossOfs(self,kQlossQ=0.0,kQlossL=0.0,kQlossC=0.0,qlossRegC=0.0):
        # these should come in in WATTS (per kVA, per kVA**2 for non constant)
        qlossQdiag = np.zeros(self.nCtrl)
        qlossQdiag[self.nPctrl:self.nSctrl] = kQlossQ*self.lScale/self.xSscale
        self.qlossQdiag = qlossQdiag
        
        qlossL = np.zeros(self.nCtrl)
        qlossL[self.nPctrl:self.nSctrl] = kQlossL*self.lScale/self.xSscale
        self.qlossL = qlossL
        
        qlossC = np.zeros(self.nCtrl)
        qlossC[self.nPctrl:self.nSctrl] = kQlossC*self.lScale/self.xSscale
        self.qlossC = qlossC
        
        qlossCzero = np.zeros(self.nCtrl)
        qlossCzero[self.nPctrl:self.nSctrl] = self.qLim/1e4
        self.qlossCzero = qlossCzero
        
        qlossReg = np.zeros(self.nCtrl)
        qlossReg[self.nPctrl:self.nSctrl] = qlossRegC*self.lScale/self.xSscale
        self.qlossReg = qlossReg
    
    def getObjFunc(self,strategy,obj,tLss=False):
        if obj in ['opCst','tsCst']:
            if strategy in ['full','part','phase','nomTap']:
                H = np.sqrt(2)*self.getHmat()
                p = self.qpLlss + self.ploadL + self.pcurtL
            elif strategy in ['maxTap']:
                H = sparse.csc_matrix((self.nCtrl,self.nCtrl))
                if self.nT>0:
                    t2vPu = np.sum( dsf.vmM( 1/self.vInKvbase,self.Kc2v[self.vIn,-self.nT:] ),axis=0 )
                    p = np.r_[np.zeros((self.nPctrl*2)),-1*t2vPu] # maximise the average voltage
                else:
                    p = np.r_[np.zeros((self.nPctrl*2))]
            elif strategy=='minTap':
                H = sparse.csc_matrix((self.nCtrl,self.nCtrl))
                if self.nT>0:
                    t2vPu = np.sum( dsf.vmM( 1/self.vInKvbase,self.Kc2v[self.vIn,-self.nT:] ),axis=0 )
                    p = np.r_[np.zeros((self.nPctrl*2)),t2vPu] # minimise the average voltage
                else:
                    p = np.r_[np.zeros((self.nPctrl*2))]
            elif strategy=='loss':
                H = np.sqrt(2)*self.getHmat()
                p = self.qpLlss + self.pcurtL
            elif strategy=='load':
                H = sparse.csc_matrix((self.nCtrl,self.nCtrl))
                p = self.ploadL + self.pcurtL
        
        if obj=='hcGen':
            if strategy in ['full','part','minTap','maxTap','nomTap','phase']:
                H = np.sqrt(2)*self.getHmat()
                p = self.qpLlss + self.ploadL + self.pcurtL
        
        if obj=='hcLds':
            if strategy in ['full','part','minTap','maxTap','nomTap','phase']:
                H = sparse.csc_matrix((self.nCtrl,self.nCtrl))
                p = np.r_[ np.ones(self.nPctrl), np.zeros(self.nPctrl + self.nT) ]
                
        p = np.array([p]).T
        return H,p
    
    def getVghConstraints(self):
        # scaleFactor = 1000/self.vKvbase
        scaleFactor = 1/self.vKvbase # this seems to give the nicest results?
        # scaleFactor = np.ones(self.nV)
        vLimLo = (( self.vLo - self.bV - self.Kc2v.dot(self.X0ctrl) )*scaleFactor )[self.vIn]
        vLimUp = (( self.vHi - self.bV - self.Kc2v.dot(self.X0ctrl) )*scaleFactor )[self.vIn]
        Kc2vIn = (dsf.vmM( scaleFactor,self.Kc2v))[self.vIn]
        
        # Gvhv = np.sum(np.abs(Kc2vIn),axis=1)/np.minimum(np.abs(vLimLo),np.abs(vLimHi))
        
        Gv = np.r_[Kc2vIn,-Kc2vIn]
        hv = np.array( [np.r_[vLimUp,-vLimLo]] ).T
        return Gv,hv
        
    def getXghConstraints(self,obj):
        if obj in ['opCst']:
            xLimUp = np.r_[ np.zeros(self.nPctrl),np.ones(self.nPctrl)*self.qLim,
                                                                                np.ones(self.nT)*self.tLim ]
            xLimLo = np.r_[ -np.ones(self.nPctrl)*self.pLim,-np.ones(self.nPctrl)*self.qLim,
                                                                                -np.ones(self.nT)*self.tLim ]
            Gx = np.r_[np.eye(self.nCtrl),-np.eye(self.nCtrl)]
            
        if obj in ['tsCst']:
            xLimUp = np.r_[ np.ones(self.nPctrl)*self.pLim,np.ones(self.nPctrl)*self.qLim,
                                                                                np.ones(self.nT)*self.tLim ]
            xLimLo = np.r_[ -np.ones(self.nPctrl)*self.pLim,-np.ones(self.nPctrl)*self.qLim,
                                                                                -np.ones(self.nT)*self.tLim ]
            Gx = np.r_[np.eye(self.nCtrl),-np.eye(self.nCtrl)]
            
            
        if obj in ['hcLds']:
            xLimUp = np.r_[ np.zeros(self.nPctrl),np.ones(self.nPctrl)*self.qLim,
                                                                                np.ones(self.nT)*self.tLim ]
            xLimLo = np.r_[ -np.ones(self.nPctrl)*self.qLim,-np.ones(self.nT)*self.tLim ]
            Gx = np.r_[np.eye(self.nCtrl),
                                    -np.c_[np.zeros((self.nCtrl-self.nPctrl,self.nPctrl)),np.eye(self.nCtrl-self.nPctrl)] ]
        if obj in ['hcGen']:
            xLimUp = np.r_[ np.ones(self.nPctrl)*self.qLim,np.ones(self.nT)*self.tLim ]
            xLimLo = np.r_[ np.zeros(self.nPctrl),-np.ones(self.nPctrl)*self.qLim,-np.ones(self.nT)*self.tLim ]
            
            Gx = np.r_[ np.c_[np.zeros((self.nCtrl-self.nPctrl,self.nPctrl)),np.eye(self.nCtrl-self.nPctrl)],
                                        -np.eye(self.nCtrl)]
        hx = np.array( [np.r_[xLimUp,-xLimLo]] ).T
        return Gx,hx
        
        
    def getInqCns(self,obj):
        # Upper Control variables, then voltages, then currents; then repeat but lower.
        # xLo <= x <= xHi
        # NB: note that we take seem to take out the effect of 'x0' in the cost/constraint
        # functions outside of this function.
        
        Gv,hv = self.getVghConstraints()
        Gx,hx = self.getXghConstraints(obj)
        
        G = np.r_[Gv,Gx]
        h = np.r_[hv,hx]
        
        return G,h
    
    def remControlVrlbs(self,strategy,obj):
        # Returns oneHat, x0, which are of the form x = oneHat.dot(xCtrl) + x0, so,
        # oneHat is of dimension self.nCtrl*nCtrlActual.
        
        oneHat = np.nan # so there is something to return if all ifs fail
        x0 = np.zeros((self.nCtrl,1)) # always of this dimension.
        if strategy in ['minTap']:
            # x0[self.nPctrl:self.nSctrl] = np.ones((self.nPctrl,1))*self.qLim
            x0[self.nPctrl:self.nSctrl] = np.ones((self.nPctrl,1)) # just use 1 kVA to avoid thermal issues
        if strategy in ['maxTap']:
            # x0[self.nPctrl:self.nSctrl] = -np.ones((self.nPctrl,1))*self.qLim
            x0[self.nPctrl:self.nSctrl] = -np.ones((self.nPctrl,1)) # just use 1 kVA to avoid thermal issues
        
        if obj=='opCst':
            if strategy in ['full','loss','load']:
                oneHat = np.r_[ np.zeros((self.nPctrl,self.nPctrl+self.nT)),np.eye(self.nPctrl+self.nT) ]
            elif strategy=='part':
                oneHat = np.zeros((self.nCtrl,1+self.nT))
                oneHat[self.nPctrl:self.nSctrl,0] = 1
                oneHat[self.nSctrl:,1:] = np.eye(self.nT)
            elif strategy=='phase':
                if 'nPh1' not in dir(self):
                    self.getLdsPhsIdx()
                oneHat = np.zeros((self.nCtrl,3+self.nT))
                oneHat[self.nPctrl:self.nSctrl,0][self.Ph1] = 1
                oneHat[self.nPctrl:self.nSctrl,1][self.Ph2] = 1
                oneHat[self.nPctrl:self.nSctrl,2][self.Ph3] = 1
                oneHat[self.nSctrl:,3:] = np.eye(self.nT)
            elif strategy in ['maxTap','minTap','nomTap']:
                if self.nT>0:
                    oneHat = np.zeros((self.nCtrl,self.nT))
                    oneHat[-self.nT:] = np.eye(self.nT)
                else:
                    oneHat = np.empty((self.nCtrl,0)) # No taps, with Q=0 fully specified.
        
        if obj=='tsCst':
            if strategy in ['full','loss','load']:
                # oneHat = np.r_[ np.zeros((self.nPctrl,self.nPctrl+self.nT)),np.eye(self.nPctrl+self.nT) ]
                oneHat = np.zeros((self.nCtrl,1+self.nPctrl+self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:self.nSctrl,1:self.nPctrl+1] = np.eye(self.nPctrl)
                oneHat[self.nSctrl:,self.nPctrl+1:] = np.eye(self.nT)
            elif strategy=='part':
                oneHat = np.zeros((self.nCtrl,1+1+self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:self.nSctrl,1] = 1
                oneHat[self.nSctrl:,2:] = np.eye(self.nT)
            elif strategy=='phase':
                if 'nPh1' not in dir(self):
                    self.getLdsPhsIdx()
                oneHat = np.zeros((self.nCtrl,1+3+self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:self.nSctrl,1][self.Ph1] = 1
                oneHat[self.nPctrl:self.nSctrl,2][self.Ph2] = 1
                oneHat[self.nPctrl:self.nSctrl,3][self.Ph3] = 1
                oneHat[self.nSctrl:,4:] = np.eye(self.nT)
            elif strategy in ['maxTap','minTap','nomTap']:
                if self.nT>0:
                    oneHat = np.zeros((self.nCtrl,1+self.nT))
                    oneHat[:self.nPctrl,0] = 1
                    oneHat[-self.nT:,1:] = np.eye(self.nT)
                else:
                    oneHat = np.zeros((self.nCtrl,1)) # No taps, with Q=0 fully specified.
                    oneHat[:self.nPctrl,0] = 1

        if obj in ['hcGen','hcLds']:
            if strategy in ['full']:
                oneHat = np.zeros((self.nCtrl,1 + self.nPctrl + self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:,1:] = np.eye(self.nPctrl + self.nT)
            if strategy in ['part']:
                oneHat = np.zeros((self.nCtrl,1 + 1 + self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:self.nSctrl,1] = 1
                oneHat[self.nSctrl:,2:] = np.eye(self.nT)
            if strategy in ['minTap','maxTap','nomTap']:
                oneHat = np.zeros((self.nCtrl,1 + self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nSctrl:,1:] = np.eye(self.nT)
            if strategy=='phase':
                if 'nPh1' not in dir(self):
                    self.getLdsPhsIdx()
                oneHat = np.zeros((self.nCtrl,3 + 1 + self.nT))
                oneHat[:self.nPctrl,0] = 1
                oneHat[self.nPctrl:self.nSctrl,1][self.Ph1] = 1
                oneHat[self.nPctrl:self.nSctrl,2][self.Ph2] = 1
                oneHat[self.nPctrl:self.nSctrl,3][self.Ph3] = 1
                oneHat[self.nSctrl:,4:] = np.eye(self.nT)
            
        return sparse.csc_matrix(oneHat),x0
    
    def runOptimization(self,H,p,G,h,oneHat,x0,strategy,obj,optType):
        if strategy in ['minTap','maxTap','load']:
            tLss=False
        else:
            tLss=True
        
        print('----->  Opt run, feeder:', self.feeder,' strgy: ',strategy,'; Solver: ',optType)
        if 'cvxopt' in optType:
            Q = matrix(H.dot(H.T)); p = matrix(p); G = matrix(G); h = matrix(h)
            self.sln = solvers.qp(Q,p,G,h)
            self.slnX = oneHat.dot(np.array(self.sln['x']).flatten())               + x0.flatten()
            self.slnS = self.sln['status']
        if 'cvxMosek' in optType:
            Q = matrix(H.dot(H.T)); p = matrix(p); G = matrix(G); h = matrix(h)
            self.sln = solvers.qp(Q,p,G,h,solver='mosek')
            self.slnX = oneHat.dot(np.array(self.sln['x']).flatten())               + x0.flatten()
            self.slnS = self.sln['status']
        if 'mosekNom' in optType:
            self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=False))      + x0.flatten()
            self.slnS = np.nan
        
        if 'mosekInt' in optType:
            try:
                self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=True)) + x0.flatten()
                self.slnS = 's' # success
            except:
                print('---> Integer optimization not working, trying continuous.')
                try:
                    self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=False))      + x0.flatten()
                    self.slnS = 'r' # relaxed success
                except:
                    print('---> Both Optimizations failed, saving null result.')
                    self.slnX = np.zeros(self.nCtrl)
                    self.slnS = 'f' # failed
        if 'mosekFull' in optType:
            try:
                self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=True,tLss=tLss,oneHat=oneHat)) + x0.flatten()
                self.slnS = 's' # success
            except:
                print('---> Integer optimization not working, trying continuous.')
                try:
                    self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=False,tLss=tLss,oneHat=oneHat)) + x0.flatten()
                    self.slnS = 'r' # relaxed success
                except:
                    print('---> All Optimizations failed, saving null (x0) result.')
                    self.slnX = np.zeros(self.nCtrl) + self.x0.flatten()
                    self.slnS = 'f' # failed
        
        # debug mosek
        # self.slnX = oneHat.dot(self.mosekQpEquiv(H,p,G,h,x0,obj,tInt=False,tLss=True,oneHat=oneHat)) + x0.flatten() # for debugging
    
    def runCvrQp(self,strategy='full',obj='opCst',optType=['mosekFull']):
        #       (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #               A*x = b.
        # sol = solvers.qp(Q,p,G,h,A,b)
        # STRATEGIES
        # 1. 'full' Full optimization
        # 2. 'maxTap' maximise all taps
        # 3. 'minTap' minimise all taps with max Q
        # 4. 'part' full optimization, just one P + Q
        # OBJECTIVES
        # 1. 'opCst' for operating cost
        # 2. 'hcGen' for gen HC
        # 3. 'hcLds' for load HC
        # 4 'tsLds' for operating cost with generation
        
        # GET RID OF APPROPRIATE VARIABLES
        self.oneHat,self.x0 = self.remControlVrlbs(strategy,obj)
        x0 = self.x0; oneHat = self.oneHat
        
        if obj in ['opCst','tsCst'] and strategy in ['minTap','maxTap','nomTap'] and self.nT==0:
            self.slnX = x0.flatten(); self.slnS = np.nan
        elif obj in ['hcGen','hcLds'] and strategy in ['loss','load']:
            self.slnX = x0.flatten(); self.slnS = np.nan
        else:
            # OBJECTIVE FUNCTION
            H,p = self.getObjFunc(strategy,obj) # H is of the form HHt = 2*Q
            
            # INEQUALITY CONSTRAINTS
            G,h = self.getInqCns(obj)
            
            # Qp = matrix(  (oneHat.T).dot(Q.dot(oneHat)) )
            # pp = matrix(  (p.T.dot(oneHat)).T  - 2*(  (x0.T).dot(Q.dot(oneHat)).T  ) )
            
            Hp = aMulBsp(H.T,oneHat).T
            pp = aMulBsp(p.T,oneHat).T  - 2*(  ( H.T.dot(x0).T).dot(aMulBsp(H.T,oneHat)).T  )
            
            del(H); del(p)
            Gp0 = aMulBsp(G,oneHat)
            GpNz = (np.sum(Gp0,axis=1)!=0)
            Gp = Gp0[GpNz]
            hp = (h - G.dot(x0))[GpNz]
            
            del(G); del(h); del(GpNz)
            self.runOptimization(Hp,pp,Gp,hp,oneHat,x0,strategy,obj,optType)
        
        self.slnF = self.runQp(self.slnX)
        if len(self.dssStuff)!=4:
            self.initialiseOpenDss()
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.slnD = self.qpDssValidation()
        
    
    def mosekQpEquiv(self,H0,p0,G0,h0,x0,obj,tInt=False,tLss=False,oneHat=[],iRelax=False):
        if type(H0)==matrix:
            H0 = np.array(H0); p0 = np.array(p0); G0 = np.array(G0); h0 = np.array(h0)
        
        nCtrlAct = p0.shape[0]
        nSctrlAct = nCtrlAct - self.nT
        nPctrlAct = np.sum(np.sum( oneHat[:self.nPctrl],axis=0 )!=0)
        nQctrlAct = np.sum(np.sum( oneHat[self.nPctrl:self.nSctrl],axis=0 )!=0)
        
        if p0.ndim==1:
            p0 = np.array([p0]).T
        if h0.ndim==1:
            h0 = np.array([h0]).T
        
        H0norm = np.linalg.norm(H0)
        H = Matrix.dense(H0.T); del(H0)
        p = Matrix.dense(p0); del(p0)
        G = Matrix.dense(G0); del(G0)
        h = Matrix.dense(h0); del(h0)

        with Model('qp') as M:
            if tLss and nQctrlAct>0:
                yP = M.variable("yP",nPctrlAct)
                yQ = M.variable("yQ",nQctrlAct)
                y = Expr.vstack( yP,yQ )
                
                qLossQ = 2*np.diag( oneHat.T.dot( sparse.csc_matrix(np.diag(self.qlossQdiag)).dot(oneHat) ).toarray() )[nPctrlAct:nPctrlAct+nQctrlAct]
                
                qLossH = Matrix.sparse( range(nQctrlAct),range(nQctrlAct),np.sqrt(qLossQ) )
                
                wQ = M.variable("wQ",1,Domain.greaterThan(0.0))
                
                if nQctrlAct==1:
                    M.constraint( "wQ", Expr.hstack(1.0, wQ,Expr.mul(qLossH,yQ)), Domain.inRotatedQCone() )
                else:
                    M.constraint( "wQ", Expr.vstack(1.0, wQ,Expr.mul(qLossH,yQ)), Domain.inRotatedQCone() )
                
                if sum(self.qlossL + self.qlossReg)>0:
                    oneNormLin = self.qlossL + self.qlossReg
                    if nQctrlAct==1:
                        qlossLabs = sum(oneNormLin)
                    if nQctrlAct==self.nPctrl:
                        qlossLabs = max(oneNormLin)
                    if nQctrlAct==3:
                        phaseWeights = aMulBsp(oneNormLin,oneHat)
                        qlossLabs = Matrix.dense( np.diag(phaseWeights[phaseWeights!=0]) )
                    
                    wL = M.variable("wL",nQctrlAct)
                    
                    M.constraint( Expr.sub( Expr.mul(qlossLabs,yQ),wL ),Domain.lessThan(0.0) )
                    M.constraint( Expr.add( Expr.mul(qlossLabs,yQ),wL ),Domain.greaterThan(0.0) )
                    
                    cw = Matrix.dense( np.ones((nQctrlAct + 1,1)) )
                    wCfunc = Expr.dot(cw,Expr.vstack(wL,wQ))
                else:
                    # wCfunc = Expr.dot(1,wQ)
                    wCfunc = wQ
            else:
                y = M.variable("y",nSctrlAct)
                wCfunc = 0
            
            if tInt and self.nT>0: # even having an 'empty' one of these makes this worse if nT=0!
                z = M.variable("z",self.nT,Domain.integral( Domain.inRange(-16,16) ))
                # NB, throws up the following warning for some reason when running, but not when called 'normally': https://stackoverflow.com/questions/52594235/futurewarning-using-a-non-tuple-sequence-for-multidimensional-indexing-is-depre
            else:
                z = M.variable("z",self.nT)
            x = Expr.vstack( y,z )

            # VOLTAGE and DOMAIN constraints:
            Gxh = M.constraint( "Gxh", Expr.mul(G,x), Domain.lessThan(h) )
            
            # CURRENT constraints:
            ii = 0
            a2iCtrl = self.aIxfmr + self.Mc2i.dot(self.X0ctrl + x0.flatten())
            Mc2iCtrl = aMulBsp(self.Mc2i,oneHat)
            if nCtrlAct>1:
                iConScaling = 1.0 # this seesm to give reasonable results for 5 + 0
                # iConScaling = 0.01 # this seesm to give reasonable results for 5 + 0
                for lim in self.iXfmrLims:
                    mc2iCpx = Mc2iCtrl[ii]*iConScaling
                    a2iCpx = a2iCtrl[ii]*iConScaling
                    mCpx = sparse.coo_matrix(  np.r_[ [mc2iCpx.real],[mc2iCpx.imag] ]  )
                    aCpx = np.r_[ a2iCpx.real, a2iCpx.imag].reshape((2,1))
                    
                    mCpx = Matrix.sparse( mCpx.shape[0],mCpx.shape[1],mCpx.row,mCpx.col,mCpx.data )
                    aCpx = Matrix.dense( aCpx )
                    iScaleReq = 1e-3
                    # iScaleReq = 1
                    if np.linalg.norm( mc2iCpx/(lim*self.iScale) )>iScaleReq:
                        iAdd = Expr.add(Expr.mul(mCpx,x),aCpx)
                        M.constraint( "i"+str(ii), Expr.vstack( iConScaling*lim*self.iScale,iAdd ),Domain.inQCone() )
                    ii+=1
            else: # no need for conic constraints if only one decision variable! (Required!)
                if sum(Mc2iCtrl==0)>0:
                    Mc2iCtrlNz = (Mc2iCtrl!=0).flatten()
                    A,b,d = self.cxpaLtK( Mc2iCtrl[Mc2iCtrlNz].flatten(),a2iCtrl[Mc2iCtrlNz].flatten(),self.iXfmrLims[Mc2iCtrlNz]*self.iScale )
                else:
                    A,b,d = self.cxpaLtK( Mc2iCtrl.flatten(),a2iCtrl.flatten(),self.iXfmrLims*self.iScale )
                # plt.plot(A,label='a'); plt.plot(np.abs(b),label='|b|'); plt.plot(d,label='d'); 
                # plt.title('|Ax + b|  < c'); plt.legend(); plt.show()
                A = Matrix.dense( A.reshape((len(A),1)) )
                b = Matrix.dense( b.reshape((len(b),1)) )
                d = Matrix.dense( d.reshape((len(d),1)) )
                
                i0 = Expr.add(Expr.mul(A,x),b)
                M.constraint( "cxpaLtKlb", Expr.add(d,i0),Domain.greaterThan(0.0) )
                M.constraint( "cxpaLtKub", Expr.sub(d,i0),Domain.greaterThan(0.0) )
                del(A); del(b); del(d)
            
            if H0norm!=0:
                # QP constraint https://docs.mosek.com/9.0/pythonfusion/tutorial-model-lib.html
                # NB: for some annoying reason the hstack and vstack need to be used differently depending
                # on if there is one or more than one variable...[?]
                t = M.variable("t",1,Domain.greaterThan(0.0))
                if nCtrlAct==1:
                    M.constraint( "Qt", Expr.hstack(1.0, t,Expr.mul(H0norm,x)), Domain.inRotatedQCone())
                else:
                    M.constraint( "Qt", Expr.vstack(1.0,t,Expr.mul(H,x)), Domain.inRotatedQCone() )
                M.objective("obj",ObjectiveSense.Minimize, Expr.add( Expr.add(Expr.dot(p,x),t),wCfunc ) )
            else:
                M.objective("obj",ObjectiveSense.Minimize, Expr.add( Expr.dot(p,x),wCfunc ) )
            del(Mc2iCtrl); del(oneHat); del(H); del(G); del(p); del(h)
            
            try:
                # FROM: https://docs.mosek.com/9.0/pythonfusion/accessing-solution.html
                # M.setLogHandler(sys.stdout)
                M.solve()
                M.acceptedSolutionStatus(AccSolutionStatus.Optimal)
                slnStatus = M.getPrimalSolutionStatus()
                print('Problem status: ',slnStatus)
                
                WD = os.path.join(os.path.expanduser('~'),'Documents','dump.opf') # debugging
                M.writeTask(WD)
            except Exception as e:
                print("Unexpected error: {0}".format(e))
            
            if tLss and nQctrlAct>0:
                xOut = np.r_[yP.level(),yQ.level(),z.level()]
            else:
                xOut = np.r_[y.level(),z.level()]
            
            print("MIP rel gap = %.2f (%f)" % (M.getSolverDoubleInfo(
                        "mioObjRelGap"), M.getSolverDoubleInfo("mioObjAbsGap")))
        return xOut
    
    def cxpaLtK( self,c,a,k ):
        # c, a are complex
        A = np.abs(c)
        b = ((a*(c.conj())).real)/np.abs(c)
        d = np.sqrt( k**2 - (np.abs(a)**2) + (b**2) )
        # then, |Ax + b| < d.
        return A,b,d
    
    def qpComparisonOpCst(self):
        self.loadQpSet()
        
        modesPlot = ['Nominal','full','part','minTap','maxTap']
        TL0 = self.qpSolutions['loss'][1][0] # get the loss minimization losses
        PL0 = self.qpSolutions['load'][1][1] # get the load minimization load
        
        extraTL = []
        extraPL = []
        costFunc = []
        for modeSet in modesPlot:
            if modeSet=='Nominal':
                self.slnF = self.slnF0
            else:
                self.loadQpSln(modeSet)
            TL,PL,TC = self.slnF[0:3]
            extraTL.append(TL-TL0)
            extraPL.append(PL-PL0)
            costFunc.append(TL + PL + TC)
        
        fig,[ax1,ax0] = plt.subplots(ncols=2)
        ax0.bar(np.arange(len(modesPlot))+0.2,extraTL,label='Extra losses',width=0.2,zorder=10)
        ax0.bar(np.arange(len(modesPlot))-0.2,extraPL,label='Extra load',width=0.2,zorder=10)
        ax0.set_xticks(np.arange(len(modesPlot)))
        ax0.set_xticklabels(modesPlot,rotation=90)
        ax0.set_ylabel('Power (kW)')
        ax0.legend()
        
        ax1.bar(np.arange(len(modesPlot)),costFunc,width=0.2,zorder=10)
        ax1.set_xticks(np.arange(len(modesPlot)))
        ax1.set_xticklabels(modesPlot,rotation=90)
        ax1.set_ylabel('Power (kW)')
        ax1.set_ylim(( np.mean(costFunc)-5*np.std(costFunc),np.mean(costFunc)+5*np.std(costFunc) ))
        plt.tight_layout()
        plt.show()
        
    def qpVarValue(self,strategy='part',obj='opCst',res='',invType=None):
        # res as 'norm' or 'power'
        self.loadQpSet(invType=invType)
        self.loadQpSln(strategy,obj)
        
        if obj in ['opCst','tsCst']:# op cost (kW):
            if res=='norm': val = sum(self.slnF[0:4])/sum(self.slnF0[0:4])
            if res=='power': val = sum(self.slnF[0:4]) - sum(self.slnF0[0:4])
        if obj=='hcGen':# HC gen (kW):
            if res=='norm': val = ( -sum(self.slnF[0:4]) - -sum(self.slnF0[0:4]) )/( sum(self.slnF0[0:4])/self.linPoint )
            if res=='power': val = -sum(self.slnF[0:4]) - -sum(self.slnF0[0:4])
        if obj=='hcLds': # loadability (kW):
            if res=='norm': val = self.slnF[2]/( sum(self.slnF0[0:4])/self.linPoint )
            if res=='power': val = self.slnF[2]
        return val
    
    def snapQpComparison(self):
        self.runCvrQp('loss')
        TL0 = self.slnF[0]
        self.runCvrQp('load')
        PL0 = self.slnF[1]
        
        modes = ['maxTap','minTap','part','full','load','loss']
        extraTL = []
        extraPL = []
        extraTC = []
        costFunc = []
        self.status = {}
        for mode in modes:
            print('Running QP '+mode)
            self.runCvrQp(mode)
            TL,PL,TC = self.slnF[0:3]
            extraTL.append(TL-TL0)
            extraPL.append(PL-PL0)
            extraTC.append(TC)
            costFunc.append(TL + PL + TC)
            self.status[mode]=self.sln['status']
        
        print(self.status)
        fig,[ax1,ax0] = plt.subplots(ncols=2)
        ax0.bar(np.arange(len(modes))-0.25,extraTL,label='Extra losses',width=0.2,zorder=10)
        ax0.bar(np.arange(len(modes)),extraPL,label='Extra load',width=0.2,zorder=10)
        ax0.bar(np.arange(len(modes))+0.25,extraTC,label='Curtailment',width=0.2,zorder=10)
        ax0.set_xticks(np.arange(len(modes)))
        ax0.set_xticklabels(modes,rotation=90)
        ax0.set_ylabel('Power (kW)')
        # ax0.grid()
        ax0.legend()
        
        ax1.bar(np.arange(len(modes)),costFunc,width=0.2,zorder=10)
        ax1.set_xticks(np.arange(len(modes)))
        ax1.set_xticklabels(modes,rotation=90)
        ax1.set_ylabel('Power (kW)')
        ax1.set_ylim(( np.mean(costFunc)-5*np.std(costFunc),np.mean(costFunc)+5*np.std(costFunc) ))
        # ax1.grid()
        plt.tight_layout()
        plt.show()
    
    def qpDssValidation(self,slnX=None,method=None):
        # SIDE EFFECT: enables all capacitor controls.
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        ctrlModel = DSSSolution.ControlMode
        
        self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        
        genNames = self.addYDgens(DSSObj)
        if slnX is None:
            slnX = self.slnX
        
        self.setDssSlnX(genNames,slnX=slnX,method=method)
        
        TG,TL,PL,YNodeV = runCircuit(DSSCircuit,DSSSolution)[1::]
        iXfrm = abs( self.v2iBrYxfmr.dot(YNodeV)[self.iXfmrModelled] )
        
        if method=='relaxT':
            DSSText.Command = 'batchedit capcontrol..* enabled=True'
            DSSSolution.ControlMode = ctrlModel # return control to nominal state
            print('Old tap pos:',slnX[self.nSctrl:] + np.array(self.TC_No0))
            print('New tap pos:',find_tap_pos(DSSCircuit))
        CL = self.getInverterLosses(slnX)
        iXfrmC = self.v2iBrYxfmr.dot(YNodeV)[self.iXfmrModelled]
        outSet = [TL,PL,-TG,CL,abs(YNodeV)[3:],iXfrm,YNodeV[3:],iXfrmC] # as in runQp
        return outSet
    
    def getInverterLosses(self,slnX=None,onLoss=True):
        if slnX is None:
            slnX = self.slnX
        
        qLossQ = self.qlossQdiag.dot(slnX**2)
        qLossL = np.linalg.norm(self.qlossL*slnX,ord=1)
        if onLoss:
            qLossC = self.qlossC.dot( np.abs(slnX)>self.qlossCzero )
        else:
            qLossC = 0
        return qLossQ + qLossL + qLossC
        
    def qpSolutionDssError(self,strategy,obj,err='V',load=True):
        # WARNING!!! BY DEFAULT this will load an old solution!!!!
        if load:
            self.loadQpSet()
            self.loadQpSln(strategy,obj)
        
        # TL,PL,TC,CL,V,I,Vc,Ic = self.slnF
        if err=='V':
            gmaV = 1e-5
            # gmaV = 0
            
            dVset = np.abs(self.slnF[4] - self.slnF0[4])
            dVsetTrue = np.abs(self.slnD[4] - self.slnD0[4])
            dssError = np.linalg.norm( (dVset - dVsetTrue)/( dVsetTrue + gmaV*self.vKvbase) )/np.sqrt(len(dVset))
        elif err=='I':
            gmaI = 1e-3
            # gmaI = 0
        
            dIset = np.abs(self.slnF[7] - self.slnF0[7])/(self.iXfmrLims*self.iScale)
            dIsetTrue = np.abs(self.slnD[7] - self.slnD0[7])/(self.iXfmrLims*self.iScale)
            
            dssError = np.linalg.norm( (dIset - dIsetTrue)/(dIsetTrue + gmaI) )/np.sqrt(len(dIset))
        elif err=='P':
            gmaP = 1e-4
            # gmaP = 0
            dPset = abs( np.sum(self.slnF[0:4]) -  np.sum(self.slnF0[0:4]) )
            dPsetTrue = abs( np.sum(self.slnD[0:4]) -  np.sum(self.slnD0[0:4]) )
            
            dssError = abs( (dPset - dPsetTrue )/( dPsetTrue + gmaP ) )
        return dssError
    
    def tsRecordSnap(self,slnTs,idxs,slnX=None,slnF=None):
        if slnX is None:
            slnX = self.slnX
        if slnF is None:
            slnF = self.slnF
        
        V = slnF[4]/self.vKvbase
        vKvbase = self.vKvbase[V>0.5]
        V = V[V>0.5]
        slnTs['vMin'][idxs] = np.min(V)
        slnTs['vMinMv'][idxs] = np.min(V[vKvbase>1000])
        slnTs['vMinLv'][idxs] = np.min(V[vKvbase<1000])
        slnTs['vMax'][idxs] = np.max(V)
        slnTs['tPwr'][idxs] = sum(slnF[0:4])
        slnTs['tLss'][idxs] = slnF[0]
        slnTs['tSet'][idxs] = np.mean(slnX[self.nSctrl:] + np.array(self.TC_No0))
        return slnTs
        
    
    def setDssSlnX(self,genNames,slnX=None,method=None):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        if slnX is None:
            slnX = self.slnX
        
        slnP = slnX[:self.nPctrl]*self.lScale/self.xSscale
        slnQ = slnX[self.nPctrl:self.nSctrl]*self.lScale/self.xSscale
        
        setGenPq(DSSCircuit,genNames,slnP,slnQ)
        # if method is None:
        # set at nominal taps first in all cases
        slnT = np.round(slnX[self.nSctrl:] + np.array(self.TC_No0)).astype(int).tolist()
        fix_tap_pos(DSSCircuit,slnT)
            
        if method=='relaxT':
            # SIDE EFFECT: Disables all cap controls.
            slnF = self.runQp(slnX)
            vOut = slnF[4]
            regIdx = np.array(get_regIdx(DSSCircuit)[0])-3
            Vrto = getRegBwVrto(DSSCircuit)[1]
            
            if len(regIdx)>0:
                regV = vOut[regIdx]/np.array(Vrto)
                
                # BW = 0.2*( self.vKvbase[regIdx]/np.array(Vrto) )*0.1/16
                BW = 0.1*( self.vKvbase[regIdx]/np.array(Vrto) )*0.1/16
                
                RGC = DSSCircuit.RegControls           
                i = RGC.First
                while i:
                    RGC.ForwardVreg = regV[i-1]
                    RGC.ForwardBand = BW[i-1]
                    RGC.ReverseVreg = regV[i-1]
                    RGC.ReverseBand = BW[i-1]
                    RGC.ForwardR=0
                    RGC.ReverseR=0
                    RGC.ForwardX=0
                    RGC.ReverseX=0
                    i = RGC.Next
            DSSText.Command='Set controlmode=static'
            DSSText.Command = 'batchedit capcontrol..* enabled=false'
    
    def printQpSln(self,slnX=None,slnF=None):
        if slnX is None:
            slnX = self.slnX
        if slnF is None:
            slnF = self.slnF
        pOut = slnX[:self.nPctrl]
        qOut = slnX[self.nPctrl:self.nPctrl*2]
        tOut = slnX[self.nPctrl*2:]
        
        TL,PL,TC,CL,V,I,Vc,Ic = slnF
        
        print('\n================== QP Solution:')
        print('Loss (kW):',TL)
        print('Load (kW):',PL)
        print('Curtailment (kW):',TC)
        print('Total power (kW):', PL + TL + TC)
        print('\nTotal converter losses (kW):', CL)
        print('Total power + CL (kW):', PL + TL + TC + CL)
        print('\n ||Q|| (1 norm, kVAr):', np.linalg.norm(qOut,ord=1)*self.lScale/self.xSscale)
        print('Tap Position:', tOut + np.array(self.TC_No0))
    
    def showQpSln(self,slnX=None,slnF=None,ctrlPerPhase=True):
        if slnX is None:
            slnX = self.slnX
        if slnF is None:
            if 'slnF' in dir(self):
                slnF = self.slnF
            else:
                slnF = self.runQp(slnX)
        
        self.printQpSln(slnX,slnF)
        pOut = slnX[:self.nPctrl]
        qOut = slnX[self.nPctrl:self.nPctrl*2]
        tOut = slnX[self.nPctrl*2:]
        
        TL,PL,TC,CL,V,I,Vc,Ic = slnF
        TL0,PL0,TC0,CL0,V0,I0,Vc0,Ic0 = self.slnF0
        if 'slnD' not in dir(self):
            self.initialiseOpenDss()
            slnD = self.qpDssValidation(slnX)
        else:
            slnD = self.slnD
        TLd,PLd,TCd,CLd,Vd,Id,Vcd,Icd = slnD
        
        iDrn = (-1)**( np.abs(np.angle(Ic/Ic0))>np.pi/2 )
        
        fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(11,4))
        
        # plot voltages versus voltage limits
        ax0.plot((Vd/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markeredgewidth=0.7);
        ax0.plot((V/self.vKvbase)[self.vIn],'x',markeredgewidth=0.7);
        ax0.plot((V0/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=3.0);
        ax0.plot((self.vHi/self.vKvbase)[self.vIn],'k_');
        ax0.plot((self.vLo/self.vKvbase)[self.vIn],'k_');
        ax0.set_title('Voltages')
        ax0.set_xlabel('Bus Index')
        ax0.set_ylabel('Voltage, pu')
        ax0.grid(True)
        # ax0.show()
        
        # plot currents versus current limits
        ax1.plot(iDrn*Id/(self.iScale*self.iXfmrLims),'o',label='OpenDSS',markerfacecolor='None')
        ax1.plot(iDrn*abs(Ic/(self.iScale*self.iXfmrLims)),'x',label='QP sln')
        ax1.plot(abs(Ic0)/(self.iScale*self.iXfmrLims),'o',label='Nominal',markerfacecolor='None',markersize=3.0)
        ax1.plot(np.ones(len(self.iXfmrLims)),'k_')
        ax1.plot(-np.ones(len(self.iXfmrLims)),'k_')
        ax1.set_xlabel('Xfmr Index')
        ax1.set_ylabel('Current, frac of iXfmr')
        ax1.set_title('Currents')
        ax1.legend()
        ax1.set_ylim( (-1.1,1.1) )
        ax1.grid(True)
        # ax1.show()
        
        # nPh1 = int(sum(phs1)); nPh2 = int(sum(phs2)); nPh3 = int(sum(phs3)); 
        self.getLdsPhsIdx()
        if ctrlPerPhase:
            ax2.plot(range(0,self.nPh1),
                                100*(slnX[self.nPctrl:self.nPctrl*2][self.Ph1])/self.qLim,'x-',label='Qgen A (%)')
            ax2.plot(range(self.nPh1,self.nPh1+self.nPh2),
                                100*slnX[self.nPctrl:self.nPctrl*2][self.Ph2]/self.qLim,'x-',label='Qgen B (%)')
            ax2.plot(range(self.nPh1+self.nPh2,self.nPctrl),
                                100*slnX[self.nPctrl:self.nPctrl*2][self.Ph3]/self.qLim,'x-',label='Qgen C (%)')
        else:
            ax2.plot(range(self.nPctrl),
                                100*slnX[self.nPctrl:self.nPctrl*2]/self.qLim,'x-',label='Qgen (%)')
        ax2.plot(range(self.nPctrl,self.nPctrl + self.nT),100*slnX[self.nPctrl*2:]/self.tLim,'x-',label='t (%)')
        ax2.set_xlabel('Control Index')
        ax2.set_ylabel('Control effort, %')
        ax2.set_title('Control settings')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim((-110,110))
        plt.tight_layout()
        plt.show()
        
    def plotArcy(self,slnX=None,slnF=None,ctrlPerPhase=True,pltShow=True):
        if slnX is None:
            slnX = self.slnX
        if slnF is None:
            slnF = self.slnF
        
        self.printQpSln(slnX,slnF)
        pOut = slnX[:self.nPctrl]
        qOut = slnX[self.nPctrl:self.nPctrl*2]
        tOut = slnX[self.nPctrl*2:]
        
        TL,PL,TC,CL,V,I,Vc,Ic = slnF
        TL0,PL0,TC0,CL0,V0,I0,Vc0,Ic0 = self.slnF0
        TLd,PLd,TCd,CLd,Vd,Id,Vcd,Icd = self.slnD
        
        iDrn = (-1)**( np.abs(np.angle(Ic/Ic0))>np.pi/2 )
        
        fig,[ax0,ax1] = plt.subplots(ncols=2,figsize=(5.5,2.6))
        
        # plot voltages versus voltage limits
        Vdpu =  ((V0 - Vd)/self.vKvbase)[self.vIn]
        Vpu =  ((V0 - V)/self.vKvbase)[self.vIn]
        
        nSortV = np.argsort(Vpu)
        ax0.plot(Vdpu[nSortV],label='$\Delta V_{\mathrm{DSS.}}$')
        ax0.plot(Vpu[nSortV],label='$\Delta V_{\mathrm{QP}}$')
        ax0.legend()
        ax0.set_xlabel('Bus Index')
        ax0.set_ylabel('Bus Voltage change, pu')
        ax0.grid(True)
        
        # plot currents versus current limits
        ax1.plot(100*abs(abs(Id) - abs(Ic0))/(self.iScale*self.iXfmrLims),'o',label='$\Delta \\,I_{\mathrm{DSS.}}$',markerfacecolor='None')
        ax1.plot(100*abs(abs(Ic) - abs(Ic0))/(self.iScale*self.iXfmrLims),'o',label='$\Delta \\,I_{\mathrm{QP}}$',markerfacecolor='None',markersize=3.0)
        
        ax1.set_xlabel('Branch Index')
        ax1.set_ylabel('Branch Current \nchange, % of $I_{\mathrm{max}}$')
        ax1.legend()
        ax1.grid()
        
        plt.tight_layout()
        if pltShow: plt.show()

        
    def getLdsPhsIdx(self):
        phs1 = np.zeros(self.nPctrl,dtype=int)
        phs2 = np.zeros(self.nPctrl,dtype=int)
        phs3 = np.zeros(self.nPctrl,dtype=int)
        ii=0
        for ld in self.SyYNodeOrder + self.SdYNodeOrder:
            phs1[ii] = ld[-1]=='1'
            phs2[ii] = ld[-1]=='2'
            phs3[ii] = ld[-1]=='3'
            ii+=1
        self.nPh1 = int(sum(phs1))
        self.nPh2 = int(sum(phs2))
        self.nPh3 = int(sum(phs3))
        
        self.Ph1 = np.where(phs1)
        self.Ph2 = np.where(phs2)
        self.Ph3 = np.where(phs3)
    
    def testQpVcpf(self,k=np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        
        print('Start CVR testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        ice=np.zeros([k.size])
        iae=np.zeros([k.size])
        
        Convrg = []
        
        DSSSolution.LoadMult=1.0
        DSSSolution.Solve
        iRglr = np.linalg.norm( self.v2iBrYxfmr.dot(tp_2_ar(DSSCircuit.YNodeVarray) )[self.iXfmrModelled]/self.iXfmrLims )
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged) # for debugging
            
            ic0 = self.v2iBrYxfmr.dot(tp_2_ar(DSSCircuit.YNodeVarray))[self.iXfmrModelled]
            ia0 = abs(ic0)
            
            vc0 = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            va0 = abs(vc0)
            
            dx = (self.X0ctrl*k[i]/self.currentLinPoint) - self.X0ctrl
            
            vaL,iaL,vcL,icL = self.runQp(dx)[4:]
            iaL = abs(iaL) # correct for any negatives
            # iaL = abs(icL)
            # icL = self.v2iBrYxfmr.dot(np.r_[self.V0,vcL])[self.iXfmrModelled]
            # icL = self.Mc2i(dx)
            
            vce[i] = np.linalg.norm( (vcL - vc0)/self.vKvbase )/np.linalg.norm(vc0/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL - va0)/self.vKvbase )/np.linalg.norm(va0/self.vKvbase)
            
            # ice[i] = np.linalg.norm( (icL - ic0)/self.iXfmrLims )/iRglr # this stops issues around zero
            # iae[i] = np.linalg.norm( (iaL - ia0)/self.iXfmrLims )/iRglr # this stops issues around zero
            ice[i] = np.linalg.norm( (icL - ic0)/self.iXfmrLims )/np.sqrt(len(self.iXfmrLims))
            iae[i] = np.linalg.norm( (iaL - ia0)/self.iXfmrLims )/np.sqrt(len(self.iXfmrLims))
        print('nrelModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        
        fig,[ax0,ax1] = plt.subplots(ncols=2,figsize=(8,3.5))
        
        ax0.plot(k,vce.real,label='vce');
        ax0.plot(k,vae,label='vae');
        ax0.set_xlabel('Continuation factor k');ax0.grid(True)
        ax0.set_title('Voltage error, '+self.feeder)
        ax0.legend()
        ax1.plot(k,ice,label='ice');
        ax1.plot(k,iae,label='iae');
        ax1.set_xlabel('Continuation factor k');ax1.grid(True)
        ax1.set_title('Current error, '+self.feeder)
        ax1.legend()
        
        return vce,vae,k,ice,iae
        
    def testGenSetting(self,k=np.arange(-10,11),dPlim=0.01,dQlim=0.01):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        
        print('Start CVR testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        ice=np.zeros([k.size])
        iae=np.zeros([k.size])
        tle=np.zeros([k.size])
        ple=np.zeros([k.size])
        tce=np.zeros([k.size])
        
        Convrg = []
        
        DSSSolution.LoadMult=1.0
        DSSSolution.Solve
        iRglr = np.linalg.norm( self.v2iBrYxfmr.dot(tp_2_ar(DSSCircuit.YNodeVarray) ) )
        TLrglr = 1e-3*DSSCircuit.Losses[0]
        PLrglr = -(TLrglr + DSSCircuit.TotalPower[0])
        
        dx = np.r_[ self.pLim*np.ones(self.nPctrl)*dPlim, self.qLim*np.ones(self.nPctrl)*dQlim,np.zeros(self.nT) ]
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            slnX = np.zeros(self.nCtrl) + dx*k[i]
            
            # set DSSCircuit, then solve and get values;
            TL0,PL0,TC0,CL0,va0,ia0,vc0,ic0 = self.qpDssValidation(slnX,full=True)
            TLL,PLL,TCL,CLL,vaL,iaL,vcL,icL = self.runQp(slnX)
            
            vce[i] = np.linalg.norm( (vcL - vc0)/self.vKvbase )/np.linalg.norm(vc0/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL - va0)/self.vKvbase )/np.linalg.norm(va0/self.vKvbase)
            
            ice[i] = np.linalg.norm( (icL - ic0) )/iRglr # this stops issues around zero
            iae[i] = np.linalg.norm( (iaL - ia0) )/iRglr # this stops issues around zero

            tle[i] = np.linalg.norm( (TLL - TL0) )/TLrglr # this stops issues around zero
            ple[i] = np.linalg.norm( (PLL - PL0) )/PLrglr # this stops issues around zero
            tce[i] = np.linalg.norm( (TCL - TC0) )/TLrglr # this stops issues around zero
        # print('nrelModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        
        fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(12,3.5))
        
        ax0.plot(k,vce.real,label='vce');
        ax0.plot(k,vae,label='vae');
        ax0.set_xlabel('k point');ax0.grid(True)
        ax0.set_title('Voltage error, '+self.feeder)
        ax0.legend()
        ax1.plot(k,ice.real,label='ice');
        ax1.plot(k,iae,label='iae');
        ax1.set_xlabel('k point');ax1.grid(True)
        ax1.set_title('Current error, '+self.feeder)
        ax1.legend()
        ax2.plot(k,tle.real,label='tle');
        ax2.plot(k,ple,label='ple');
        ax2.plot(k,tce,label='tce');
        ax2.set_xlabel('k point');ax2.grid(True)
        ax2.set_title('Loss and load error, '+self.feeder)
        ax2.legend()
        
        return vce,vae,k
    def testCvrQp(self):
        # THREE TESTS in terms of the sensitivity of the model to real power generation, reactive power, and tap changes.
        print('Start CVR QP testing. \n',time.process_time())
        
        
        # do complex voltages over loadmult first.
        self.testQpVcpf()
        # Then: Taps.
        if self.nT>0:
            self.testQpTcpf()
        
        self.testQpScpf()
        
    
    def testQpScpf(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        # Test 1. Putting taps up and down one. Things to check:
        # - voltages; currents; loads; losses; generation
        # Test 2. Put a whole load of generators in and change real and reactive powers.
        TLboth = []
        TLestBoth = []
        PLboth = []
        PLestBoth = []
        vErrBoth = []
        SsetBoth = []
        iErrBoth = []
        iBoth = []
        icBoth = []
        iaBoth = []
        
        for ii in range(2):
            self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
            genNames = self.addYDgens(DSSObj)
            
            xCtrl = np.zeros(self.nCtrl)
            
            if ii==0:
                Sset = np.linspace(-1,1,31)*self.qLim
                if self.nT>0:
                    xCtrl[self.nPctrl:-self.nT] = 1
                else:
                    xCtrl[self.nPctrl:] = 1
            elif ii==1:
                Sset = np.linspace(-1,1,30)*self.pLim # NB: This changes with the units.
                xCtrl[:self.nPctrl] = 1
            
            TL = np.zeros(len(Sset))
            TLest = np.zeros(len(Sset))
            PL = np.zeros(len(Sset))
            PLest = np.zeros(len(Sset))
            TC = np.zeros(len(Sset))
            TCest = np.zeros(len(Sset))
            vErr = np.zeros(len(Sset))
            iErr = np.zeros(len(Sset))
            iCalcNorm = np.zeros(len(Sset))
            icCalcNorm = np.zeros(len(Sset))
            iaCalcNorm = np.zeros(len(Sset))
            
            for i in range(len(Sset)):
                if ii==0:
                    setGenPq(DSSCircuit,genNames,np.zeros(self.nPctrl),np.ones(self.nPctrl)*Sset[i]*self.lScale/self.xSscale)
                elif ii==1:
                    setGenPq(DSSCircuit,genNames,np.ones(self.nPctrl)*Sset[i]*self.lScale/self.xSscale,np.zeros(self.nPctrl))
                TG,TL[i],PL[i],YNodeV = runCircuit(DSSCircuit,DSSSolution)[1::]
                absYNodeV = abs(YNodeV[3:])
                Icalc = (self.v2iBrYxfmr.dot(YNodeV))[self.iXfmrModelled]
                TC[i] = -TG
                
                dx = xCtrl*Sset[i]
                TLest[i],PLest[i],TCest[i],CL,Vest,Iest,Vcest,Icest = self.runQp(dx)
                
                vErr[i] = np.linalg.norm(absYNodeV - Vest)/np.linalg.norm(absYNodeV)
                iErr[i] = np.linalg.norm((Icalc - Icest)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
                
                iCalcNorm[i] = np.linalg.norm(np.abs(Icalc)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
                icCalcNorm[i] = np.linalg.norm(np.abs(Icest)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
                iaCalcNorm[i] = np.linalg.norm(Iest/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
            
            
            if ii==0:
                xlbl = 'Reactive power per load (kVar)'
            elif ii==1:
                xlbl = 'Real power per load (kW)'
            
            fig,[ax0,ax1,ax2,ax3,ax4] = plt.subplots(ncols=5,figsize=(11,5))
            ax0.plot(Sset*self.lScale/self.xSscale,TL,label='dss'); ax0.grid(True)
            ax0.plot(Sset*self.lScale/self.xSscale,TLest,label='apx')
            ax0.set_title('Losses (kW)'); ax0.set_xlabel(xlbl)
            ax0.legend()
            ax1.plot(Sset*self.lScale/self.xSscale,PL); ax1.grid(True)
            ax1.plot(Sset*self.lScale/self.xSscale,PLest)
            ax1.set_title('Load power (kW)'); ax1.set_xlabel(xlbl)
            ax2.plot(Sset*self.lScale/self.xSscale,TC); ax2.grid(True)
            ax2.plot(Sset*self.lScale/self.xSscale,TCest)
            ax2.set_title('Curtailment (kW)'); ax2.set_xlabel(xlbl)
            ax3.plot(Sset*self.lScale/self.xSscale,vErr); ax3.grid(True)
            ax3.set_title('Abs voltage error'); ax3.set_xlabel(xlbl)
            ax4.plot(Sset*self.lScale/self.xSscale,iErr); ax4.grid(True)
            ax4.set_title('Abs current error'); ax4.set_xlabel(xlbl)
            plt.tight_layout()
            
            TLboth.append(TL)
            TLestBoth.append(TLest)
            PLboth.append(PL)
            PLestBoth.append(PLest)
            vErrBoth.append(vErr)
            iErrBoth.append(iErr)
            SsetBoth.append(Sset)
            iBoth.append(iCalcNorm)
            icBoth.append(icCalcNorm)
            iaBoth.append(iaCalcNorm)
        # plt.show()
        
        return TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,iErrBoth,SsetBoth,iBoth,icBoth,iaBoth
    
    
    def testQpTcpf(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        nStt = 6
        tapChng = np.array([-nStt]+(nStt*2*[1])+(nStt*2*[-1]))
        dxScale = np.cumsum(tapChng)*self.tLim/16
        TL = np.zeros(len(tapChng))
        TLest = np.zeros(len(tapChng))
        TLcalc = np.zeros(len(tapChng))
        PL = np.zeros(len(tapChng))
        PLest = np.zeros(len(tapChng))
        TC = np.zeros(len(tapChng))
        TCest = np.zeros(len(tapChng))
        vErr = np.zeros(len(tapChng))
        iErr = np.zeros(len(tapChng))
        iCalcNorm = np.zeros(len(tapChng))
        icCalcNorm = np.zeros(len(tapChng))
        iaCalcNorm = np.zeros(len(tapChng))
        DSSText.Command='Set controlmode=off'
        
        xCtrl = np.zeros(self.nCtrl)
        xCtrl[-self.nT::] = 1
        
        for i in range(len(tapChng)):
            # set all of the taps at one above
            j = DSSCircuit.RegControls.First
            while j:
                tapNo = DSSCircuit.RegControls.TapNumber
                if abs(tapNo)==16:
                    print('Sodding taps are saturated!')
                DSSCircuit.RegControls.TapNumber = tapChng[i].item()+tapNo
                j = DSSCircuit.RegControls.Next
            
            TG,TL[i],PL[i],YNodeV = runCircuit(DSSCircuit,DSSSolution)[1::]
            absYNodeV = abs(YNodeV[3:])
            # Icalc = abs(self.v2iBrYxfmr.dot(YNodeV))[self.iXfmrModelled]
            Icalc = (self.v2iBrYxfmr.dot(YNodeV))[self.iXfmrModelled]
            TC[i] = -TG
            
            dx = xCtrl*dxScale[i]
            TLest[i],PLest[i],TCest[i],CL,Vest,Iest,Vcest,Icest = self.runQp(dx)
            
            vErr[i] = np.linalg.norm(absYNodeV - Vest)/np.linalg.norm(absYNodeV)
            iErr[i] = np.linalg.norm((Icalc - Icest)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
            
            iCalcNorm[i] = np.linalg.norm(np.abs(Icalc)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
            icCalcNorm[i] = np.linalg.norm(np.abs(Icest)/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
            iaCalcNorm[i] = np.linalg.norm(Iest/(self.iXfmrLims*self.iScale))/np.sqrt(len(Icalc))
            
            # VcplxEst = self.Mc2v.dot(self.X0ctrl + dx) + self.aV
            # YNodeVaug = np.concatenate((YNodeV[:3],VcplxEst,np.array([0])))
            # iOut = self.v2iBrY.dot(YNodeVaug[:-1])
            # vBusOut=YNodeVaug[list(self.yzW2V)]
            # TLcalc[i] = sum(np.delete(1e-3*vBusOut*iOut.conj(),self.wregIdxs).real) # for debugging
        
        fig,[ax0,ax1,ax2,ax3,ax4] = plt.subplots(ncols=5,figsize=(11,5))
        ax0.plot(dxScale,TL,label='dss'); ax0.grid(True)
        ax0.plot(dxScale,TLest,label='apx')
        ax0.set_title('Losses'); ax0.set_xlabel('Tap (pu)')
        ax0.legend()
        ax1.plot(dxScale,PL); ax1.grid(True)
        ax1.plot(dxScale,PLest)
        ax1.set_title('Load power'); ax1.set_xlabel('Tap (pu)')
        ax2.plot(dxScale,TC); ax2.grid(True)
        ax2.plot(dxScale,TCest); ax2.grid(True)
        ax2.set_title('Curtailment'); ax2.set_xlabel('Tap (pu)')
        ax3.plot(dxScale,vErr); ax3.grid(True)
        ax3.set_title('Abs voltage error'); ax3.set_xlabel('Tap (pu)')
        ax4.plot(dxScale,iErr); ax4.grid(True)
        ax4.set_title('Abs current error'); ax4.set_xlabel('Tap (pu)')
        plt.tight_layout()
        # plt.show()
        
        return TL,TLest,PL,PLest,vErr,iErr,dxScale,iCalcNorm,icCalcNorm,iaCalcNorm

    
    def addYDgens(self,DSSObj):
        genNamesY = add_generators(DSSObj,self.SyYNodeOrder,False)
        genNamesD = add_generators(DSSObj,self.SdYNodeOrder,True)
        return genNamesY + genNamesD
    
    def solveQpUnc(self,paSet=None):
        # paSet should be tuple, if used (pLoad,aCvr).
        # based on runQp.
        if 'qQpUnc' not in dir(self):
            H = self.getHmat()
            self.qQpUnc = (H.dot(H.T))
        
        Q = self.qQpUnc[self.nPctrl:self.nSctrl,self.nPctrl:self.nSctrl] + np.diag(self.qlossQdiag[self.nPctrl:self.nSctrl])
        Qpq = self.qQpUnc[:self.nPctrl,self.nPctrl:self.nSctrl]
        
        # pLoad0 = self.ploadL[self.nPctrl:self.nSctrl]/self.pCvr
        pLoad0 = self.ploadL[self.nPctrl:self.nSctrl]/self.pCvr
        pLoss0 = (self.X0ctrl[:self.nPctrl]/self.linPoint).dot(Qpq)
        
        if paSet is None:
            pLoad = self.ploadL[self.nPctrl:self.nSctrl]
            pLoss = self.qpLlss[self.nPctrl:self.nSctrl]
        else:
            Pctrl = self.X0ctrl[:self.nPctrl]*( (paSet[0]-self.linPoint)/self.linPoint )
            pLoss = 2*Pctrl.dot(Qpq) + self.qpLlss[self.nPctrl:self.nSctrl]
            pLoad = (paSet[1]/self.pCvr)*self.ploadL[self.nPctrl:self.nSctrl]*( paSet[0]/self.linPoint )
            
        p = pLoad + pLoss
        # xStar = np.linalg.solve(Q,-0.5*p)
        xStar = np.r_[np.zeros(self.nPctrl),np.linalg.solve(Q,-0.5*p),np.zeros(self.nT)]
        
        # aNonlin = 1.5*pLoad0.dot( np.linalg.solve(Q,pLoad0) ) # this does not seem to work well.
        # pNonlin = pLoss0.dot( np.linalg.solve(Q,pLoss0) ) # this is NOT correct!
        
        # fStar = xStar.dot(Q).dot(xStar) + p.dot(xStar)
        # fStar = self.runQp(np.r_[np.zeros(len(xStar)),xStar,np.zeros(self.nT)])
        fStar = self.runQp(xStar)
        
        gradP = 1e3*np.linalg.norm(p,ord=1)/p.shape[0]
        pInf = 1e3*np.linalg.norm(p,ord=np.inf)
        
        return xStar, fStar, gradP, pInf
    
    
    def runQp(self,dx,onLoss=True):
        
        if 'Mc2v' in dir(self):
            Vcest = self.Mc2v.dot(dx + self.X0ctrl) + self.aV
        else:
            Vcest = np.nan
        
        if 'Kc2i' in dir(self): # can be created with self.recreateKc2i if wanted.
            Iest = self.Kc2i.dot(dx + self.X0ctrl) + self.bW
        else:
            Iest = np.nan
        
        if 'qpHlss' not in dir(self):
            self.qpHlss = self.getHmat()
        
        # dx needs to be in units of kW, kVAr, tap no.
        # TL = dx.dot(self.qpQlss.dot(dx) + self.qpLlss) + self.qpClss
        TL = np.linalg.norm(self.qpHlss.T.dot(dx))**2 + self.qpLlss.dot(dx) + self.qpClss
        PL = dx.dot(self.ploadL) + self.ploadC
        TC = dx.dot(self.pcurtL)
        Vest = self.Kc2v.dot(dx + self.X0ctrl) + self.bV
        Icest = self.Mc2i.dot(dx + self.X0ctrl) + self.aIxfmr
        
        CL = self.getInverterLosses(dx,onLoss=onLoss)
        
        return TL,PL,TC,CL,Vest,Iest,Vcest,Icest

    def createNrelModel(self,lin_point=1.0):
        print('\nCreate NREL model, feeder:',self.feeder,'\nLin Point:',lin_point,'\nCap pos model:',self.setCapsModel,'\nCap Pos points:',self.capPosLin,'\n',time.process_time())
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=1 status=variable'
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = lin_point
        DSSSolution.Solve()
        print('\nNominally converged (createNrelModel):',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        self.Cap_No0 = getCapPos(DSSCircuit)
        if self.capPosLin==None:
            self.capPosLin = self.Cap_No0
        print(self.Cap_No0)
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        self.YbusN2 = Ybus.shape[0]**2
        self.YbusNnz = Ybus.nnz
        self.YbusFrac = self.YbusNnz/(self.YbusN2)
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        self.loadDssModel(loadMult=lin_point)
        self.vKvbase = get_Yvbase(DSSCircuit)[3:]
        self.xY, self.xD, self.pyIdx, self.pdIdx,  = ldValsOnly( DSSCircuit ) # NB these do not change with the circuit!
        
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        if len(self.pdIdx[0])>0:
            H = create_Hmat(DSSCircuit)[self.pdIdx[0]+3]
        else:
            H = []
        
        # sY,sD,iY,iD,yzD,iTot,Hold = get_sYsD(DSSCircuit) # useful for debugging
        # self.xhy0 = -1e3*s_2_x(sY[3:])
        # self.xhd0 = -1e3*s_2_x(sD)
        # fig,[ax0,ax1] = plt.subplots(2)
        # ax0.plot((self.xY-self.xhy0))
        # ax1.plot((self.xD-self.xhd0)); 
        # plt.show()
        
        self.YZ = DSSCircuit.YNodeOrder[3:]
        # plt.plot(self.xY); plt.plot(xhy0); plt.show()
        # plt.plot((self.xY-xhy0)/xhy0); plt.show()
        
        # >>> 3. Create linear models for voltage in S
        V0 = YNodeV[0:3]
        Vh = YNodeV[3:]
        DSSSolution.LoadMult=0.0
        DSSSolution.Solve()
        VnoLoad = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
        
        if len(H)==0:
            print('Create linear models My:\n',time.process_time());  t = time.time()
            My,a = nrel_linearization_My( Ybus,Vh,V0 )
            self.Mtime = time.time() - t
            print('Time M:',self.Mtime,'\nCreate linear models Ky:\n',time.process_time()); t = time.time()
            Ky,b = nrelLinKy(My,Vh,self.xY*lin_point)
            print('Time K:',time.time()-t)
            Md = np.zeros((len(Vh),0), dtype=complex); Kd = np.zeros((len(Vh),0))
        else:
            print('Create linear models My + Md:\n',time.process_time()); t = time.time()
            My,Md,a = nrel_linearization( Ybus,Vh,V0,H )
            self.Mtime = time.time() - t
            print('Time M:',self.Mtime,'\nCreate linear models Ky + Kd:\n',time.process_time()); t = time.time()
            Ky,Kd,b = nrelLinK(My,Md,Vh,self.xY*lin_point,self.xD*lin_point)
            print('Time K:',time.time()-t)
        
        Vh0 = (My.dot(self.xY) + Md.dot(self.xD))*lin_point + a # for validation
        Va0 = (Ky.dot(self.xY) + Kd.dot(self.xD))*lin_point + b # for validation
        
        print('\nVoltage clx error (lin point), Volts:',np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh)) # Print checks
        print('Voltage clx error (no load point), Volts:',np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad),'\n') # Print checks
        print('\nVoltage abs error (lin point), Volts:',np.linalg.norm(Va0-abs(Vh))/np.linalg.norm(abs(Vh))) # Print checks
        print('Voltage abs error (no load point), Volts:',np.linalg.norm(abs(b)-abs(VnoLoad))/np.linalg.norm(abs(VnoLoad)),'\n') # Print checks
        
        # self.syIdx = self.xY.nonzero()[0]
        self.syIdx = np.concatenate((self.pyIdx[0],self.pyIdx[0]+DSSCircuit.NumNodes-3))
        self.My = My[:,self.syIdx]
        self.Ky = Ky[:,self.syIdx]
        self.aV = a
        self.bV = b
        self.H = H
        self.V0 = V0
        self.nV = len(a)
        
        self.vYNodeOrder = YNodeOrder[3:]
        self.SyYNodeOrder = vecSlc(self.vYNodeOrder,self.pyIdx)
        self.SdYNodeOrder = vecSlc(self.vYNodeOrder,self.pdIdx)
        self.Md = Md
        self.Kd = Kd
        self.currentLinPoint = lin_point
        self.Ybus = Ybus
    
    def createCvrModel(self,lin_point=None):
        # NOTE: opendss and the documentation have different definitions of CVR factor (!) one uses linear, one uses exponential. Results suggest tt seems to be an exponential model...?
        
        if lin_point==None:
            linPoint = self.linPoint
        else:
            linPoint = lin_point
        
        pCvr = self.pCvr
        qCvr = self.qCvr
        
        print('\nCreate NREL model, feeder:',self.feeder,'\nLin Point:',linPoint,'\npCvr, qCvr:',pCvr,qCvr,'\nCap Pos points:',self.capPosLin,'\n',time.process_time())
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(pCvr)+' cvrvars='+str(qCvr)
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = linPoint
        DSSSolution.Solve()
        print('\nNominally converged (createCvrModel):',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        self.loadCvrDssModel(pCvr,qCvr,loadMult=linPoint)
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        
        self.xY, self.xD, self.pyIdx, self.pdIdx  = ldValsOnly( DSSCircuit ) # NB these do not change with the circuit!
        
        self.vKvbase = get_Yvbase(DSSCircuit)[3:]
        self.vKvbaseY = self.vKvbase[self.pyIdx[0]]
        self.vKvbaseD = self.vKvbase[self.pdIdx[0]]*np.sqrt(3)
        self.vKvbaseX = np.concatenate( (self.vKvbaseY,self.vKvbaseY,self.vKvbaseD,self.vKvbaseD) )
        
        if len(self.pdIdx[0])>0:
            H = create_Hmat(DSSCircuit)[self.pdIdx[0]+3]
        else:
            H = np.zeros((0,Ybus.shape[0]))
        
        # # Only consider single phase loads for now.
        VhYpu = abs(YNodeV[3:]/self.vKvbase)
        VhDpu = abs(H.dot(YNodeV)/self.vKvbaseD)
        self.kYcvr = np.concatenate((VhYpu**pCvr,VhYpu**qCvr))
        self.kDcvr = np.concatenate((VhDpu**pCvr,VhDpu**qCvr))
        
        # xYcvr = linPoint*self.xY*self.kYcvr
        # xDcvr = linPoint*self.xD*self.kDcvr
        # [sY0,sD0] = get_sYsD(DSSCircuit)[0:2] # useful for debugging
        # xY0 = -1e3*s_2_x(sY0[3:])
        # xD0 = -1e3*s_2_x(sD0)
        # err0 = xY0[:len(xYcvr)//2] - xYcvr[:len(xYcvr)//2] # ignore Q for now (coz of caps)
        # err1 = xY0[:len(xYcvr)//2] - self.xY[:len(xYcvr)//2]
        # err0d = xD0[:len(xDcvr)//2] - xDcvr[:len(xDcvr)//2]
        # err1d = xD0[:len(xDcvr)//2] - self.xD[:len(xDcvr)//2]
        # fig,[ax0,ax1] = plt.subplots(ncols=2)
        # ax0.plot(err0); ax0.plot(err1);
        # ax1.plot(err0d); ax1.plot(err1d); plt.show()
        
        # sY,sD,iY,iD,yzD,iTot,Hold = get_sYsD(DSSCircuit) # useful for debugging ... ?
        # self.xhy0 = -1e3*s_2_x(sY[3:])
        # self.xhd0 = -1e3*s_2_x(sD)
        # fig,[ax0,ax1] = plt.subplots(2)
        # ax0.plot((self.xY-self.xhy0))
        # ax1.plot((self.xD-self.xhd0)); 
        # plt.show()
        
        self.YZ = DSSCircuit.YNodeOrder[3:]
        
        # >>> 3. Create linear models for voltage in S
        V0 = YNodeV[0:3]
        Vh = YNodeV[3:]
        dVh = H.dot(YNodeV)
        DSSSolution.LoadMult=0.0
        DSSSolution.Solve()
        VoffLoad = tp_2_ar(DSSCircuit.YNodeVarray) 
        VnoLoad = VoffLoad[3:]
        dVnoLoad = H.dot(VoffLoad)

        
        print('Create linear models My + Md:\n',time.process_time()); t = time.time()
        My,Md,a,dMy,dMd,da = cvrLinearization( Ybus,Vh,V0,H,pCvr,qCvr,self.vKvbase,self.vKvbaseD )
        
        print('Time M:',time.time()-t,'\nCreate linear models Ky + Kd:\n',time.process_time()); t = time.time()
        Ky,Kd,b = nrelLinK( My,Md,Vh,linPoint*self.xY,linPoint*self.xD )
        
        dKy,dKd,db = nrelLinK( dMy,dMd,dVh,linPoint*self.xY,linPoint*self.xD )
        
        print('Time K:',time.time()-t)
        
        Vh0 = (My.dot(self.xY) + Md.dot(self.xD))*linPoint + a # for validation
        Va0 = (Ky.dot(self.xY) + Kd.dot(self.xD))*linPoint + b # for validation
        dVh0 = (dMy.dot(self.xY) + dMd.dot(self.xD))*linPoint + da # for validation
        dVa0 = (dKy.dot(self.xY) + dKd.dot(self.xD))*linPoint + db # for validation
        
        # Print checks:
        print('\nVoltage clx error (lin point), Volts:',np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh))
        print('Voltage clx error (no load point), Volts:',np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad),'\n') 
        print('\nVoltage abs error (lin point), Volts:',np.linalg.norm(Va0-abs(Vh))/np.linalg.norm(abs(Vh))) 
        print('Voltage abs error (no load point), Volts:',np.linalg.norm(abs(b)-abs(VnoLoad))/np.linalg.norm(abs(VnoLoad)),'\n') 
        print('\n Delta voltage clx error (lin point), Volts:',np.linalg.norm(dVh0-dVh)/np.linalg.norm(dVh)) 
        print('Delta voltage clx error (no load point), Volts:',np.linalg.norm(da-dVnoLoad)/np.linalg.norm(dVnoLoad),'\n') 
        print('\nDelta voltage abs error (lin point), Volts:',np.linalg.norm(dVa0-abs(dVh))/np.linalg.norm(abs(dVh))) 
        print('Delta voltage abs error (no load point), Volts:',np.linalg.norm(abs(db)-abs(dVnoLoad))/np.linalg.norm(abs(dVnoLoad)),'\n')
        
        # self.syIdx = self.xY.nonzero()[0]
        self.syIdx = np.concatenate((self.pyIdx[0],self.pyIdx[0]+DSSCircuit.NumNodes-3))
        self.My = My[:,self.syIdx]
        self.Ky = Ky[:,self.syIdx]
        self.aV = a
        self.bV = b
        self.H = H
        self.V0 = V0
        self.nV = len(a)
        
        self.vYNodeOrder = YNodeOrder[3:]
        self.SyYNodeOrder = vecSlc(self.vYNodeOrder,self.pyIdx)
        self.SdYNodeOrder = vecSlc(self.vYNodeOrder,self.pdIdx)
        self.Md = Md
        self.Kd = Kd
        self.currentLinPoint = lin_point
        
        self.n2y = node_to_YZ(DSSObj.ActiveCircuit)
        
        self.dMy = dMy[:,self.syIdx]
        self.dMd = dMd
        self.daV = da
        
        self.dKy = dKy[:,self.syIdx]
        self.dKd = dKd
        self.dbV = db
        
        self.nxYp = len(self.xY)//2
        self.nxDp = len(self.xD)//2
        self.nYp = self.My.shape[1]//2
        self.nDp = self.Md.shape[1]//2
        self.nx = 2*(self.nYp + self.nDp)
        
        # BUILD the quadradtic matrices for finding the total load.
        iH = np.zeros( (self.nx,self.nx) )
        iH[range(self.nYp),range(self.nYp)] = 1
        iH[range(self.nYp*2,self.nYp*2+self.nDp),range(self.nYp*2,self.nYp*2+self.nDp)] = 1
        
        self.KtotPu = np.block( [[dsf.vmM(1/self.vKvbase,self.Ky),dsf.vmM(1/self.vKvbase,self.Kd)],[dsf.vmM(1/self.vKvbaseD,self.dKy),dsf.vmM(1/self.vKvbaseD,self.dKd)]] )
        # self.KtotPu = np.block( [[dsf.vmM(1/self.vKvbase,self.Ky)]] )
        
        self.yKPu = np.block( [[dsf.vmM(1/self.vKvbase,self.Ky),dsf.vmM(1/self.vKvbase,self.Kd)]] )
        self.dKPu = np.block( [[dsf.vmM(1/self.vKvbaseD,self.dKy),dsf.vmM(1/self.vKvbaseD,self.dKd)]] )
        # self.yKPu = np.block( [[dsf.vmM(1/self.vKvbase,self.Ky)]] )
        # self.dKPu = np.block( [[dsf.vmM(1/self.vKvbaseD,self.dKy)]] )
        
        self.bTotPu = np.concatenate( (self.bV/self.vKvbase,self.dbV/self.vKvbaseD) )
        self.bYpu = self.bV/self.vKvbase
        self.bDpu = self.dbV/self.vKvbaseD
        
        self.kYcvrK = np.concatenate( (self.pCvr*self.yKPu,self.qCvr*self.yKPu))
        self.kYcvr0 = np.concatenate(( (1-self.pCvr) + self.pCvr*self.bYpu,(1-self.qCvr) + self.qCvr*self.bYpu))
        
        self.kDcvrK = np.concatenate( (self.pCvr*self.dKPu,self.qCvr*self.dKPu))
        self.kDcvr0 = np.concatenate(( (1-self.pCvr) + self.pCvr*self.bDpu,(1-self.qCvr) + self.qCvr*self.bDpu))
        
        # self.qpQ = iH.dot(np.concatenate( (self.kYcvrK[self.syIdx],self.kDcvrK) )) # kCvr K
        self.qpQ = iH.dot(np.concatenate( (self.kYcvrK[self.syIdx],self.kDcvrK) )) # kCvr K
        # self.qpQ = 0.5*(self.qpQ + self.qpQ.T)
        self.qpL = iH.dot(np.concatenate( (self.kYcvr0[self.syIdx],self.kDcvr0) )) # kCvr0 offset
        
    
    def loadDssModel(self,loadMult=1.0):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile ('+self.fn+')'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=1 status=variable'
        fix_tap_pos(DSSCircuit, self.TC_No0)
        # fix_cap_pos(DSSCircuit, self.capPosLin)
        fix_cap_pos(DSSCircuit, self.Cap_No0)
        DSSText.Command='Set controlmode=off'
        DSSSolution.LoadMult = loadMult
        DSSSolution.Solve()
    
    def loadCvrDssModel(self,pCvr,qCvr,loadMult=1.0):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile ('+self.fn+')'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(pCvr)+' cvrvars='+str(qCvr)
        fix_tap_pos(DSSCircuit, self.TC_No0)
        # fix_cap_pos(DSSCircuit, self.capPosLin)
        fix_cap_pos(DSSCircuit, self.Cap_No0)
        DSSText.Command='Set controlmode=off'
        DSSSolution.LoadMult = loadMult
        DSSSolution.Solve()
        
    
    def createWmodel(self,linPoint=None):
        # >>> 4. For regulation problems
        if linPoint==None:
            self.loadDssModel(loadMult=self.linPoint)
        else:
            self.loadDssModel(loadMult=self.currentLinPoint)
            
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # BUILD all branch model.
        branchNames = getBranchNames(DSSCircuit)
        print('Build Yprimmat',time.process_time())
        YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,branchNames)
        print('Build v2iBrY',time.process_time())
        v2iBrY = getV2iBrY(DSSCircuit,YprimMat,WbusSet)
        print('Complete',time.process_time())
        
        Wy = v2iBrY[:,3:].dot(self.My)
        Wd = v2iBrY[:,3:].dot(self.Md)
        Wt = v2iBrY[:,3:].dot(self.Mt)
        aI = v2iBrY.dot(np.concatenate((self.V0,self.aV)))
        self.v2iBrY = v2iBrY
        self.Wy = Wy
        self.Wd = Wd
        self.Wt = Wt
        self.aI = aI
        self.WunqIdent = WunqIdent
        self.WbusSet = WbusSet
        self.WbrchSet = WbrchSet
        self.WtrmlSet = WtrmlSet
        self.yzW2V = getYzW2V(self.WbusSet,DSSCircuit.YNodeOrder)
        
        regXfmrs = get_regXfmr(DSSCircuit)
        wregIdxs = np.array([],dtype=int)
        for reg in regXfmrs:
            wregIdxs = np.concatenate((wregIdxs,np.where(np.array(self.WbrchSet)=='Transformer.'+reg)[0]))
        
        self.wregIdxs = wregIdxs
        WbusSetRed = np.delete(np.array(self.WbusSet),self.wregIdxs)
        
        # YZred = tuple(np.delete(np.array(DSSCircuit.YNodeOrder),self.wregIdxs))
        # print(YZred)
        # self.yzW2Vred = getYzW2V(WbusSetRed,YZred)
        self.yzW2Vred = getYzW2V(WbusSetRed,DSSCircuit.YNodeOrder)
        
        # Regulator model, for LTC control:
        regWlineIdx,regIdx = getRegWlineIdx(DSSCircuit,self.WbusSet,self.WtrmlSet)
        WyReg = Wy[regWlineIdx,:]
        WdReg = Wd[regWlineIdx,:]
        aIreg = aI[list(regWlineIdx)]
        
        WregBus = vecSlc(WunqIdent,np.array(regWlineIdx))
        
        self.WyReg = WyReg
        self.aIreg = aIreg
        self.WregBus = WregBus
        self.WdReg = WdReg
        self.WregIdx = regIdx
        
        
        # Vh0 = self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD) + self.aV # for validation
        # IprimReg = WyReg.dot(self.xY[self.syIdx]) + WdReg.dot(self.xD) + aIreg # for debugging
        # Iprim = Wy.dot(self.xY[self.syIdx]) + Wd.dot(self.xD) + aI # for debugging
        # Iprim0 = v2iBrY.dot(np.concatenate((self.V0,Vh0))) # for debugging
        # # printBrI(WregBus,IprimReg) # for debugging. Note: there seem to be some differences between python and opendss native.
        
        # Tranformer model, for thermal limits:
        xfrmNames = getBranchNames(DSSCircuit,xfmrSet=True)
        YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,xfrmNames)
        
        v2iBrYxfmr = getV2iBrY(DSSCircuit,YprimMat,WbusSet) 
        self.v2iBrYxfmr = v2iBrYxfmr
        
        WyXfmr = v2iBrYxfmr[:,3:].dot(self.My)
        WdXfmr = v2iBrYxfmr[:,3:].dot(self.Md)
        WtXfmr = v2iBrYxfmr[:,3:].dot(self.Mt)
        aIxfmr = v2iBrYxfmr.dot(np.concatenate((self.V0,self.aV)))
        
        TRN = DSSCircuit.Transformers
        ACE = DSSCircuit.ActiveElement
        i = TRN.First
        xmfrImaxSet = {}
        while i:
            xmfrImaxSet[TRN.Name] = []
            for j in range(TRN.NumWindings):
                TRN.Wdg=str(j+1)
                kva = TRN.kva
                kv = TRN.kV
                nPhases = ACE.NumPhases
                # xmfrImaxSet[TRN.Name].append(kva/(kv/np.sqrt(nPhases)))
                xmfrImaxSet[TRN.Name].append( kva/( nPhases*(kv/np.sqrt(nPhases)) ) )
            i = TRN.Next
        
        iLims = np.zeros(len(WyXfmr))
        
        for i in range(len(WbrchSet)):
            j = WtrmlSet[i]
            if not WbusSet[i][-1]=='0':
                iLims[i] = xmfrImaxSet[WbrchSet[i].split('.')[-1]][j]
        
        
        
        modelled = (iLims!=0)
        self.WyXfmr = WyXfmr[modelled]
        self.WdXfmr = WdXfmr[modelled]
        self.WtXfmr = WtXfmr[modelled]
        self.aIxfmr = aIxfmr[modelled]
        self.iXfmrLims = iLims[modelled]
        self.iXfmrModelled = modelled
        
        self.WbrchXfmrSet = vecSlc(WbrchSet,np.where(modelled)[0])
        self.WunqXfmrIdent = vecSlc(WunqIdent,np.where(modelled)[0])
    
    def createTapModel(self,linPoint=None,cvrModel=False):
        if cvrModel:
            if linPoint==None:
                self.loadCvrDssModel(loadMult=self.linPoint,pCvr=self.pCvr,qCvr=self.qCvr)
            else:
                self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        else:
            if linPoint==None:
                self.loadDssModel(loadMult=self.linPoint)
            else:
                self.loadDssModel(loadMult=self.currentLinPoint)
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        # self.loadDssModel(loadMult=linPoint) # <--- should this be here (!) (NO...!)
        
        j = DSSCircuit.RegControls.First
        dVdt = np.zeros((self.nV,DSSCircuit.RegControls.Count))
        dVdt_cplx = np.zeros((DSSCircuit.NumNodes - 3,DSSCircuit.RegControls.Count),dtype=complex)
        
        while j!=0:
            tap0 = DSSCircuit.RegControls.TapNumber
            if abs(tap0)<16:
                tap_hi = tap0+1; tap_lo=tap0-1
                dt = 2*0.00625
            elif tap0==16:
                tap_hi = tap0; tap_lo=tap0-1
                dt = 0.00625
            else:
                tap_hi = tap0+1; tap_lo=tap0
                dt = 0.00625
            DSSCircuit.RegControls.TapNumber = tap_hi
            DSSSolution.Solve()
            V1 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:]) # NOT the same order as AllBusVmag!
            V1_cplx = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            DSSCircuit.RegControls.TapNumber = tap_lo
            DSSSolution.Solve()
            V0 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:])
            V0_cplx = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            # dVdt[:,j-1] = (V1 - V0)/(dt*Yvbase)
            dVdt[:,j-1] = (V1 - V0)/(dt)
            dVdt_cplx[:,j-1] = (V1_cplx - V0_cplx)/(dt)
            DSSCircuit.RegControls.TapNumber = tap0
            j = DSSCircuit.RegControls.Next
        self.Kt = dVdt
        self.Mt = dVdt_cplx
        # Wt = v2iBrY[:,3:].dot(dVdt_cplx)
        # WtReg = v2iBrY[regWlineIdx,3:].dot(dVdt_cplx)

    def createFxdModel(self,linPoint=None):
        if linPoint==None:
            self.loadDssModel(loadMult=self.linPoint)
        else:
            self.loadDssModel(loadMult=self.currentLinPoint)
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
    
        if DSSCircuit.RegControls.Count:
            regIdx,regBus = get_regIdx(DSSCircuit)
            self.reIdx = (np.array(get_reIdx(regIdx,DSSCircuit.NumNodes)[3:])-3).tolist()
            self.regVreg = get_regVreg(DSSCircuit)
            
            self.Afxd, self.Bfxd = lmKronRed(self,self.reIdx,self.regVreg)
        else:
            print('No fxd model (no regulators).')
    
    def createLtcModel(self,linPoint=None):
        if linPoint==None:
            self.loadDssModel(loadMult=self.linPoint)
        else:
            self.loadDssModel(loadMult=self.currentLinPoint)
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
    
        if DSSCircuit.RegControls.Count:
            # regIdx,regBus = get_regIdx(DSSCircuit)
            # self.reIdx = (np.array(get_reIdx(regIdx,DSSCircuit.NumNodes)[3:])-3).tolist()
            # self.regVreg = get_regVreg(DSSCircuit)
            rReg,xReg = getRxVltsMat(DSSCircuit)
            Rreg = np.diag(rReg)
            Xreg = np.diag(xReg)
            
            regIdx = get_regIdx(DSSCircuit)[0]
            Vprim = tp_2_ar(DSSCircuit.YNodeVarray)[regIdx]
            VprimAng = Vprim/abs(Vprim)
            
            
            Vprim = tp_2_ar(DSSCircuit.YNodeVarray)[regIdx]
            VprimAng = Vprim/abs(Vprim)
            
            WyRot = dsf.vmM(VprimAng.conj(),self.WyReg)
            WdRot = dsf.vmM(VprimAng.conj(),self.WdReg)
            
            # WyS = -dsf.vmM(abs(Vprim),WyRot).conj()[:,s_idx_shf]
            # WdS = -dsf.vmM(abs(Vprim),WdRot).conj()[:,sD_idx_shf]
            self.WyS = -dsf.vmM(abs(Vprim),WyRot).conj()
            self.WdS = -dsf.vmM(abs(Vprim),WdRot).conj()
            
            # regIdxMatYs = WyS[:,0:len(xhy0)//2].real
            # regIdxMatDs = WdS[:,0:len(xhd0)//2].real
            regIdxMatYs = self.WyS[:,0:len(self.pyIdx[0])].real
            regIdxMatDs = self.WdS[:,0:len(self.pdIdx[0])].real
            
            self.regIdxMatVlts = -np.concatenate( (Rreg.dot(regIdxMatYs),Xreg.dot(regIdxMatYs),Rreg.dot(regIdxMatDs),Xreg.dot(regIdxMatDs)),axis=1 )
            
            self.Rreg =Rreg
            self.Xreg =Xreg
            
            self.Altc,self.Bltc = lmLtcKronRed(self,self.reIdx,self.regVreg,self.regIdxMatVlts)
            # self.Altc,self.Bltc = kron_red_ltc(KyR,KdR,KtR,bVR,regVreg,)
            # ,self.KvReg
        else:
            print('No fxd model (no regulators).')
        
    def testVoltageModel(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadDssModel()
        
        print('Start nrel testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        vc0 = np.zeros((len(k),self.nV),dtype=complex)
        va0 = np.zeros((len(k),self.nV))
        vcL = np.zeros((len(k),self.nV),dtype=complex)
        vaL = np.zeros((len(k),self.nV))
        Convrg = []
        TP = np.zeros((len(k)),dtype=complex)
        
        dM = self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD)
        dK = self.Ky.dot(self.xY[self.syIdx]) + self.Kd.dot(self.xD)
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged) # for debugging
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1] # for debugging
            
            vc0[i] = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            va0[i] = abs(vc0[i])
            
            vcL[i] = dM*k[i] + self.aV
            vaL[i] = dK*k[i] + self.bV
            
            vce[i] = np.linalg.norm( (vcL[i] - vc0[i])/self.vKvbase )/np.linalg.norm(vc0[i]/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL[i] - va0[i])/self.vKvbase )/np.linalg.norm(va0[i]/self.vKvbase)
        
        
        fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
        ax0.plot(k,abs(vce)); ax0.grid(True)
        ax0.set_ylabel('Vc,e')
        ax1.plot(k,vae); ax1.grid(True)
        ax1.set_ylabel('Va,e')
        ax1.set_xlabel('Continuation factor k')
        ax0.set_title(self.feeder)
        plt.tight_layout()
        plt.show()
        
        print('testVoltageModel, converged:',100*sum(Convrg)/len(Convrg),'%')
        return vce,vae,k

    def testCvrModel(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadCvrDssModel(pCvr=self.pCvr,qCvr=self.qCvr,loadMult=self.linPoint)
        
        print('Start CVR testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        dvce=np.zeros([k.size])
        dvae=np.zeros([k.size])
        TP = np.zeros((len(k)),dtype=complex)
        TLss = np.zeros((len(k)),dtype=complex)
        TL = np.zeros((len(k)),dtype=complex)
        TLq = np.zeros((len(k)),dtype=complex)
        TLp = np.zeros([k.size])
        TLa = np.zeros([k.size])
        TL0 = np.zeros([k.size])
        Convrg = []
        
        X0 = np.concatenate((self.xY[self.syIdx],self.xD))
        
        # NB: the makeCvrModel() is different from makeQpModel().
        dM = self.My.dot((self.xY)[self.syIdx]) + self.Md.dot(self.xD)
        dK = self.Ky.dot((self.xY)[self.syIdx]) + self.Kd.dot(self.xD)
        ddM = self.dMy.dot((self.xY)[self.syIdx]) + self.dMd.dot(self.xD)
        ddK = self.dKy.dot((self.xY)[self.syIdx]) + self.dKd.dot(self.xD)
        
        dKtotPu = self.KtotPu.dot(X0)
        
        # lin_point*self.xY*self.kYcvr
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged) # for debugging
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1] # for debugging
            TLss[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1]) # for debugging
            TL[i] = TP[i] + TLss[i]
            
            vc0 = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            va0 = abs(vc0)
            dvc0 = self.H[:,3:].dot(vc0)
            dva0 = abs(dvc0)
            
            X = k[i]*np.concatenate((self.xY[self.syIdx],self.xD))
            TLq[i] = 1e-3*( X.dot(self.qpL + self.qpQ.dot(X)) )
            
            vcL = dM*k[i] + self.aV
            vaL = dK*k[i] + self.bV
            dvcL = ddM*k[i] + self.daV
            dvaL = ddK*k[i] + self.dbV
            
            # vYpu = abs(vcL/self.vKvbase)
            # vDpu = abs(dvcL/self.vKvbaseD)
            vYpu = vaL/self.vKvbase
            vDpu = dvaL/self.vKvbaseD
            
            # kYcvr = np.concatenate((vYpu**self.pCvr,vYpu**self.qCvr))
            # kDcvr = np.concatenate((vDpu**self.pCvr,vDpu**self.qCvr))
            kYcvr = np.concatenate(( (1-self.pCvr) + self.pCvr*vYpu,(1-self.qCvr) + self.qCvr*vYpu))
            kDcvr = np.concatenate(( (1-self.pCvr) + self.pCvr*vDpu,(1-self.qCvr) + self.qCvr*vDpu))
            
            xYpred = k[i]*self.xY*kYcvr
            xDpred = k[i]*self.xD*kDcvr
            
            xYactl,xDactl = returnXyXd(DSSCircuit,self.n2y)
            pActl = np.concatenate((xYactl[:len(xYactl)//2],xDactl[:len(xDactl)//2])) # ignore Q for now (coz of caps)
            
            pPred = np.concatenate((xYpred[:len(xYpred)//2],xDpred[:len(xDpred)//2]))
            p0pred = k[i]*np.concatenate((self.xY[:len(self.xY)//2],self.xD[:len(self.xD)//2]))
            
            vce[i] = np.linalg.norm( (vcL - vc0)/self.vKvbase )/np.linalg.norm(vc0/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL - va0)/self.vKvbase )/np.linalg.norm(va0/self.vKvbase)
            dvce[i] = np.linalg.norm( (dvcL - dvc0)/self.vKvbaseD )/np.linalg.norm(dvc0/self.vKvbaseD)
            dvae[i] = np.linalg.norm( (dvaL - dva0)/self.vKvbaseD )/np.linalg.norm(dva0/self.vKvbaseD)
            
            TLa[i] = sum(pActl)
            TLp[i] = sum(pPred)
            TL0[i] = sum(1e-3*p0pred)
            
        print('nrelModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        
        fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(11,4))
        
        ax0.plot(k,vce.real,label='vce');
        ax0.plot(k,vae,label='vae');
        ax0.set_xlabel('Continuation factor k');ax0.grid(True)
        ax0.set_title('Voltage error, '+self.feeder)
        ax0.legend()
        
        ax1.plot(k,dvce,label='dvce');
        ax1.plot(k,dvae,label='dvae');
        ax1.set_xlabel('Continuation factor k');ax1.grid(True)
        ax1.set_title('Voltage error, '+self.feeder)
        ax1.legend()
        
        ax2.plot(k,-TL.real,label='ap=0.75 DSS');
        ax2.plot(k,-TLq,label='ap=0.75 QP');
        ax2.set_xlabel('Continuation factor k');ax2.grid(True)
        ax2.set_title('Load, '+self.feeder)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # vce,vae,dvce,dvae,TLcvr,TLcvrq,TLa,TLp,TLcvr0,kN = self.testCvrModel(k=np.linspace(-0.5,1.2,18))
        return vce,vae,dvce,dvae,TL,TLq,TLa,TLp,TL0,k
        
    def testWmodel(self,k = np.arange(-1.5,1.6,0.1),cvrModel=False):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        if cvrModel:
            self.loadCvrDssModel(pCvr=self.pCvr,qCvr=self.qCvr)
        else:
            self.loadDssModel()
        
        print('Start W-model testing. \n',time.process_time())
        ice=np.zeros([k.size]) # current error
        vce=np.zeros([k.size]) # voltage error
        
        Convrg = []
        TLout = np.zeros((len(k)),dtype=complex)
        TLest = np.zeros((len(k)),dtype=complex)
        TLest0 = np.zeros((len(k)),dtype=complex)
        
        
        dMsl = np.concatenate((np.zeros(3,dtype=complex),self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD),np.zeros(1,dtype=complex)))[list(self.yzW2V)]
        aVsl = np.concatenate((self.V0,self.aV,np.array([0])))[list(self.yzW2V)]
        
        dW = self.Wy.dot(self.xY[self.syIdx]) + self.Wd.dot(self.xD)
        
        X0 = np.concatenate((self.xY[self.syIdx],self.xD))
        
        # M = np.concatenate((np.zeros((3,len(X0)),dtype=complex),self.My,self.Md,np.zeros((1,len(X0)),dtype=complex)),axis=1)
        M = np.block([[np.zeros((3,len(X0)),dtype=complex)],[np.concatenate((self.My,self.Md),axis=1)],[np.zeros((1,len(X0)),dtype=complex)]])
        Wcnj = np.concatenate((self.Wy,self.Wd),axis=1).conj()
        
        aV = np.concatenate((self.V0,self.aV,np.array([0])))
        aIcnj = (self.aI).conj()
        
        P = np.zeros((len(self.yzW2V),len(M)))
        P[range(len(self.yzW2V)),self.yzW2V] = 1
        PT = P.T
        self.qpQlss = np.real( (M.T).dot(PT.dot(Wcnj)) )
        self.qpLlss = np.real( aV.dot(PT.dot(Wcnj)) + aIcnj.dot(P.dot(M)) )
        self.qpClss = np.real( aV.dot(PT.dot(aIcnj)) )
        
        print('Min eigenvalue: ',np.min(np.linalg.eigvals(self.qpQlss)))
        # print('Norm:',np.linalg.norm(self.qpQlss))
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            # TLout[i] = (DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1]) # in Watts
            TLout[i] = DSSCircuit.Losses[0] # in Watts
            
            vOut = np.concatenate((tp_2_ar(DSSCircuit.YNodeVarray),np.array([0])))
            iOut = self.v2iBrY.dot(vOut[:-1])
            vBusOut=vOut[list(self.yzW2V)]
            # TLest[i] = vBusOut.dot(iOut.conj()) # for debugging
            
            # now do the same with estimated quantities.
            iEst = dW*k[i] + self.aI
            vBusEst = dMsl*k[i] + aVsl
            TLest0[i] = vBusEst.dot(iEst.conj()) # in Watts
            
            X0k =k[i]*X0
            TLest[i] = self.qpClss + X0k.dot( self.qpLlss + self.qpQlss.dot(X0k) )
            
            
            
            TLerr = abs(TLout - TLest)/abs(TLout)
            
            ice[i] = np.linalg.norm( (iOut - iEst) )/np.linalg.norm(iOut)
            vce[i] = np.linalg.norm( (vBusEst - vBusOut) )/np.linalg.norm(vBusOut)
        print('testWmodel, converged:',100*sum(Convrg)/len(Convrg),'%')
        # print('testWmodel, QP correct:',np.linalg.norm(TLest0 - TLest)/np.linalg.norm(TLest0),'%')
        
        # plot
        fig,[ax0,ax1] = plt.subplots(ncols=2,sharey=True)
        fig,ax0 = plt.subplots()
        ax0.plot(k,1e-3*TLout,label='DSS')
        ax0.plot(k,1e-3*TLest,label='QP'); ax0.grid(True)
        ax0.set_ylabel('Power (kW)')
        ax0.set_xlabel('Continuation factor k')
        ax0.legend()
        ax0.set_title('Real Power Loss, '+self.feeder)
        plt.tight_layout()
        plt.show()        
        
        # Loss/current errors
        fig,[ax0,ax1,ax2] = plt.subplots(3,figsize=(4,8),sharex=True)
        ax0.plot(k,TLerr);
        ax0.grid(True)
        ax0.set_ylabel('Loss Error')
        ax0.set_xlabel('Continuation factor k')
        ax1.plot(k,ice);
        ax1.grid(True)
        ax1.set_ylabel('Current Error')
        ax1.set_xlabel('Continuation factor k')
        ax2.plot(k,vce);
        ax2.grid(True)
        ax2.set_ylabel('Vprim error')
        ax2.set_xlabel('Continuation factor k')
        plt.tight_layout()
        plt.show()
        
        # TL,TLcalc,TLerr,ice,vceI,k = self.testWmodel(k=np.linspace(-0.5,1.2,100),cvrModel=True)
        return TLout,TLest,TLerr,ice,vce,k
        
    def testFxdModel(self,k = np.arange(-1.5,1.6,0.1),cvrModel=False):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        if cvrModel:
            self.loadCvrDssModel(pCvr=self.pCvr,qCvr=self.qCvr)
        else:
            self.loadDssModel()
        
        print('Start fxd testing. \n',time.process_time())
        
        dF = self.Afxd.dot(np.concatenate((self.xY[self.syIdx],self.xD)))
        dK = self.Ky.dot(self.xY[self.syIdx]) + self.Kd.dot(self.xD)
        
        vFxdeLck = np.zeros(len(k))
        vLckeLck = np.zeros(len(k))
        vFxdeFxd = np.zeros(len(k))
        vLckeFxd = np.zeros(len(k))
        
        Convrg = []
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            vOut = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][self.reIdx]
            
            vFxd = dF*k[i] + self.Bfxd
            vLck = (dK*k[i] + self.bV)[self.reIdx]
            
            vFxdeLck[i] = np.linalg.norm( vOut - vFxd )/np.linalg.norm(vOut)
            vLckeLck[i] = np.linalg.norm( vOut - vLck )/np.linalg.norm(vOut)
            
        DSSText.Command='set controlmode=static'
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            vOut = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][self.reIdx]
            
            vFxd = dF*k[i] + self.Bfxd
            vLck = (dK*k[i] + self.bV)[self.reIdx]
            
            vFxdeFxd[i] = np.linalg.norm( vOut - vFxd )/np.linalg.norm(vOut)
            vLckeFxd[i] = np.linalg.norm( vOut - vLck )/np.linalg.norm(vOut)
        print('testFxdModel, converged:',100*sum(Convrg)/len(Convrg),'%')
        # print('Testing Complete.\n',time.process_time())
        
        fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
        ax0.plot(k,vLckeLck);
        ax0.plot(k,vFxdeLck);
        ax0.grid(True)
        ax0.set_ylabel('Voltage Error')
        ax0.set_xlabel('Continuation factor k')
        ax1.plot(k,vLckeFxd);
        ax1.plot(k,vFxdeFxd);
        ax1.plot(k,vLckeLck,'k--');
        ax1.grid(True)
        ax1.set_ylabel('Voltage Error')
        ax1.set_xlabel('Continuation factor k')
        plt.tight_layout()
        plt.show()
        
        return vFxdeLck, vLckeLck, vFxdeFxd, vLckeFxd,k
    
    def testLtcModel(self,k = np.concatenate((np.arange(-1.5,1.6,0.1),np.arange(1.5,-1.6,-0.1))),cvrModel=False):
        # NB: Currently a direct port of testFxdModel!
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        if cvrModel:
            self.loadCvrDssModel(pCvr=self.pCvr,qCvr=self.qCvr)
        else:
            self.loadDssModel()
        
        # self.loadDssModel()
        
        print('Start fxd testing. \n',time.process_time())
        
        dF = self.Altc.dot(np.concatenate((self.xY[self.syIdx],self.xD)))
        dK = self.Ky.dot(self.xY[self.syIdx]) + self.Kd.dot(self.xD)
        
        vFxdeLck = np.zeros(len(k))
        vLckeLck = np.zeros(len(k))
        vFxdeFxd = np.zeros(len(k))
        vLckeFxd = np.zeros(len(k))
        
        Convrg = []
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            vOut = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][self.reIdx]
            
            vFxd = dF*k[i] + self.Bltc
            vLck = (dK*k[i] + self.bV)[self.reIdx]
            
            vFxdeLck[i] = np.linalg.norm( vOut - vFxd )/np.linalg.norm(vOut)
            vLckeLck[i] = np.linalg.norm( vOut - vLck )/np.linalg.norm(vOut)
            
        DSSText.Command='set controlmode=static'
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            vOut = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][self.reIdx]
            
            # vFxd = dF*k[i] + self.Bfxd
            vFxd = dF*k[i] + self.Bltc
            vLck = (dK*k[i] + self.bV)[self.reIdx]
            
            vFxdeFxd[i] = np.linalg.norm( vOut - vFxd )/np.linalg.norm(vOut)
            vLckeFxd[i] = np.linalg.norm( vOut - vLck )/np.linalg.norm(vOut)
        print('testLtcModel, converged:',100*sum(Convrg)/len(Convrg),'%')
        
        fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
        ax0.plot(k,vLckeLck);
        ax0.plot(k,vFxdeLck);
        ax0.grid(True)
        ax0.set_ylabel('Voltage Error')
        ax0.set_xlabel('Continuation factor k')
        ax1.plot(k,vLckeFxd);
        ax1.plot(k,vFxdeFxd);
        ax1.plot(k,vLckeLck,'k--');
        ax1.grid(True)
        ax1.set_ylabel('Voltage Error')
        ax1.set_xlabel('Continuation factor k')
        plt.tight_layout()
        plt.show()
        
        return vFxdeLck, vLckeLck, vFxdeFxd, vLckeFxd,k
        
    def saveNrelModel():
        header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+self.fn
        
        dir0 = self.WD + '\\lin_models\\' + self.feeder
        sn0 = dir0 + '\\' + self.feeder
        
        lp_str = str(round(lin_point*100).astype(int)).zfill(3)
        if not os.path.exists(dir0):
            os.makedirs(dir0)        
        np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
        
        np.save(sn0+'Ky'+lp_str+'.npy',self.Ky)
        np.save(sn0+'xY'+lp_str+'.npy',self.xY)
        np.save(sn0+'bV'+lp_str+'.npy',self.bV)
        
        np.save(sn0+'xhyCap0'+lp_str+'.npy',self.xhyCap0)
        np.save(sn0+'xhyLds0'+lp_str+'.npy',self.xhyLds0)
        
        # np.save(sn0+'v_idx'+lp_str+'.npy',self.v_idx)
        
        np.save(sn0+'vKvbase'+lp_str+'.npy',self.vKvbase)
        np.save(sn0+'vYNodeOrder'+lp_str+'.npy',self.vYNodeOrder)
        np.save(sn0+'SyYNodeOrder'+lp_str+'.npy',self.SyYNodeOrder)
        np.save(sn0+'SdYNodeOrder'+lp_str+'.npy',self.SdYNodeOrder)
        if 'calcReg' in locals():
            np.save(sn0+'WyReg'+lp_str+'.npy',self.WyReg)
            np.save(sn0+'aIreg'+lp_str+'.npy',self.aIreg)
            np.save(sn0+'WregBus'+lp_str+'.npy',self.WregBus)
        
        if len(H)!=0:
            np.save(sn0+'Kd'+lp_str+'.npy',self.Kd)
            np.save(sn0+'xD'+lp_str+'.npy',self.xD)
            np.save(sn0+'xhdCap0'+lp_str+'.npy',self.xhdCap0)
            np.save(sn0+'xhdLds0'+lp_str+'.npy',self.xhdLds0)
            if 'calcReg' in locals():
                np.save(sn0+'WdReg'+lp_str+'.npy',WdReg)
                # np.save(sn0+'WdBus'+lp_str+'.npy',WdBus)
    
    def saveCc(self):
        self.createNrelModel(self.linPoint)
        self.testVoltageModel()
        
        lp_str = str(np.round(self.linPoint*100).astype(int)).zfill(3)
        MyCC = self.My
        xhyCC = self.xY

        aCC = self.aV
        V0CC = self.V0

        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        # Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        # YbusCC = Ybus

        allLds = DSSCircuit.Loads.AllNames
        loadBuses = {}
        for ld in allLds:
            DSSCircuit.SetActiveElement('Load.'+ld)
            loadBuses[ld]=DSSCircuit.ActiveElement.BusNames[0]

        dirCC = self.WD + '\\lin_models\\ccModelsNew\\' + self.feeder
        snCC = dirCC + '\\' + self.feeder

        if not os.path.exists(dirCC):
            os.makedirs(dirCC)

        np.save(snCC+'vYNodeOrder'+lp_str+'.npy',self.vYNodeOrder)
        np.save(snCC+'SyYNodeOrder'+lp_str+'.npy',self.SyYNodeOrder)
            
        np.save(snCC+'MyCc'+lp_str+'.npy',MyCC)
        np.save(snCC+'xhyCc'+lp_str+'.npy',xhyCC)
        np.save(snCC+'aCc'+lp_str+'.npy',aCC)
        np.save(snCC+'V0Cc'+lp_str+'.npy',V0CC)
        np.save(snCC+'YbusCc'+lp_str+'.npy',self.Ybus)
        np.save(snCC+'YNodeOrderCc'+lp_str+'.npy',DSSCircuit.YNodeOrder)

        busCoords = getBusCoords(DSSCircuit,DSSText)
        busCoordsAug,PDelements,PDparents = getBusCoordsAug(busCoords,DSSCircuit,DSSText)

        np.save(snCC+'busCoords'+lp_str+'.npy',busCoordsAug)
        np.save(snCC+'loadBusesCc'+lp_str+'.npy',[loadBuses]) # nb likely to be similar to vecSlc(YNodeOrder[3:],p_idx)?
    
    
    def setupPlots(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        busCoords = getBusCoords(DSSCircuit,DSSText)
        busCoordsAug,PDelements,PDparents = getBusCoordsAug(busCoords,DSSCircuit,DSSText)
        self.busCoords = busCoordsAug
        self.branches = getBranchBuses(DSSCircuit)
        self.getSourceBus()
        self.regBuses = get_regIdx(DSSCircuit)[1]
        
        if self.feeder in ['epri24','8500node','123bus','123busCvr','epriJ1','epriK1','epriM1']: # if there is a regulator 'on' the source bus
            self.srcReg = 1
        else:
            self.srcReg = 0        
        
        plotMarkerDict = {'13bus':150,'34bus':150,'eulv':100,'epri5':10,'epriK1cvr':15,'epriK1':15}
        
        if self.feeder in plotMarkerDict.keys():
            self.plotMarkerSize=plotMarkerDict[self.feeder]
        elif self.feeder[0]=='n':
            self.plotMarkerSize=25
        else:
            self.plotMarkerSize=50
            
        self.sfDict3ph = {'13bus':6,'123bus':60,'123busCvr':60,'epriK1cvr':40,'34bus':80,'n1':4,'eulv':2,'epri5':90,'epri7':50,'n10':3,'n4':3,'n27':4}
            
        vMap = cm.RdBu
        pMap = cm.GnBu
        qMap = cm.PiYG
        sOnly = cm.Blues
        self.colormaps = { 'v0':vMap,'p0':pMap,'q0':qMap,'qSln':qMap,'vSln':vMap,'ntwk':sOnly,'qSlnPh':qMap }
    
    def getConstraints(self):
        # cns = {'mvHi':1.055,'lvHi':1.05,'mvLo':0.95,'lvLo':0.92,'plim':1e3,'qlim':1e3,'tlim':0.1,'iScale':1.2}
        cns = {'mvHi':1.055,'lvHi':1.05,'mvLo':0.95,'lvLo':0.92,'plim':1e3,'qlim':2400,'tlim':0.1,'iScale':1.2}
        
        nHouses = {'n1':200 ,'n4':186,'n10':64,'n27':200 ,'eulv':55}
        # EU style networks have slightly different characteristics
        if self.feeder=='eulv' or self.feeder[0]=='n':
            cns['mvHi']=1.055 # keep the same
            # cns['mvLo']=0.95
            cns['mvLo']=0.90
            cns['lvHi']=1.10
            cns['lvLo']=0.90
            cns['iScale'] = (nHouses[self.feeder]*2)/800
        if self.feeder=='epriK1cvr':
            cns['iScale'] = 1.5
        if self.feeder=='epri5':
            cns['iScale'] = 1.7
        if self.feeder=='epri24cvr':
            cns['iScale'] = 4.0
            cns['mvHi'] = 1.10
            cns['mvLo'] = 0.92
            cns['lvHi'] = 1.10
            cns['lvLo'] = 0.92

            
        return cns
    
    def setupConstraints(self,cns=None):
        self.hvBuses = (self.vKvbase>1000)
        self.lvBuses = (self.vKvbase<=1000)
        
        if cns==None:
            cns = self.getConstraints()
        
        self.vIn = np.where((abs(self.aV)/self.vKvbase)>0.5)[0].tolist()
        self.vInYNodeOrder = vecSlc(self.vYNodeOrder,self.vIn)
        self.vInKvbase = self.vKvbase[self.vIn]
        
        self.vHi = cns['mvHi']*self.vKvbase*self.hvBuses + cns['lvHi']*self.vKvbase*self.lvBuses
        self.vLo = cns['mvLo']*self.vKvbase*self.hvBuses + cns['lvLo']*self.vKvbase*self.lvBuses
        
        self.pLim = cns['plim']*self.xSscale # convert from W to power units
        self.qLim = cns['qlim']*self.xSscale # convert from W to power units
        self.tLim = cns['tlim']*self.xTscale # convert from pu to tap units
        
        self.iScale = cns['iScale']
        
        solvers.options['show_progress']=False
        
    
    # PLOTTING FUNCTIONS cannibalised from linSvdCalcs
    def plotNetwork(self,pltShow=True,pltSave=False):
        fig,ax = plt.subplots()
        self.getBusPhs()
        self.plotBranches(ax)
        
        scoreNom = np.ones((self.nPctrl))
        scores, minMax0 = self.getSetVals(scoreNom,busType='s')
        self.plotBuses(ax,scores,minMax0,cmap=self.colormaps['ntwk'],edgeOn=False)

        xlm = ax.get_xlim()
        ylm = ax.get_xlim()
        dx = xlm[1] - xlm[0]; dy = ylm[1] - ylm[0] # these seem to be in feet for k1
        
        self.plotSub(ax,pltSrcReg=False)
        # srcCoord = self.busCoords[self.vSrcBus]
        # ax.annotate('Substation',(srcCoord[0]+0.01*dx,srcCoord[1]+0.01*dy))
        
        # # DO NOT delete - useful for plotting with distances but full working model not yet put in.
        # dist = 50
        # # x0 = xlm[0] + 0.2*dx # N1, N7
        # x0 = xlm[0] + 0.8*dx # N10
        # y0 = ylm[0] + 0.05*dy
        # ax.plot([x0,x0+dist],[y0,y0],'k-')
        # ax.plot([x0,x0],[y0-0.005*dy,y0+0.005*dy],'k-')
        # ax.plot([x0+dist,x0+dist],[y0-0.005*dy,y0+0.005*dy],'k-')
        # ax.annotate('50 metres',(x0+(dist/2),y0+dy*0.02),ha='center')
        
        ax.axis('off')
        plt.grid(False)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        if pltSave:
            # SN = os.path.join(self.SD,'plotNetwork','plotNetwork'+self.feeder)
            SN = os.path.join(self.SD,'plotNetwork'+self.feeder)
            plt.savefig(SN+'.png',bbox_inches='tight', pad_inches=0.01)
            plt.savefig(SN+'.pdf',bbox_inches='tight', pad_inches=0)
            print('Network plot saved to '+SN)
        if pltShow:
            plt.title('Feeder: '+self.feeder,loc='left') # TITLE only in show.
            plt.tight_layout()
            plt.show()
        return ax

    def getBusPhs(self):
        self.bus0v, self.phs0v = self.busPhsLoop(self.vYNodeOrder)
        self.bus0vIn, self.phs0vIn = self.busPhsLoop(self.vInYNodeOrder)
        
        sYZ = np.concatenate((self.SyYNodeOrder,self.SdYNodeOrder))
        self.bus0s, self.phs0s = self.busPhsLoop(sYZ)
        
    def busPhsLoop(self,YZset):
        bus0 = []
        phs0 = []
        for yz in YZset:
            fullBus = yz.split('.')
            bus0 = bus0+[fullBus[0].lower()]
            if len(fullBus)>1:
                phs0 = phs0+[fullBus[1::]]
            else:
                phs0 = phs0+[['1','2','3']]
        return np.array(bus0), np.array(phs0)
    
    def plotBranches(self,ax,scores=[]):
        # branchCoords = self.branchCoords
        branches = self.branches
        busCoords = self.busCoords
        segments = []
        for branch,buses in branches.items():
            bus1 = buses[0].split('.')[0]
            bus2 = buses[1].split('.')[0]
            segments = segments + [[busCoords[bus1],busCoords[bus2]]]
            # if branch.split('.')[0]=='Transformer':
                # ax.plot(points0[-1],points1[-1],'--',Color='#777777')
        if len(scores)==0:
            coll = LineCollection(segments, Color='#cccccc')
        else:
            coll = LineCollection(segments, cmap=plt.cm.viridis)
            coll.set_array(scores)
        ax.add_collection(coll)
        ax.autoscale_view()
        self.segments = segments
    
    def getSourceBus(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSCircuit.Vsources.First
        self.vSrcBus = DSSCircuit.ActiveElement.BusNames[0]
    
    def plotSub(self,ax,pltSrcReg=True):
        
        srcCoord = self.busCoords[self.vSrcBus]
        if np.isnan(srcCoord[0]):
            print('Nominal source bus coordinate not working, trying z appended...')
            try:
                srcCoord = self.busCoords[self.vSrcBus+'z']
            except:
                srcCoord = [np.nan,np.nan]
        if np.isnan(srcCoord[0]):
            print('Nominal source bus coordinate not working, trying 1...')
            srcCoord = self.busCoords['1']
        
        
        if not np.isnan(srcCoord[0]):
            ax.plot(srcCoord[0],srcCoord[1],'k',marker='H',markersize=8,zorder=+20)
            ax.plot(srcCoord[0],srcCoord[1],'w',marker='H',markersize=3,zorder=+21)
            if self.srcReg and pltSrcReg:
                ax.plot(srcCoord[0],srcCoord[1],'r',marker='H',markersize=3,zorder=+21)
            else:
                ax.plot(srcCoord[0],srcCoord[1],'w',marker='H',markersize=3,zorder=+21)
        else:
            print('Could not plot source bus'+self.vSrcBus+', no coordinate')
    
    def getSetVals(self,Set,aveType='mean',busType='v'):
        busCoords = self.busCoords
        if busType=='v':
            bus0 = self.bus0v
            phs0 = self.phs0v.flatten()
        elif busType=='s':
            bus0 = self.bus0s
            phs0 = self.phs0s.flatten()
        elif busType=='vIn':
            bus0 = self.bus0vIn
            phs0 = self.phs0vIn.flatten()

        setMin = 1e100
        setMax = -1e100
        setVals = {}
        if aveType=='ph':
            setVals['1'] = {};  setVals['2'] = {}; setVals['3'] = {}
            for bus in busCoords: #initialise
                setVals['1'][bus] = np.nan;setVals['2'][bus] = np.nan;setVals['3'][bus] = np.nan
        else:
            for bus in busCoords: setVals[bus] = np.nan #initialise
        
        for bus in busCoords:
            if not np.isnan(busCoords[bus][0]):
                if aveType=='ph':
                    phs = phs0[bus0==bus.lower()]
                
                vals = Set[bus0==bus.lower()]
                vals = vals[~np.isnan(vals)]
                if len(vals)>0:
                    if aveType=='mean':
                        setVals[bus] = np.mean(vals)
                        setMax = max(setMax,np.mean(vals))
                        setMin = min(setMin,np.mean(vals))
                    elif aveType=='max':
                        setVals[bus] = np.max(vals)
                        setMax = max(setMax,np.max(vals))
                        setMin = min(setMin,np.max(vals))
                    elif aveType=='min':
                        setVals[bus] = np.min(vals)
                        setMax = max(setMax,np.min(vals))
                        setMin = min(setMin,np.min(vals))
                    elif aveType=='ph':
                        i=0
                        for ph in phs:
                            setVals[ph[0]][bus] = vals[i]
                            setMax = max(setMax,np.min(vals[i]))
                            setMin = min(setMin,np.min(vals[i]))
                            i+=1
                        
        if setMin==setMax:
            setMinMax=None
        else:
            setMinMax = [setMin,setMax]
        return setVals, setMinMax
        
    def plotBuses(self,ax,scores,minMax,cmap=plt.cm.viridis,edgeOn=True):
        busCoords = self.busCoords
        x0scr = []
        y0scr = []
        xyClr = []
        x0nne = []
        y0nne = []
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
                    
                    xyClr = xyClr + [cmap(score)]
        
        
        marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='tab:red')
        plt.scatter(x0scr,y0scr,marker='.',Color=xyClr,zorder=+10,s=self.plotMarkerSize)
        if edgeOn: plt.scatter(x0scr,y0scr,marker='.',zorder=+11,s=self.plotMarkerSize,facecolors='none',edgecolors='k')
        
        plt.scatter(x0nne,y0nne,Color='#cccccc',marker='.',zorder=+5,s=15)
    
    def plotBuses3ph(self,ax,scores3ph,minMax,cmap=plt.cm.viridis,edgeOn=True):
        busCoords = self.busCoords
        x0scr = []
        y0scr = []
        xyClr = []
        x0nne = []
        y0nne = []
        phscr = []
        scale = 50
        phsOffset = {'1':[0.0,1.0],'2':[-0.86,-0.5],'3':[0.86,-0.5]}
        for ph,scores in scores3ph.items():
            for bus,coord in busCoords.items():
                if not np.isnan(busCoords[bus][0]):
                    if np.isnan(scores[bus]):
                        x0nne = x0nne + [coord[0]]
                        y0nne = y0nne + [coord[1]] 
                    else:
                        phscr = phscr + [ph]
                        x0scr = x0scr + [coord[0]]
                        y0scr = y0scr + [coord[1]]
                        if minMax==None:
                            score=scores[bus]
                        else:
                            score = (scores[bus]-minMax[0])/(minMax[1]-minMax[0])
                        
                        xyClr = xyClr + [cmap(score)]
            
            self.hex3phPlot(ax,x0scr,y0scr,xyClr,phscr,sf=self.sfDict3ph[self.feeder])
            # plt.scatter(x0scr,y0scr,marker='.',Color=xyClr,zorder=+10,s=self.plotMarkerSize)
            # if edgeOn: plt.scatter(x0scr,y0scr,marker='.',zorder=+11,s=self.plotMarkerSize,facecolors='none',edgecolors='k')
            plt.scatter(x0nne,y0nne,Color='#cccccc',marker='.',zorder=+5,s=15)
    
    def hex3phPlot(self,ax,x,y,xyClr,xyPhs,sf):
        xy0 = np.exp(1j*np.linspace(np.pi/2,5*np.pi/2,7))
        # phsOffset = {'1':np.exp(1j*np.pi/6),'2':np.exp(1j*5*np.pi/6),'3':np.exp(1j*-np.pi/2)}
        phsOffset = {'1':np.exp(1j*0),'2':np.exp(1j*2*np.pi/3),'3':np.exp(1j*4*np.pi/3)}
        patches = []
        for i in range(len(x)):
            brSf = 1.15
            xyPts = np.c_[ sf*(xy0 + brSf*phsOffset[xyPhs[i]]).real + x[i],sf*(xy0 + brSf*phsOffset[xyPhs[i]]).imag + y[i]]
            patches.append(Polygon(xyPts,True,fill=1,color=xyClr[i])) #NB putting zorder here doesn't seem to work
            patches.append(Polygon(xyPts,True,fill=0,linestyle='-',linewidth=0.4,edgecolor='k'))
            patches.append( Polygon( [[x[i],y[i]],[x[i]+sf*brSf*phsOffset[xyPhs[i]].real,y[i]+sf*brSf*phsOffset[xyPhs[i]].imag]],False,color='k',linewidth=1.0 ))
        p = PatchCollection(patches,match_original=True)
        p.set_zorder(10)
        ax.add_collection(p)
    
    def plotNetBuses(self,type,regsOn=True,pltShow=True,minMax=None,pltType='mean',varMax=10,cmap=None):
        self.initialiseOpenDss(); self.setupPlots()
        fig,ax = plt.subplots()
        self.getBusPhs()
        
        self.plotBranches(ax)
        self.plotSub(ax)
        self.plotRegs(ax)
        
        ttl = None
        if type=='v0':
            # scoreNom = self.bV/self.vKvbase
            # scores, minMax0 = self.getSetVals(scoreNom,pltType)
            scoreNom = self.slnF0[4][self.vIn]/self.vInKvbase
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'vIn')
            ttl = 'Voltage (pu)'
        elif type =='p0':
            scoreNom = -1e-3*np.r_[self.xY[self.syIdx][:self.nPy],self.xD[:self.nPd]]
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'s')
            if minMax0 is None:
                minMax0 = [0,np.nanmax(list(scores.values()))]
            else:
                minMax0[0] = 0
            ttl = 'Load Power (kW)'
        elif type=='q0':
            scoreNom = 1e-3*np.r_[self.xY[self.syIdx][self.nPy:],self.xD[self.nPd:]]*self.lScale/self.xSscale
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'s')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [-minMaxAbs,minMaxAbs]
            ttl = 'Qgen (kVAr)' # Positive implies capacitive
        elif type=='qSln':
            # scoreNom = 1e-3*(np.r_[self.xY[self.syIdx][self.nPy:],self.xD[self.nPd:]] + self.slnX[self.nPctrl:self.nSctrl])
            scoreNom = (self.slnX[self.nPctrl:self.nSctrl])*self.lScale/self.xSscale
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'s')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [-minMaxAbs,minMaxAbs]
            ttl = 'Opt Qgen (kVAr)' # Positive implies capacitive
        elif type=='vSln':
            # TL,PL,TC,V,I = self.slnF
            scoreNom = self.slnF[4][self.vIn]/self.vInKvbase
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'vIn')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [0.92,1.05]
            ttl = 'Voltage (pu)' # Positive implies capacitive
        elif type=='qSlnPh':
            scoreNom = (self.slnX[self.nPctrl:self.nSctrl])*self.lScale/self.xSscale
            scores, minMax0 = self.getSetVals(scoreNom,aveType='ph',busType='s')
            # minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            scoresFull = list(scores['1'].values()) + list(scores['2'].values()) + list(scores['3'].values())
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(scoresFull),np.nanmin(scoresFull)]) ) )
            minMax0 = [-minMaxAbs,minMaxAbs]
            ttl = '$Q^{*}$, kVAr' # Positive implies capacitive
        
        if cmap is None: cmap = self.colormaps[type]
        
        if minMax==None:
            minMax = minMax0
        
        if type[-2:]=='Ph':
            self.plotBuses3ph(ax,scores,minMax,cmap=cmap)
        else:
            self.plotBuses(ax,scores,minMax,cmap=cmap)
        
        self.plotNetColorbar(ax,minMax,cmap,ttl=ttl)
        
        ax.axis('off')
        if pltShow: plt.title('Feeder: '+self.feeder,loc='left')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        if pltShow:
            plt.show()
        else:
            self.currentAx = ax
    
    def plotNetColorbar(self,ax,minMax,cmap,ttl=None,nCbar=150):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = [xlim[0]-1]
        y0 = [ylim[0]-1]
        cntr = ax.contourf( np.array([x0*2,x0*2]),np.array([y0*2,y0*2]), np.array([minMax,minMax[::-1]]),nCbar,cmap=cmap )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # cbar = plt.colorbar(cntr,shrink=0.75,ticks=np.linspace(minMax[0],minMax[1],5))
        cbar = plt.colorbar(cntr,shrink=0.6,ticks=np.linspace(minMax[0],minMax[1],5))
        if ttl!=None:
            cbar.ax.set_title(ttl,pad=10,fontsize=10)
    
    
    def plotRegs(self,ax):
        if self.nT>0:
            i=0
            for regBus in self.regBuses:
                regCoord = self.busCoords[regBus.split('.')[0].lower()]
                if not np.isnan(regCoord[0]):
                    ax.plot(regCoord[0],regCoord[1],'r',marker=(6,1,0),zorder=+5)
                    # ax.annotate(str(i),(regCoord[0],regCoord[1]),zorder=+40)
                else:
                    print('Could not plot regulator bus'+regBus+', no coordinate')
                i+=1
        else:
            print('No regulators to plot.')
            
    # TIME SERIES ANALYSIS STUFF
    def loadLoadProfiles(self,n=10):
        # CREATES the load profiles. The first output is not normalised, the second output is.
        self.loadProfiles={}
        for i in range(1,101):
            LD = os.path.join(self.WD,'LVTestCase_copy','Daily_1min_100profiles','load_profile_'+str(i)+'.txt')
            with open(LD,'r') as loadFile:
                data = loadFile.read()
            nos = data.replace(' \n','').split(' ')[1:]
            
            load = []
            for no in nos: load.append(float(no))
            
            loadAgg = []
            for j in range(1440//n): loadAgg.append(sum(load[j*n:(j+1)*n ])/n)
            
            
            self.loadProfiles[i]=[loadAgg,list(np.array(loadAgg)/max(loadAgg))]
    
    def runLoadFlow(self):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        LDS = DSSCircuit.Loads
        self.nLoads = LDS.Count
        
        np.random.seed(0)
        loadInt = np.random.randint(1,101,self.nLoads)
        
        # first records nominal positions.
        i = LDS.First
        loadPQ = {}
        while i:
            loadPQ[i] = [LDS.kW,LDS.kvar]
            i = LDS.Next
        
        TP = np.array([])
        for ii in range(len(self.loadProfiles[1][0])): 
            i = LDS.First
            while i:
                mult = self.loadProfiles[loadInt[i-1]][1][ii]
                LDS.kW = mult*loadPQ[i][0]
                LDS.kvar = mult*loadPQ[i][1]
                i = LDS.Next
            DSSSolution.Solve()
            TP = np.r_[TP,-tp_2_ar(DSSCircuit.TotalPower)]
            print(ii)

        plt.plot(TP.real)
        plt.plot(TP.imag)
        plt.show()