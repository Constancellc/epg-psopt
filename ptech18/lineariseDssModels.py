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

def equalMat(n):
    return toeplitz([1]+[0]*(n-2),[1,-1]+[0]*(n-2))

def dirPrint(obj):
    print(*dir(obj),sep='\n')

from cvxopt import matrix, solvers

from win32com.client import makepy

class buildLinModel:
    def __init__(self,fdr_i=6,linPoints=[None],pCvr = 0.75,saveModel=False,setCapsModel='linPoint',
                                    FD=sys.argv[0],modelType=None,method='fpl',SD=[],pltSave=False):
        self.WD = os.path.dirname(FD)
        self.setCapsModel = setCapsModel
        self.SD = SD
        
        logging.basicConfig(filename=os.path.join(self.WD,'example.log'),filemode='w',level=logging.INFO)
        self.log = logging.getLogger()
        self.log.info('Feeder: '+str(fdr_i))
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr']
        
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
            if self.setCapsModel=='linPoint':
                self.capPosLin=lp0data['capPosOut']
            else:
                self.capPosLin=None
        except:
            if linPoints[0]==None:
                linPoints = np.array([1.0])
            self.capPosLin=None
        
        self.initialiseOpenDss()
        
        self.pCvr = pCvr
        self.qCvr = 0.0
        self.linPoint = linPoints[0]
        self.method=method
        
        # BUILD MODELS
        if modelType == None:
            self.makeCvrQp()
        elif modelType == 'linOnly':
            self.createNrelModel(linPoints[0])
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
        
        
        self.setupPlots()
        if modelType in [None,'linOnly','fxd','ltc','cvx']: 
            self.setupConstraints()
            # TL0,PL0,TC0,V0,I0 = self.runQp(np.zeros(self.nCtrl))
            self.slnF0 = self.runQp(np.zeros(self.nCtrl))
        
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
        
    
    def saveLinModel(self):
        SD = os.path.join(self.WD,'lin_models','cvr_models',self.feeder)
        power = str(np.round(self.linPoint*100).astype(int)).zfill(3)
        aCvr = str(np.round(self.pCvr*100).astype(int)).zfill(3)
        SN = os.path.join(SD,self.feeder+'P'+power+'A'+aCvr+'.pkl')
        if not os.path.exists(SD):
            os.makedirs(SD)
        self.dssStuff = [] # can't be saved
        with open(SN,'wb') as outFile:
            pickle.dump(self,outFile)
        
    
    def initialiseOpenDss(self):
        sys.argv=["makepy","OpenDSSEngine.DSS"]
        makepy.main()
        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.dssStuff = [DSSObj,DSSObj.Text,DSSObj.ActiveCircuit,DSSObj.ActiveCircuit.Solution]
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile '+self.fn
        
    def makeCvrQp(self,verbose=True):
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
        #
        # put in a 
        
        self.QP = {} # this is our goal.
        
        lin_point = self.linPoint
        self.currentLinPoint = self.linPoint
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(self.pCvr)+' cvrvars='+str(self.qCvr)
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = lin_point
        DSSSolution.Solve()
        print('\nNominally converged:',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        self.Cap_No0 = getCapPstns(DSSCircuit)
        if self.capPosLin==None:
            self.capPosLin = self.Cap_No0
        print(self.Cap_No0)
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        self.loadCvrDssModel(self.pCvr,self.qCvr,loadMult=lin_point)
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        
        self.xY, self.xD, self.pyIdx, self.pdIdx  = ldValsOnly( DSSCircuit ) # NB these do not change with the circuit!
        
        self.qyIdx = [self.pyIdx[0]+DSSCircuit.NumNodes-3] # NB: note that this is wrt M, not YNodeOrder.
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
        
        print('Create linear models:\n',time.process_time()); t = time.time()
        
        if self.method=='fot':
            My,Md,a,dMy,dMd,da = firstOrderTaylor( Ybus,Vh,V0,xYcvr,xDcvr,H[:,3:] ); \
                            print('===== Using the FOT method =====')
        elif self.method=='fpl':
            My,Md,a,dMy,dMd,da = cvrLinearization( Ybus,Vh,V0,H,0,0,self.vKvbase,self.vKvbaseD ); \
                            print('Using the FLP method')
        
        
        self.nV = len(a)
        self.createTapModel(lin_point,cvrModel=True) # this seems to be ok. creates Kt and Mt matrices.
        self.nT = self.Mt.shape[1]
        Ky,Kd,b = nrelLinK( My,Md,Vh,xYcvr,xDcvr )
        # dKy,dKd,db = nrelLinK( dMy,dMd,dVh,xYcvr,xDcvr )
        dKy,dKd,self.dKt,db = lineariseMfull( dMy,dMd,H[:,3:].dot(self.Mt),dVh,xYcvr,xDcvr,np.zeros(self.nT) )
        
        print('Linear models created.:',time.time()-t)
        
        Vh0 = (My.dot(xYcvr) + Md.dot(xDcvr)) + a # for validation
        Va0 = (Ky.dot(xYcvr) + Kd.dot(xDcvr)) + b # for validation
        dVh0 = (dMy.dot(xYcvr) + dMd.dot(xDcvr)) + da # for validation
        dVa0 = (dKy.dot(xYcvr) + dKd.dot(xDcvr)) + db # for validation
        
        self.log.info('\nVoltage clx error (lin point), Volts:'+str(np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh)))
        self.log.info('Voltage clx error (no load point), Volts:'+str(np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad)))
        self.log.info('\nVoltage abs error (lin point), Volts:'+str(np.linalg.norm(Va0-abs(Vh))/np.linalg.norm(abs(Vh))))
        self.log.info('Voltage abs error (no load point), Volts:'+str(np.linalg.norm(abs(b)-abs(VnoLoad))/np.linalg.norm(abs(VnoLoad))))
        if len(dVh)>0:
            self.log.info('\n Delta voltage clx error (lin point), Volts:'+str(np.linalg.norm(dVh0-dVh)/np.linalg.norm(dVh)))
            self.log.info('Delta voltage clx error (no load point), Volts:'+str(np.linalg.norm(da-dVnoLoad)/np.linalg.norm(dVnoLoad)))
            self.log.info('\nDelta voltage abs error (lin point), Volts:'+str(np.linalg.norm(dVa0-abs(dVh))/np.linalg.norm(abs(dVh))))
            self.log.info('Delta voltage abs error (no load point), Volts:'+str(np.linalg.norm(abs(db)-abs(dVnoLoad))/np.linalg.norm(abs(dVnoLoad))))
            
        self.syIdx = np.concatenate((self.pyIdx[0],self.qyIdx[0]))
        self.My = My[:,self.syIdx]
        self.Ky = Ky[:,self.syIdx]
        self.aV = a
        self.bV = b
        self.H = H
        self.V0 = V0
        
        self.vYNodeOrder = YNodeOrder[3:]
        self.SyYNodeOrder = vecSlc(self.vYNodeOrder,self.pyIdx)
        self.SdYNodeOrder = vecSlc(self.vYNodeOrder,self.pdIdx)
        self.Md = Md
        self.Kd = Kd
        self.currentLinPoint = lin_point
        
        self.createWmodel(lin_point) # this seems to be ok
        
        f0 = self.v2iBrYxfmr.dot(YNodeV)[self.iXfmrModelled] # voltages to currents through xfmrs
        fNoload = self.v2iBrYxfmr.dot(VoffLoad)[self.iXfmrModelled] # voltages to currents through xfmrs
        
        KyW, KdW, KtW, self.bW = lineariseMfull(self.WyXfmr,self.WdXfmr,self.WtXfmr,f0,xYcvr[self.syIdx],xDcvr,np.zeros(self.WtXfmr.shape[1]))
        
        f0cpx = self.WyXfmr.dot(xYcvr[self.syIdx]) + self.WdXfmr.dot(xDcvr) + self.WtXfmr.dot(np.zeros(self.WtXfmr.shape[1])) + self.aIxfmr
        f0lin = KyW.dot(xYcvr[self.syIdx]) + KdW.dot(xDcvr) + KtW.dot(np.zeros(self.WtXfmr.shape[1])) + self.bW
        
        self.log.info('\nCurrent cpx error (lin point):'+str(np.linalg.norm(f0cpx-f0)/np.linalg.norm(f0)))
        self.log.info('Current cpx error (no load point):'+str(np.linalg.norm(self.aIxfmr-fNoload)/np.linalg.norm(fNoload)))
        self.log.info('\nCurrent abs error (lin point):'+str(np.linalg.norm(f0lin-abs(f0))/np.linalg.norm(abs(f0))))
        self.log.info('Current abs error (no load point):'+str(np.linalg.norm(self.bW-abs(fNoload))/np.linalg.norm(abs(fNoload)))+'( note that this is usually not accurate, it seems.)')
        
        # control (c) variables (in order): Pgen(Y then D) (kW),Qgen(Y then D) (kvar),t (pu).
        # notation as 'departure2arrival'
        self.nPctrl = self.nPy+self.nPd
        self.nSctrl = self.nPctrl*2
        self.nCtrl = self.nSctrl + self.nT
        
        self.X0 = np.concatenate( (xYcvr[self.syIdx],xDcvr,np.zeros(self.nT)) )
        self.X0ctrl = np.concatenate( (xYcvr[self.pyIdx[0]],xDcvr[:self.nPd],xYcvr[self.qyIdx[0]],xDcvr[self.nPd::],np.zeros(self.nT)) )
        
        self.Kc2v = np.concatenate( (Ky[:,self.pyIdx[0]],Kd[:,:self.nPd],
                                    Ky[:,self.qyIdx[0]],Kd[:,self.nPd::],
                                    self.Kt),axis=1)
        self.Kc2i = np.concatenate( (KyW[:,:self.nPy],KdW[:,:self.nPd],
                                KyW[:,self.nPy::],KdW[:,self.nPd::],
                                KtW),axis=1) # limits for these are in self.iXfmrLims.
                                    
        Kc2d = np.concatenate( (dKy[:,self.pyIdx[0]],dKd[:,:self.nPd],
                                dKy[:,self.qyIdx[0]],dKd[:,self.nPd::],
                                self.dKt),axis=1)
        
        Kc2vloadpu = np.concatenate( (  dsf.vmM( 1/self.vKvbaseY,self.Kc2v[self.pyIdx[0]] ),dsf.vmM( 1/self.vKvbaseD, Kc2d[:self.nPd] ) ),axis=0)
        
        Kc2pC = np.concatenate((xYcvr[self.pyIdx[0]],xDcvr[:self.nPd]))
        Kc2p =  dsf.vmM(Kc2pC*self.pCvr,Kc2vloadpu)
        
        Mmid = np.concatenate((self.My[:,:self.nPy],self.Md[:,:self.nPd],self.My[:,self.nPy::],self.Md[:,self.nPd::],self.Mt),axis=1)
        M = np.block([[np.zeros((3,len(self.X0)),dtype=complex)],[Mmid],[np.zeros((1,len(self.X0)),dtype=complex)]])
        print(M.shape)
        
        self.Mmid = Mmid # useful for debugging
        
        Wcnj = np.concatenate((self.Wy[:,:self.nPy],self.Wd[:,:self.nPd],self.Wy[:,self.nPy::],self.Wd[:,self.nPd::],self.Wt),axis=1).conj()
        aIcnj = (self.aI).conj()
        
        Wcnj = np.delete(Wcnj,self.wregIdxs,axis=0)
        aIcnj = np.delete(aIcnj,self.wregIdxs,axis=0)
        
        aV = np.concatenate((self.V0,self.aV,np.array([0])))
        
        P = np.zeros((len(self.yzW2V),len(M)))
        P[range(len(self.yzW2V)),self.yzW2V] = 1
        P = np.delete(P,self.wregIdxs,axis=0) # don't forget we take out the regulator models!
        PT = P.T
        
        qpQlss = 1e-3*np.real( (M.T).dot(PT.dot(Wcnj)) )
        self.qpQlss = 0.5*(qpQlss + qpQlss.T) # make symmetric.
        self.qpLlss0 = 1e-3*np.real( aV.dot(PT.dot(Wcnj)) + aIcnj.dot(P.dot(M)) )
        self.qpLlss = self.qpLlss0 + 2*self.qpQlss.dot(self.X0ctrl)
        self.qpClss0 = 1e-3*np.real( aV.dot(PT.dot(aIcnj)) )
        self.qpClss = self.qpClss0 + self.X0ctrl.dot( self.qpLlss0 + self.qpQlss.dot(self.X0ctrl))
        
        self.ploadL = -1e-3*np.sum(Kc2p,axis=0)
        self.ploadC = -1e-3*sum(Kc2pC)
        
        pcurtL = np.zeros(self.nCtrl)
        pcurtL[:self.nPctrl] = -1e-3
        self.pcurtL = pcurtL
        
        self.loadCvrDssModel(self.pCvr,self.qCvr,loadMult=lin_point)
        self.log.info('Actual losses:'+str(DSSCircuit.Losses))
        self.log.info('Model losses'+str(self.qpClss))
        self.log.info('TLoad:'+str(-DSSCircuit.TotalPower[0] - 1e-3*DSSCircuit.Losses[0]))
        self.log.info('TLoadEst:'+str(-1e-3*sum(Kc2pC)))
        
        print('Q PD test:',pdTest(self.qpQlss))
        
        return
    
    def runCvrQp(self,mode='full'):
        # minimize    (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #               A*x = b.
        # sol = solvers.qp(Q,p,G,h,A,b)
        # MODES
        # 1. 'full' Full optimization
        # 2. 'maxTap' maximise all taps
        # 3. 'minTap' minimise all taps with max Q
        # 4. 'part' full optimization, just one P + Q
        
        # OBJECTIVE FUNCTION
        if mode=='full' or mode=='part':
            # losses, load, curtailment
            Q = matrix( 2*self.qpQlss )
            p = matrix( self.qpLlss + self.ploadL + self.pcurtL )
        elif mode=='maxTap':
            Q = matrix( 0*self.qpQlss )
            if self.nT>0:
                t2vPu = np.sum( dsf.vmM( 1/self.vInKvbase,self.Kc2v[self.vIn,-self.nT:] ),axis=0 )
                p = matrix( np.r_[np.zeros((self.nPctrl*2)),-1*t2vPu] ) # maximise the average voltage
            else:
                p = matrix( np.r_[np.zeros((self.nPctrl*2))] )
        elif mode=='minTap':
            Q = matrix( 0*self.qpQlss )
            if self.nT>0:
                t2vPu = np.sum( dsf.vmM( 1/self.vInKvbase,self.Kc2v[self.vIn,-self.nT:] ),axis=0 )
                p = matrix( np.r_[np.zeros((self.nPctrl*2)),t2vPu] ) # minimise the average voltage
            else:
                p = matrix( np.r_[np.zeros((self.nPctrl*2))] )
                
        elif mode=='loss':
            Q = matrix( 2*self.qpQlss )
            p = matrix( self.qpLlss + self.pcurtL )
        elif mode=='load':
            Q = matrix( 0*self.qpQlss )
            p = matrix( self.ploadL + self.pcurtL )

        # INEQUALITY CONSTRAINTS
        # Upper Control variables, then voltages, then currents; then repeat but lower.
        # xLo <= Ax <= xHi
        xLimUp = matrix( np.concatenate(( np.zeros(self.nPctrl),np.ones(self.nPctrl)*self.qLim,np.ones(self.nT)*self.tLim)) )
        xLimLo = matrix( np.concatenate(( -np.ones(self.nPctrl)*self.pLim,-np.ones(self.nPctrl)*self.qLim,-np.ones(self.nT)*self.tLim)) )
        
        vLimLo = matrix( self.vLo - self.bV - self.Kc2v.dot(self.X0ctrl) )[self.vIn]
        vLimUp = matrix( self.vHi - self.bV - self.Kc2v.dot(self.X0ctrl) )[self.vIn]
        
        iLim = matrix( self.iScale*self.iXfmrLims - self.Kc2i.dot(self.X0ctrl) + self.bW )
        
        # recall: lower constraints by -1
        G = matrix( np.r_[np.eye(self.nCtrl),self.Kc2v[self.vIn],self.Kc2i,-np.eye(self.nCtrl),-self.Kc2v[self.vIn]] )
        h = matrix( np.r_[xLimUp,vLimUp,iLim,-xLimLo,-vLimLo] )

        if mode=='full' or mode=='loss' or mode=='load':
            sol = solvers.qp(Q,p,G,h)
        elif mode=='part':
            # EQUALITY CONSTRAINTS
            A0 = matrix( np.c_[np.zeros((self.nPctrl-1,self.nPctrl)),equalMat(self.nPctrl),np.zeros((self.nPctrl-1,self.nT))] )
            b0 = matrix( np.zeros(self.nPctrl - 1) )
            A1 = matrix( np.c_[equalMat(self.nPctrl),np.zeros((self.nPctrl-1,self.nPctrl)),np.zeros((self.nPctrl-1,self.nT))] )
            b1 = matrix( np.zeros(self.nPctrl - 1) )
            A = matrix([[A0,A1]])
            b = matrix([[b0,b1]])
            sol = solvers.qp(Q,p,G,h,A,b)
        elif mode=='maxTap':
            A = matrix( np.c_[np.eye(self.nPctrl*2),np.zeros((self.nPctrl*2,self.nT))] )
            b = matrix(np.zeros(self.nPctrl*2))
            sol = solvers.qp(Q,p,G,h,A,b)
        elif mode=='minTap':
            A = matrix( np.c_[np.eye(self.nPctrl*2),np.zeros((self.nPctrl*2,self.nT))] )
            b = matrix(np.r_[np.zeros(self.nPctrl),np.ones(self.nPctrl)*self.qLim] )
            sol = solvers.qp(Q,p,G,h,A,b)
            # if sol['status']=='unknown':
                # print('Unknown Mintap status - trying with MOSEK!')
                # minimize    c'*x
                # subject to  G*x + s = h
                #               A*x = b
                #               s >= 0
                # sol = solvers.lp(p,G,h,A,b,solver='mosek')
        self.sln = sol
        self.slnX = np.array(self.sln['x']).flatten()
        # pOut = slnX[:self.nPctrl]
        # qOut = slnX[self.nPctrl:self.nPctrl*2]
        # tOut = slnX[self.nPctrl*2:]
        
        # TL,PL,TC,V,I = self.runQp(slnX)
        self.slnF = self.runQp(self.slnX)
        # TL,PL,TC,V,I = self.slnF
    
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
    
    def showQpSln(self,slnX,slnF):
        
        pOut = slnX[:self.nPctrl]
        qOut = slnX[self.nPctrl:self.nPctrl*2]
        tOut = slnX[self.nPctrl*2:]
        
        TL,PL,TC,V,I = self.slnF
        TL0,PL0,TC0,V0,I0 = self.slnF0
        
        print('\n================== QP Solution:\nLoss before (kW):',TL0)
        print('Loss after (kW):',TL)
        print('Load before (kW):',PL0)
        print('Load after (kW):',PL)
        print('Curtailment before (kW):',TC0)
        print('Curtailment after (kW):',TC)
        print('Total power before (kW):', PL0 + TL0 + TC0)
        print('Total power after (kW):', PL + TL + TC)
        
        
        fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(11,4))
        
        # plot voltages versus voltage limits
        ax0.plot((V0/self.vKvbase)[self.vIn],'o');
        ax0.plot((V/self.vKvbase)[self.vIn],'x');
        ax0.plot((self.vHi/self.vKvbase)[self.vIn],'k_');
        ax0.plot((self.vLo/self.vKvbase)[self.vIn],'k_');
        ax0.set_title('Voltages')
        ax0.grid(True)
        # ax0.show()
        
        # plot currents versus current limits
        # ax1.plot(I0,'o')
        # ax1.plot(I,'x')
        # ax1.plot(self.iXfmrLims,'k_')
        # ax1.set_xlabel('Bus Index')
        # ax1.set_ylabel('Current (A)')
        # ax1.set_title('Currents')
        # ax1.grid(True)
        ax1.plot(I0/(self.iScale*self.iXfmrLims),'o')
        ax1.plot(I/(self.iScale*self.iXfmrLims),'x')
        # ax1.plot(self.iXfmrLims,'k_')
        ax1.plot(np.ones(len(self.iXfmrLims)),'k_')
        ax1.set_xlabel('Bus Index')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('Currents')
        ax1.grid(True)
        # ax1.show()
        
        ax2.plot(range(self.nPctrl),100*slnX[:self.nPctrl]/self.pLim,'x-',label='Pgen (%)')
        ax2.plot(range(self.nPctrl,self.nPctrl*2),100*slnX[self.nPctrl:self.nPctrl*2]/self.qLim,'x-',label='Qgen (%)')
        ax2.plot(range(self.nPctrl*2,self.nPctrl*2 + self.nT),100*slnX[self.nPctrl*2:]/self.tLim,'x-',label='t (%)')
        ax2.set_xlabel('Control Index')
        ax2.set_ylabel('Control effort, %')
        ax2.set_title('Control settings')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
    
    def testQpVcpf(self,k=np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        print('Start CVR testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        Convrg = []
        
        X0 = np.concatenate((self.xY[self.syIdx],self.xD))
        
        # NB: the makeCvrModel() is different from makeQpModel().
        dM = self.My.dot((self.xY*self.kYcvr0)[self.syIdx]) + self.Md.dot(self.xD*self.kDcvr0)
        dK = self.Ky.dot((self.xY*self.kYcvr0)[self.syIdx]) + self.Kd.dot(self.xD*self.kDcvr0)
        
        # lin_point*self.xY*self.kYcvr
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged) # for debugging
            
            vc0 = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            va0 = abs(vc0)
            
            X = k[i]*np.concatenate((self.xY[self.syIdx],self.xD))
            
            vcL = dM*k[i] + self.aV
            vaL = dK*k[i] + self.bV
            
            vce[i] = np.linalg.norm( (vcL - vc0)/self.vKvbase )/np.linalg.norm(vc0/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL - va0)/self.vKvbase )/np.linalg.norm(va0/self.vKvbase)
            
        print('nrelModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        
        fig,ax = plt.subplots()
        
        ax.plot(k,vce.real,label='vce');
        ax.plot(k,vae,label='vae');
        ax.set_xlabel('Continuation factor k');ax.grid(True)
        ax.set_title('Voltage error, '+self.feeder)
        ax.legend()
        return vce,vae,k
    
    def testCvrQp(self):
        # THREE TESTS in terms of the sensitivity of the model to real power generation, reactive power, and tap changes.
        
        print('Start CVR QP testing. \n',time.process_time())
        self.testQpVcpf()
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
        
        # do complex voltages over loadmult first.
        
        # do TAPS first.
        # Test 1. Putting taps up and down one. Things to check:
        # - voltages; currents; loads; losses; generation
        tapChng = [-2,1,1,1,1]
        dxScale = np.array([-2,-1,0,1,2])*0.1/16
        TL = np.zeros(5)
        TLest = np.zeros(5)
        TLcalc = np.zeros(5)
        PL = np.zeros(5)
        PLest = np.zeros(5)
        TC = np.zeros(5)
        TCest = np.zeros(5)
        vErr = np.zeros(5)
        iErr = np.zeros(5)
        DSSText.Command='Set controlmode=off'
        
        xCtrl = np.zeros(self.nCtrl)
        xCtrl[-self.nT::] = 1
        
        for i in range(5):
            # set all of the taps at one above
            j = DSSCircuit.RegControls.First
            while j:
                tapNo = DSSCircuit.RegControls.TapNumber
                if abs(tapNo)==16:
                    print('Sodding taps are saturated!')
                DSSCircuit.RegControls.TapNumber = tapChng[i]+tapNo
                j = DSSCircuit.RegControls.Next
            
            TG,TL[i],PL[i],YNodeV = runCircuit(DSSCircuit,DSSSolution)[1::]
            absYNodeV = abs(YNodeV[3:])
            Icalc = abs(self.v2iBrYxfmr.dot(YNodeV))[self.iXfmrModelled]
            TC[i] = -TG
            
            dx = xCtrl*dxScale[i]
            TLest[i],PLest[i],TCest[i],Vest,Iest = self.runQp(dx)
            
            vErr[i] = np.linalg.norm(absYNodeV - Vest)/np.linalg.norm(absYNodeV)
            iErr[i] = np.linalg.norm(Icalc - Iest)/np.linalg.norm(Icalc)
            
            VcplxEst = self.Mmid.dot(self.X0ctrl + dx) + self.aV
            YNodeVaug = np.concatenate((YNodeV[:3],VcplxEst,np.array([0])))
            iOut = self.v2iBrY.dot(YNodeVaug[:-1])
            vBusOut=YNodeVaug[list(self.yzW2V)]
            TLcalc[i] = sum(np.delete(1e-3*vBusOut*iOut.conj(),self.wregIdxs).real) # for debugging
        
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
        
        # Test 2. Put a whole load of generators in and change real and reactive powers.
        for ii in range(2):
            self.loadCvrDssModel(loadMult=self.currentLinPoint,pCvr=self.pCvr,qCvr=self.qCvr)
            genNamesY = add_generators(DSSObj,vecSlc(self.YZ,self.pyIdx[0]),False)
            genNamesD = add_generators(DSSObj,vecSlc(self.YZ,self.pdIdx[0]),True)
            genNames = genNamesY + genNamesD
            
            xCtrl = np.zeros(self.nCtrl)
            
            if ii==0:
                Sset = np.linspace(-1,1,100)*self.qLim
                xCtrl[self.nPctrl:-self.nT] = 1
            elif ii==1:
                Sset = np.linspace(-1,1,100)*self.pLim
                xCtrl[:self.nPctrl] = 1
            
            TL = np.zeros(len(Sset))
            TLest = np.zeros(len(Sset))
            PL = np.zeros(len(Sset))
            PLest = np.zeros(len(Sset))
            TC = np.zeros(len(Sset))
            TCest = np.zeros(len(Sset))
            vErr = np.zeros(len(Sset))
            iErr = np.zeros(len(Sset))
            
            for i in range(len(Sset)):
                if ii==0:
                    setGenPq(DSSCircuit,genNames,np.zeros(self.nPctrl),np.ones(self.nPctrl)*Sset[i]*1e-3)
                elif ii==1:
                    setGenPq(DSSCircuit,genNames,np.ones(self.nPctrl)*Sset[i]*1e-3,np.zeros(self.nPctrl))
                TG,TL[i],PL[i],YNodeV = runCircuit(DSSCircuit,DSSSolution)[1::]
                absYNodeV = abs(YNodeV[3:])
                Icalc = abs(self.v2iBrYxfmr.dot(YNodeV))[self.iXfmrModelled]
                TC[i] = -TG
                
                dx = xCtrl*Sset[i]
                TLest[i],PLest[i],TCest[i],Vest,Iest = self.runQp(dx)
                
                vErr[i] = np.linalg.norm(absYNodeV - Vest)/np.linalg.norm(absYNodeV)
                iErr[i] = np.linalg.norm(Icalc - Iest)/np.linalg.norm(Icalc)
            
            
            if ii==0:
                xlbl = 'Reactive power per load (kVar)'
            elif ii==1:
                xlbl = 'Real power per load (kW)'
            
            fig,[ax0,ax1,ax2,ax3,ax4] = plt.subplots(ncols=5,figsize=(11,5))
            ax0.plot(Sset*1e-3,TL,label='dss'); ax0.grid(True)
            ax0.plot(Sset*1e-3,TLest,label='apx')
            ax0.set_title('Losses (kW)'); ax0.set_xlabel(xlbl)
            ax0.legend()
            ax1.plot(Sset*1e-3,PL); ax1.grid(True)
            ax1.plot(Sset*1e-3,PLest)
            ax1.set_title('Load power (kW)'); ax1.set_xlabel(xlbl)
            ax2.plot(Sset*1e-3,TC); ax2.grid(True)
            ax2.plot(Sset*1e-3,TCest)
            ax2.set_title('Curtailment (kW)'); ax2.set_xlabel(xlbl)
            ax3.plot(Sset*1e-3,vErr); ax3.grid(True)
            ax3.set_title('Abs voltage error'); ax3.set_xlabel(xlbl)
            ax4.plot(Sset*1e-3,iErr); ax4.grid(True)
            ax4.set_title('Abs current error'); ax4.set_xlabel(xlbl)
            plt.tight_layout()
        plt.show()
        return
    
    def runQp(self,dx):
        TL = (dx.dot(self.qpQlss.dot(dx) + self.qpLlss) + self.qpClss)
        PL = dx.dot(self.ploadL) + self.ploadC
        TC = dx.dot(self.pcurtL)
        Vest = self.Kc2v.dot(dx) + self.Kc2v.dot(self.X0ctrl) + self.bV
        Iest = self.Kc2i.dot(dx + self.X0ctrl) + self.bW
        return TL,PL,TC,Vest,Iest

    def createNrelModel(self,lin_point=1.0):
        print('\nCreate NREL model, feeder:',self.feeder,'\nLin Point:',lin_point,'\nCap pos model:',self.setCapsModel,'\nCap Pos points:',self.capPosLin,'\n',time.process_time())
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=1 status=variable'
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = lin_point
        DSSSolution.Solve()
        print('\nNominally converged:',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        
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
            print('Time M:',time.time()-t,'\nCreate linear models Ky:\n',time.process_time()); t = time.time()
            Ky,b = nrelLinKy(My,Vh,self.xY*lin_point)
            print('Time K:',time.time()-t)
            Md = np.zeros((len(Vh),0), dtype=complex); Kd = np.zeros((len(Vh),0))
        else:
            print('Create linear models My + Md:\n',time.process_time()); t = time.time()
            My,Md,a = nrel_linearization( Ybus,Vh,V0,H )
            print('Time M:',time.time()-t,'\nCreate linear models Ky + Kd:\n',time.process_time()); t = time.time()
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
        print('\nNominally converged:',DSSSolution.Converged)
        
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
                xmfrImaxSet[TRN.Name].append(kva/(kv/np.sqrt(nPhases)))
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
        
        print('PD of loss model: ',pdTest(self.qpQlss))
        print('PD of loss model: ',pdTest(self.qpQlss + 1e-14*np.linalg.norm(self.qpQlss)*np.eye(len(self.qpQlss))))
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
        
        if self.feeder in ['epri24','8500node','123bus','epriJ1','epriK1','epriM1']: # if there is a regulator 'on' the source bus
            self.srcReg = 1
        else:
            self.srcReg = 0        
        
        if self.feeder in ['13bus','34bus']:
            self.plotMarkerSize=150
        elif self.feeder in ['eulv']:
            self.plotMarkerSize=100
        elif self.feeder[0]=='n':
            self.plotMarkerSize=25
        else:
            self.plotMarkerSize=50
            
            
        vMap = cm.RdBu
        pMap = cm.GnBu
        qMap = cm.PiYG
        sOnly = cm.Blues
        self.colormaps = { 'v0':vMap,'p0':pMap,'q0':qMap,'qSln':qMap,'vSln':vMap,'ntwk':sOnly }
    
    def setupConstraints(self,mvHi=1.05,lvHi=1.05,mvLo=0.925,lvLo=0.92,plim=1e3,qlim=1e3,tlim=0.1,iScale=2.0):
        self.hvBuses = (self.vKvbase>1000)
        self.lvBuses = (self.vKvbase<=1000)
    
        self.vIn = np.where((abs(self.aV)/self.vKvbase)>0.5)[0].tolist()
        self.vInYNodeOrder = vecSlc(self.vYNodeOrder,self.vIn)
        self.vInKvbase = self.vKvbase[self.vIn]
        
        self.vHi = 1.05*self.vKvbase
        self.vLo = mvLo*self.vKvbase*self.hvBuses + lvLo*self.vKvbase*self.lvBuses
        
        self.pLim = plim # only a lower bound (curtailment), W
        self.qLim = qlim # (k?)VAr
        self.tLim = tlim # pu
        
        self.iScale = iScale
        
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
            SN = os.path.join(self.SD,'plotNetwork','plotNetwork'+self.feeder)
            plt.savefig(SN+'.png',bbox_inches='tight', pad_inches=0.01)
            plt.savefig(SN+'.pdf',bbox_inches='tight', pad_inches=0)
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
            srcCoord = self.busCoords[self.vSrcBus+'z']
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
        elif busType=='s':
            bus0 = self.bus0s
        elif busType=='vIn':
            bus0 = self.bus0vIn

        setVals = {}
        setMin = 1e100
        setMax = -1e100
        setVals = {}
        for bus in busCoords: setVals[bus] = np.nan #initialise
            

        for bus in busCoords:
            if not np.isnan(busCoords[bus][0]):
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
    
    def plotNetBuses(self,type,regsOn=True,pltShow=True,minMax=None,pltType='mean',varMax=10,cmap=None):
        fig,ax = plt.subplots()
        self.getBusPhs()
        
        self.plotBranches(ax)
        self.plotSub(ax)
        self.plotRegs(ax)
        
        ttl = None
        if type=='v0':
            # scoreNom = self.bV/self.vKvbase
            # scores, minMax0 = self.getSetVals(scoreNom,pltType)
            scoreNom = self.slnF0[3][self.vIn]/self.vInKvbase
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
            scoreNom = 1e-3*np.r_[self.xY[self.syIdx][self.nPy:],self.xD[self.nPd:]]
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'s')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [-minMaxAbs,minMaxAbs]
            ttl = 'Qgen (kVAr)' # Positive implies capacitive
        elif type=='qSln':
            # scoreNom = 1e-3*(np.r_[self.xY[self.syIdx][self.nPy:],self.xD[self.nPd:]] + self.slnX[self.nPctrl:self.nSctrl])
            scoreNom = 1e-3*(self.slnX[self.nPctrl:self.nSctrl])
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'s')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [-minMaxAbs,minMaxAbs]
            ttl = 'Opt Qgen (kVAr)' # Positive implies capacitive
        elif type=='vSln':
            # TL,PL,TC,V,I = self.slnF
            scoreNom = self.slnF[3][self.vIn]/self.vInKvbase
            scores, minMax0 = self.getSetVals(scoreNom,pltType,'vIn')
            minMaxAbs = np.max(np.abs( np.array([np.nanmax(list(scores.values())),np.nanmin(list(scores.values()))]) ) )
            minMax0 = [0.92,1.05]
            ttl = 'Voltage (pu)' # Positive implies capacitive
            
        
        if cmap is None: cmap = self.colormaps[type]
        
        if minMax==None:
            minMax = minMax0
        self.plotBuses(ax,scores,minMax,cmap=cmap)
        
        # if type in ['v0','p0','q0']:
        self.plotNetColorbar(ax,minMax,cmap,ttl=ttl)
        
        ax.axis('off')
        plt.title('Feeder: '+self.feeder,loc='left')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        
        if pltShow:
            plt.show()
        else:
            self.currentAx = ax
    
    def plotNetColorbar(self,ax,minMax,cmap,ttl=None,nCbar=10):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = [xlim[0]-1]
        y0 = [ylim[0]-1]
        cntr = ax.contourf( np.array([x0*2,x0*2]),np.array([y0*2,y0*2]), np.array([minMax,minMax[::-1]]),nCbar,cmap=cmap )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        cbar = plt.colorbar(cntr,shrink=0.75)
        if ttl!=None:
            cbar.ax.set_title(ttl,pad=10,fontsize=10)
    
    
    def plotRegs(self,ax):
        if self.nT>0:
            i=0
            for regBus in self.regBuses:
                regCoord = self.busCoords[regBus.split('.')[0].lower()]
                if not np.isnan(regCoord[0]):
                    ax.plot(regCoord[0],regCoord[1],'r',marker=(6,1,0),zorder=+15)
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