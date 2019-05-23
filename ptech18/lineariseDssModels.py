# This script is a script to build all of the linear models that we want to use for building the QP model in Q, t, P, etc. It borrows heavily from linearise_manc_py.m; the plan is to try and get rid of as much detritus as possible for this case now we understand a bit better what is going on...

import numpy as np
import os, sys, win32com.client, time, pickle
from scipy import sparse
from scipy.linalg import block_diag
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
# from dss_voltage_funcs import get_regIdx, getRegWlineIdx, get_reIdx, get_regVreg
from dss_voltage_funcs import *
import dss_stats_funcs as dsf

from win32com.client import makepy

class buildLinModel:
    def __init__(self,fdr_i=6,linPoints=[None],saveModel=False,setCapsModel='linPoint',FD=sys.argv[0],nrelTest=False):
        
        self.WD = os.path.dirname(FD)
        self.setCapsModel = setCapsModel
        
        sys.argv=["makepy","OpenDSSEngine.DSS"]
        makepy.main()
        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        
        self.dssStuff = [DSSObj,DSSObj.Text,DSSObj.ActiveCircuit,DSSObj.ActiveCircuit.Solution]
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy']
        
        self.feeder=fdrs[fdr_i]
        self.fn = get_ckt(self.WD,self.feeder)[1]
        
        with open(os.path.join(self.WD,'lin_models',self.feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
            lp0data = pickle.load(handle)
        
        if linPoints[0]==None:
            linPoints=[lp0data['k']]
        
        if self.setCapsModel=='linPoint':
            self.capPosLin=lp0data['capPosOut']
        else:
            self.capPosLin=None
        
        self.createCvrModel(linPoints[0],pCvr=0.75)
        vce,vae,dvce,dvae,TL,TLq,TLa,TLp,TL0,kN = self.cvrModelTest(k=np.linspace(-0.5,1.2,18))
        
        fig,ax = plt.subplots()
        ax.plot(kN,TL.real - TL.real,label='p=0.75 DSS');
        ax.plot(kN,TLq.real - TL.real,label='p=0.75 QP');
        ax.plot(kN,TL0.real - TL.real,label='p=0.0 QP');
        # ax.plot(kN,TLa2.real - 1e3*TL.real,label='p=0.75 DSS');
        # ax.plot(kN,TLp2.real - 1e3*TL.real,label='p=0.75 QP')
        ax.set_ylabel('Total loss,e')
        ax.set_xlabel('Continuation factor k')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        
        # fig,ax = plt.subplots()
        # ax.plot(kN,vce); ax.grid(True)
        # ax.plot(kN,vae); ax.grid(True)
        # ax.plot(kN,dvce,'--'); ax.grid(True)
        # ax.plot(kN,dvae,'--'); ax.grid(True)
        # ax.set_ylabel('Vc,e')
        # ax.set_xlabel('Continuation factor k')
        # plt.tight_layout()
        # plt.show()
        
        # fig,ax = plt.subplots()
        # ax.plot(kN,pe); ax.grid(True)
        # ax.plot(kN,p0e); ax.grid(True)
        # ax.set_ylabel('P,e')
        # ax.set_xlabel('Continuation factor k')
        # plt.tight_layout()
        # plt.show()
        
        
        # self.createNrelModel(linPoints[0])
        # self.createWmodel(linPoints[0])
        # self.createTapModel(linPoints[0])
        # vce,vae,kN = self.nrelModelTest(k=np.linspace(-0.5,1.2,18))
        
        # if DSSObj.ActiveCircuit.RegControls.Count>0:
            # self.createFxdModel(linPoints[0])
            # self.createLtcModel(linPoints[0])
            # TL,TLcalc,TLerr,ice,vceI,k = self.wModelTest(k=np.linspace(-0.5,1.2,100))
            # vFxdeLck, vLckeLck, vFxdeFxd, vLckeFxd, kFxd = self.fxdModelTest()
            # vFxdeLck, vLckeLck, vFxdeFxd, vLckeFxd, kFxd = self.ltcModelTest()
    
        # if nrelTest:
            # # Voltage errors
            # fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
            # ax0.plot(kN,abs(vce)); ax0.grid(True)
            # ax0.set_ylabel('Vc,e')
            # ax1.plot(kN,vae); ax1.grid(True)
            # ax1.set_ylabel('Va,e')
            # ax1.set_xlabel('Continuation factor k')
            # plt.tight_layout()
            # plt.show()
            
            # # # Tap change tests
            # # fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
            # # ax0.plot(kFxd,vLckeLck);
            # # ax0.plot(kFxd,vFxdeLck);
            # # ax0.grid(True)
            # # ax0.set_ylabel('Voltage Error')
            # # ax0.set_xlabel('Continuation factor k')
            # # ax1.plot(kFxd,vLckeFxd);
            # # ax1.plot(kFxd,vFxdeFxd);
            # # ax1.plot(kFxd,vLckeLck,'k--');
            # # ax1.grid(True)
            # # ax1.set_ylabel('Voltage Error')
            # # ax1.set_xlabel('Continuation factor k')
            # # plt.tight_layout()
            # # plt.show()

            # # # Real and imag losses
            # # fig,[ax0,ax1] = plt.subplots(2,figsize=(4,7),sharex=True)
            # # ax0.plot(k,TL.real)
            # # ax0.plot(k,TLcalc.real,'x'); ax0.grid(True)
            # # ax0.set_ylabel('Real power loss')
            # # ax1.plot(k,TL.imag)
            # # ax1.plot(k,TLcalc.imag,'x'); ax1.grid(True)
            # # ax1.set_ylabel('Imag power loss')
            # # ax1.set_xlabel('Continuation factor k')
            # # plt.tight_layout()
            # # plt.show()
        
            # # # Loss/current errors
            # # fig,[ax0,ax1,ax2] = plt.subplots(3,figsize=(4,8),sharex=True)
            # # ax0.plot(k,TLerr);
            # # ax0.grid(True)
            # # ax0.set_ylabel('Loss Error')
            # # ax0.set_xlabel('Continuation factor k')
            # # ax1.plot(k,ice);
            # # ax1.grid(True)
            # # ax1.set_ylabel('Current Error')
            # # ax1.set_xlabel('Continuation factor k')
            # # ax2.plot(k,vceI);
            # # ax2.grid(True)
            # # ax2.set_ylabel('Vprim error')
            # # ax2.set_xlabel('Continuation factor k')
            # # plt.tight_layout()
            # # plt.show()
        
        # # if saveModel:
            # # self.saveNrelModel()
            # # print('\nSaving Model')
        
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
        print('Ybus shape:',Ybus.shape)
        
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
    
    def createCvrModel(self,lin_point=1.0,pCvr=0.5,qCvr=0.0):
        # NOTE: opendss and the documentation have different definitions of CVR factor (!) one uses linear, one uses exponential. Results suggest tt seems to be an exponential model...?
        print('\nCreate NREL model, feeder:',self.feeder,'\nLin Point:',lin_point,'\npCvr, qCvr:',pCvr,qCvr,'\nCap Pos points:',self.capPosLin,'\n',time.process_time())
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(pCvr)+' cvrvars='+str(qCvr)
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = lin_point
        DSSSolution.Solve()
        print('\nNominally converged:',DSSSolution.Converged)
        
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        print('Ybus shape:',Ybus.shape)
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        self.loadCvrModel(pCvr,qCvr,loadMult=lin_point)
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
        
        # xYcvr = lin_point*self.xY*self.kYcvr
        # xDcvr = lin_point*self.xD*self.kDcvr
        # [sY0,sD0] = get_sYsD(DSSCircuit)[0:2] # useful for debugging
        # xY0 = -1e3*s_2_x(sY0[3:])
        # xD0 = -1e3*s_2_x(sD0)
        # err0 = xY0[:len(xYcvr)//2] - xYcvr[:len(xYcvr)//2] # ignore Q for now (coz of caps)
        # err1 = xY0[:len(xYcvr)//2] - self.xY[:len(xYcvr)//2]
        # # err0 = xD0 - xDcvr
        # # err1 = xD0 - self.xD
        # plt.plot(err0); plt.plot(err1); plt.show()
        
        # sY,sD,iY,iD,yzD,iTot,Hold = get_sYsD(DSSCircuit) # useful for debugging
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
        Ky,Kd,b = nrelLinK( My,Md,Vh,lin_point*self.xY,lin_point*self.xD )
        
        dKy,dKd,db = nrelLinK( dMy,dMd,dVh,lin_point*self.xY,lin_point*self.xD )
        
        print('Time K:',time.time()-t)
        
        Vh0 = (My.dot(self.xY) + Md.dot(self.xD))*lin_point + a # for validation
        Va0 = (Ky.dot(self.xY) + Kd.dot(self.xD))*lin_point + b # for validation
        dVh0 = (dMy.dot(self.xY) + dMd.dot(self.xD))*lin_point + da # for validation
        dVa0 = (dKy.dot(self.xY) + dKd.dot(self.xD))*lin_point + db # for validation
        
        # # Print checks:
        # print('\nVoltage clx error (lin point), Volts:',np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh))
        # print('Voltage clx error (no load point), Volts:',np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad),'\n') 
        # print('\nVoltage abs error (lin point), Volts:',np.linalg.norm(Va0-abs(Vh))/np.linalg.norm(abs(Vh))) 
        # print('Voltage abs error (no load point), Volts:',np.linalg.norm(abs(b)-abs(VnoLoad))/np.linalg.norm(abs(VnoLoad)),'\n') 
        # print('\n Delta voltage clx error (lin point), Volts:',np.linalg.norm(dVh0-dVh)/np.linalg.norm(dVh)) 
        # print('Delta voltage clx error (no load point), Volts:',np.linalg.norm(da-dVnoLoad)/np.linalg.norm(dVnoLoad),'\n') 
        # print('\nDelta voltage abs error (lin point), Volts:',np.linalg.norm(dVa0-abs(dVh))/np.linalg.norm(abs(dVh))) 
        # print('Delta voltage abs error (no load point), Volts:',np.linalg.norm(abs(db)-abs(dVnoLoad))/np.linalg.norm(abs(dVnoLoad)),'\n')
        
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
        
        self.pCvr = pCvr
        self.qCvr = qCvr
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
        self.yKPu = np.block( [[dsf.vmM(1/self.vKvbase,self.Ky),dsf.vmM(1/self.vKvbase,self.Kd)]] )
        self.dKPu = np.block( [[dsf.vmM(1/self.vKvbaseD,self.dKy),dsf.vmM(1/self.vKvbaseD,self.dKd)]] )
        
        self.bTotPu = np.concatenate( (self.bV/self.vKvbase,self.dbV/self.vKvbaseD) )
        self.bYpu = self.bV/self.vKvbase
        self.bDpu = self.dbV/self.vKvbaseD
        
        self.kYcvrK = np.concatenate( (self.pCvr*self.yKPu,self.qCvr*self.yKPu))
        self.kYcvr0 = np.concatenate(( (1-self.pCvr) + self.pCvr*self.bYpu,(1-self.qCvr) + self.qCvr*self.bYpu))
        
        self.kDcvrK = np.concatenate( (self.pCvr*self.dKPu,self.qCvr*self.dKPu))
        self.kDcvr0 = np.concatenate(( (1-self.pCvr) + self.pCvr*self.bDpu,(1-self.qCvr) + self.qCvr*self.bDpu))
        self.qpQ = iH.dot(np.concatenate( (self.kYcvrK[self.syIdx],self.kDcvrK) )) # kCvr K
        self.qpL = iH.dot(np.concatenate( (self.kYcvr0[self.syIdx],self.kDcvr0) )) # kCvr0 offset
        
        
    
    def loadDssModel(self,loadMult=1.0):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile ('+self.fn+')'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=1 status=variable'
        fix_tap_pos(DSSCircuit, self.TC_No0)
        fix_cap_pos(DSSCircuit, self.capPosLin)
        DSSText.Command='Set controlmode=off'
        DSSSolution.LoadMult = loadMult
        DSSSolution.Solve()
    
    def loadCvrModel(self,pCvr,qCvr,loadMult=1.0):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        DSSText.Command='Compile ('+self.fn+')'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=4 status=variable cvrwatts='+str(pCvr)+' cvrvars='+str(qCvr)
        fix_tap_pos(DSSCircuit, self.TC_No0)
        fix_cap_pos(DSSCircuit, self.capPosLin)
        DSSText.Command='Set controlmode=off'
        DSSSolution.LoadMult = loadMult
        DSSSolution.Solve()
        
    
    def createWmodel(self,linPoint=None):
        # >>> 4. For regulation problems
        if linPoint==None:
            self.loadDssModel(loadMult=self.lin_point)
        else:
            self.loadDssModel(loadMult=self.currentLinPoint)
            
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        branchNames = getBranchNames(DSSCircuit)
        print('Build Yprimmat',time.process_time())
        YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,branchNames)
        print('Build v2iBrY',time.process_time())
        v2iBrY = getV2iBrY(DSSCircuit,YprimMat,WbusSet)
        print('Complete',time.process_time())
        
        Wy = v2iBrY[:,3:].dot(self.My)
        Wd = v2iBrY[:,3:].dot(self.Md)
        aI = v2iBrY.dot(np.concatenate((self.V0,self.aV)))
        regWlineIdx,regIdx = getRegWlineIdx(DSSCircuit,WbusSet,WtrmlSet)
        
        self.v2iBrY = v2iBrY
        self.Wy = Wy
        self.Wd = Wd
        self.aI = aI
        self.WunqIdent = WunqIdent
        self.WbusSet = WbusSet
        self.WbrchSet = WbrchSet
        
        self.yzW2V = getYzW2V(self.WbusSet,DSSCircuit.YNodeOrder)
        
        WyReg = Wy[regWlineIdx,:]
        WdReg = Wd[regWlineIdx,:]
        aIreg = aI[list(regWlineIdx)]
        
        WregBus = vecSlc(WunqIdent,np.array(regWlineIdx))
        
        self.WyReg = WyReg
        self.aIreg = aIreg
        self.WregBus = WregBus
        self.WdReg = WdReg
        
        Vh0 = self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD) + self.aV # for validation
        
        IprimReg = WyReg.dot(self.xY[self.syIdx]) + WdReg.dot(self.xD) + aIreg # for debugging
        Iprim = Wy.dot(self.xY[self.syIdx]) + Wd.dot(self.xD) + aI # for debugging
        Iprim0 = v2iBrY.dot(np.concatenate((self.V0,Vh0))) # for debugging
        printBrI(WregBus,IprimReg) # for debugging. Note: there seem to be some differences between python and opendss native.
    
    def createTapModel(self,linPoint=None):
        if linPoint==None:
            self.loadDssModel(loadMult=self.lin_point)
        else:
            self.loadDssModel(loadMult=self.currentLinPoint)
            
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadDssModel(loadMult=linPoint)
        
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
            self.loadDssModel(loadMult=self.lin_point)
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
            self.loadDssModel(loadMult=self.lin_point)
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
        
    def nrelModelTest(self,k = np.arange(-1.5,1.6,0.1)):
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
        print('nrelModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        return vce,vae,k

    def cvrModelTest(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadCvrModel(pCvr=self.pCvr,qCvr=self.qCvr)
        
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
        
        dM = self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD)
        dK = self.Ky.dot(self.xY[self.syIdx]) + self.Kd.dot(self.xD)
        ddM = self.dMy.dot(self.xY[self.syIdx]) + self.dMd.dot(self.xD)
        ddK = self.dKy.dot(self.xY[self.syIdx]) + self.dKd.dot(self.xD)
        
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
        return vce,vae,dvce,dvae,TL,TLq,TLa,TLp,TL0,k
        
    def wModelTest(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadDssModel()
        
        print('Start W-model testing. \n',time.process_time())
        ice=np.zeros([k.size]) # current error
        vce=np.zeros([k.size]) # voltage error
        
        Convrg = []
        TLout = np.zeros((len(k)),dtype=complex)
        TLest = np.zeros((len(k)),dtype=complex)
        
        
        dMsl = np.concatenate((np.zeros(3,dtype=complex),self.My.dot(self.xY[self.syIdx]) + self.Md.dot(self.xD),np.zeros(1,dtype=complex)))[list(self.yzW2V)]
        aVsl = np.concatenate((self.V0,self.aV,np.array([0])))[list(self.yzW2V)]
        
        dW = self.Wy.dot(self.xY[self.syIdx]) + self.Wd.dot(self.xD)
        
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k)) # track progress
            
            DSSSolution.LoadMult = k[i]
            DSSSolution.Solve()
            
            Convrg.append(DSSSolution.Converged)
            
            TLout[i] = (DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1]) # in Watts
            
            vOut = np.concatenate((tp_2_ar(DSSCircuit.YNodeVarray),np.array([0])))
            iOut = self.v2iBrY.dot(vOut[:-1])
            vBusOut=vOut[list(self.yzW2V)]
            # TLest[i] = vBusOut.dot(iOut.conj()) # for debugging
            
            # now do the same with estimated quantities.
            iEst = dW*k[i] + self.aI
            vBusEst = dMsl*k[i] + aVsl
            TLest[i] = vBusEst.dot(iEst.conj()) # in Watts
            
            TLerr = abs(TLout - TLest)/abs(TLout)
            
            ice[i] = np.linalg.norm( (iOut - iEst) )/np.linalg.norm(iOut)
            vce[i] = np.linalg.norm( (vBusEst - vBusOut) )/np.linalg.norm(vBusOut)
        print('wModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        return TLout,TLest,TLerr,ice,vce,k
        
    def fxdModelTest(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
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
        print('fxdModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        # print('Testing Complete.\n',time.process_time())
        return vFxdeLck, vLckeLck, vFxdeFxd, vLckeFxd,k
    
    def ltcModelTest(self,k = np.concatenate((np.arange(-1.5,1.6,0.1),np.arange(1.5,-1.6,-0.1)))):
        # NB: Currently a direct port of fxdModelTest!
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        self.loadDssModel()
        
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
        print('fxdModelTest, converged:',100*sum(Convrg)/len(Convrg),'%')
        # print('Testing Complete.\n',time.process_time())
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