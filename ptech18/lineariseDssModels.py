# This script is a script to build all of the linear models that we want to use for building the QP model in Q, t, P, etc. It borrows heavily from linearise_manc_py.m; the plan is to try and get rid of as much detritus as possible for this case now we understand a bit better what is going on...

import numpy as np
import os, sys, win32com.client, time, pickle
from scipy import sparse
from scipy.linalg import block_diag
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
from dss_voltage_funcs import get_regIdx, getRegWlineIdx

from win32com.client import makepy

class buildLinModel:
    def __init__(self,fdr_i=6,linPoints=np.array([None]),saveModel=False,setCapsModel='linPoint',FD=sys.argv[0]):
        
        self.WD = os.path.dirname(FD)
        self.setCapsModel = setCapsModel
        
        sys.argv=["makepy","OpenDSSEngine.DSS"]
        makepy.main()
        DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        
        self.dssStuff = [DSSObj,DSSObj.Text,DSSObj.ActiveCircuit,DSSObj.ActiveCircuit.Solution]
        
        fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
        self.feeder=fdrs[fdr_i]
        
        with open(os.path.join(self.WD,'lin_models',self.feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
            lp0data = pickle.load(handle)
        
        if linPoints[0]==None:
            linPoints=np.array([lp0data['k']])
        
        if self.setCapsModel=='linPoint':
            self.capPosLin=lp0data['capPosOut']
        else:
            self.capPosLin=None
        
        
        self.createNrelModel(linPoints[0])
        vce,vae,k = self.nrelModelTest()
        
        plt.plot(k,abs(vce)); plt.grid(True); plt.show()
        plt.plot(k,vae); plt.grid(True); plt.show()
        
        if saveModel:
            self.saveNrelModel()
            print('\nSaving Model')
        
    def createNrelModel(self,lin_point=1.0):
        print('\nCreate NREL model, feeder:',self.feeder,'\nLin Point:',lin_point,'\nCap pos model:',self.setCapsModel,'\nCap Pos points:',self.capPosLin,'\n',time.process_time())
        
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        
        fn_ckt,fn = get_ckt(self.WD,self.feeder)
        fn_y = fn+'_y'
        dir0 = self.WD + '\\lin_models\\' + self.feeder
        sn0 = dir0 + '\\' + self.feeder
        
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'
        
        BB0,SS0 = cpf_get_loads(DSSCircuit)
        cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=self.setCapsModel,capPos=self.capPosLin)    
        DSSSolution.Solve()
        sYbstrd = get_sYsD(DSSCircuit)[0] # <---- for some annoying reason this gives different zeros to sY below; use for indexes
        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        self.BB0SS0 = [BB0,SS0]
        
        print('Load Ybus\n',time.process_time())
        # Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 ) # for LV networks
        Ybus, YNodeOrder = create_tapped_ybus_very_slow( DSSObj,fn_y,self.TC_No0 )
        print('Ybus shape:',Ybus.shape)
        
        # >>> 2. Reproduce delta-y power flow eqns (1)
        DSSText.Command='Compile ('+fn+'.dss)'
        YZ0 = DSSCircuit.YNodeOrder # CHECK 1: YNodeOrder 
        fix_tap_pos(DSSCircuit, self.TC_No0)
        DSSText.Command='Set Controlmode=off'
        DSSSolution.Solve()
        
        Yvbase = get_Yvbase(DSSCircuit)[3:]
        cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=self.setCapsModel,capPos=self.capPosLin)
        DSSSolution.Solve()
        
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
        BB,SS = cpf_get_loads(DSSCircuit,getCaps=self.setCapsModel)
        
        dI = iTot + Ybus.dot(YNodeV) # For checking how things are going
        dIang0 = np.rad2deg(np.angle(iTot[iTot!=0]) - np.angle(Ybus.dot(YNodeV)[iTot!=0]) )
        dIang = np.mod(dIang0+270,180)-90
        
        xhy0 = -1e3*s_2_x(sY[3:])
        xhd0 = -1e3*s_2_x(sD) # not [3:] like sY!
        
        cpf_set_loads(DSSCircuit,BB0,SS0,0.0,setCaps=self.setCapsModel,capPos=self.capPosLin)
        DSSSolution.Solve()
        YNodeVnoLoad = tp_2_ar(DSSCircuit.YNodeVarray)
        
        V0 = YNodeV[0:3]
        Vh = YNodeV[3:]
        VnoLoad = YNodeVnoLoad[3:]
        
        # >>> 3. Get capacitor and load powers separately
        cpf_set_loads(DSSCircuit,BB0,SS0,1.0,setCaps=True) # set caps back to 1
        cpf_set_loads(DSSCircuit,BB0,SS0,0.0,setCaps=self.setCapsModel,capPos=self.capPosLin)
        DSSSolution.Solve()

        sYcap,sDcap = get_sYsD(DSSCircuit)[0:2]
        xhyCap0 = -1e3*s_2_x(sYcap[3:])
        xhdCap0 = -1e3*s_2_x(sDcap)
        if len(xhdCap0)==0 and len(sD)!=0:
            xhdCap0 = np.zeros(len(sD)*2)
        
        cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=self.setCapsModel,capPos=self.capPosLin)
        DSSSolution.Solve()
        sYlds,sDlds = get_sYsD(DSSCircuit)[0:2]
        xhyLds = -1e3*s_2_x(sYlds[3:])
        xhdLds = -1e3*s_2_x(sDlds)
        
        xhyLds0 = xhyLds - xhyCap0
        xhdLds0 = xhdLds - xhdCap0
        
        # >>> 4. Create linear models for voltage in S
        if len(H)==0:
            print('Create linear models My:\n',time.process_time());  t = time.time()
            My,a = nrel_linearization_My( Ybus,Vh,V0 )
            print('Time M:',time.time()-t,'\nCreate linear models Ky:\n',time.process_time()); t = time.time()
            Ky,b = nrel_linearization_Ky(My,Vh,sY)
            print('Time K:',time.time()-t)
            Vh0 = My.dot(xhy0) + a # for validation
        else:
            print('Create linear models My + Md:\n',time.process_time()); t = time.time()
            My,Md,a = nrel_linearization( Ybus,Vh,V0,H )
            print('Time M:',time.time()-t,'\nCreate linear models Ky + Kd:\n',time.process_time()); t = time.time()
            Ky,Kd,b = nrel_linearization_K(My,Md,Vh,sY,sD)
            print('Time K:',time.time()-t)
            Vh0 = My.dot(xhy0) + Md.dot(xhd0) + a # for validation
        
        # Print various checks.
        print('\nYNodeOrder Check - matching:',YZ0==YNodeOrder)
        print('\nMax abs(dI), Amps:',max(np.abs(dI[3:])))
        print('Max angle(dI), deg:',max(abs(dIang)))
        print('\nVoltage error (lin point), Volts:',np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh))
        print('Voltage error (no load point), Volts:',np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad),'\n')
    
        # >>> 5. Load only the voltages that we want to use
        DSSText.Command='Compile ('+fn+')'
        fix_tap_pos(DSSCircuit, self.TC_No0)
        DSSText.Command='Set controlmode=off'
        # DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'

        # NB!!! -3 required for models which have the first three elements chopped off!
        v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
        v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3 # NB: this is extremely slow! Try to load where possible
        v_idx = v_idx[v_idx>=0]
        YvbaseV = Yvbase[v_idx]
        
        # p_idx = np.array(sY[3:].nonzero())
        p_idx = np.array(sYbstrd[3:].nonzero()) # this gives ever ever so slightly different answers to sY (see EPRI M1)
        s_idx = np.concatenate((p_idx,p_idx+len(sY)-3),axis=1)[0]
        
        MyV = My[v_idx,:][:,s_idx]
        aV = a[v_idx]
        KyV = Ky[v_idx,:][:,s_idx]
        bV = b[v_idx]
        
        if len(H)!=0: # already gotten rid of s_idx
            MdV = Md[v_idx,:]
            KdV = Kd[v_idx,:]
        
        # For regulation problems
        if 'calcReg' in locals():
            branchNames = getBranchNames(DSSCircuit)
            print('Build Yprimmat',time.process_time())
            YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,branchNames)
            print('Build v2iBrY',time.process_time())
            
            v2iBrY = getV2iBrY(DSSCircuit,YprimMat,WbusSet)
            print('Complete',time.process_time())
            Wy = v2iBrY[:,3:].dot(My)
            aI = v2iBrY.dot(np.concatenate((V0,a)))
            if len(H)!=0: # already gotten rid of s_idx
                Wd = v2iBrY[:,3:].dot(Md)
            regWlineIdx,regIdx = getRegWlineIdx(DSSCircuit,WbusSet,WtrmlSet)
            
            WyReg = Wy[regWlineIdx,:][:,s_idx]
            aIreg = aI[list(regWlineIdx)]
            WregBus = vecSlc(WunqIdent,np.array(regWlineIdx))
            if len(H)!=0: # already gotten rid of s_idx
                WdReg = Wd[regWlineIdx,:]
        
        # IprimReg = WyReg.dot(xhy0[s_idx]) + WdReg.dot(xhd0) + aIreg # for debugging
        # Iprim = Wy.dot(xhy0) + Wd.dot(xhd0) + aI # for debugging
        # Iprim0 = v2iBrY.dot(np.concatenate((V0,Vh0))) # for debugging
        # printBrI(WregBus,IprimReg) # for debugging. Note: there seem to be some differences between python and opendss native.
    
        # SAVING all of the relevant bits and pieces
        vYNodeOrder = vecSlc(YNodeOrder[3:],v_idx)
        SyYNodeOrder = vecSlc(YNodeOrder[3:],p_idx)[0]
        SdYNodeOrder = yzD
        
        self.My = MyV
        self.aV = aV
        
        self.Ky = KyV
        self.xhy0 = xhy0[s_idx]
        self.bV = bV
        
        self.xhyCap0 = xhyCap0[s_idx]
        self.xhyLds0 = xhyLds0[s_idx]
        self.v_idx = v_idx
        
        self.s_idx = s_idx # although, this is not actually saved in the end
        
        self.vKvbase = YvbaseV
        self.vYNodeOrder = vYNodeOrder
        self.SyYNodeOrder = SyYNodeOrder
        self.SdYNodeOrder = SyYNodeOrder
        if 'calcReg' in locals():
            self.WyReg = WyReg
            self.aIreg = aIreg
            self.WregBus = WregBus
        if len(H)!=0:
            self.Md = MdV
            self.Kd = KdV
            self.xhd0 = xhd0
            self.xhdCap0 = xhdCap0
            self.xhdLds0 = xhdLds0
            if 'calcReg' in locals():
                np.save(sn0+'WdReg'+lp_str+'.npy',WdReg)
                self.WdBus = WdBus

    def nrelModelTest(self,k = np.arange(-1.5,1.6,0.1)):
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        BB0,SS0 = self.BB0SS0
        
        print('Start nrel testing. \n',time.process_time())
        vce=np.zeros([k.size])
        vae=np.zeros([k.size])
        vc0 = np.zeros((len(k),len(self.v_idx)),dtype=complex)
        va0 = np.zeros((len(k),len(self.v_idx)))
        vcL = np.zeros((len(k),len(self.v_idx)),dtype=complex)
        vaL = np.zeros((len(k),len(self.v_idx)))
        Convrg = []
        TP = np.zeros((len(k)),dtype=complex)
        TL = np.zeros((len(k)),dtype=complex)
        for i in range(len(k)):
            if (i % (len(k)//4))==0: print('Solution:',i,'/',len(k))
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i],setCaps=self.setCapsModel,capPos=self.capPosLin)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1] # for debugging
            TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1]) # for debugging
            
            vOut = tp_2_ar(DSSCircuit.YNodeVarray)[3:][self.v_idx]
            vc0[i,:] = vOut
            va0[i,:] = abs(vOut)
            
            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])[self.s_idx]
            
            if len(H)==0:
                vcL[i,:] = self.My.dot(xhy) + self.aV
                vaL[i,:] = self.Ky.dot(xhy) + self.bV
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                vcL[i,:] = self.My.dot(xhy) + self.Md.dot(xhd) + self.aV
                vaL[i,:] = self.Ky.dot(xhy) + self.Kd.dot(xhd) + self.bV
            
            vce[i] = np.linalg.norm( (vcL[i,:] - vc0[i,:])/self.vKvbase )/np.linalg.norm(vc0[i,:]/self.vKvbase)
            vae[i] = np.linalg.norm( (vaL[i,:] - va0[i,:])/self.vKvbase )/np.linalg.norm(va0[i,:]/self.vKvbase)
        return vce,vae,k

    def saveNrelModel():
        header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
        lp_str = str(round(lin_point*100).astype(int)).zfill(3)
        if not os.path.exists(dir0):
            os.makedirs(dir0)        
        np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
        
        np.save(sn0+'Ky'+lp_str+'.npy',self.Ky)
        np.save(sn0+'xhy0'+lp_str+'.npy',self.xhy0)
        np.save(sn0+'bV'+lp_str+'.npy',self.bV)
        
        np.save(sn0+'xhyCap0'+lp_str+'.npy',self.xhyCap0)
        np.save(sn0+'xhyLds0'+lp_str+'.npy',self.xhyLds0)
        
        np.save(sn0+'v_idx'+lp_str+'.npy',self.v_idx)
        
        np.save(sn0+'vKvbase'+lp_str+'.npy',self.vKvbase)
        np.save(sn0+'vYNodeOrder'+lp_str+'.npy',self.vYNodeOrder)
        np.save(sn0+'SyYNodeOrder'+lp_str+'.npy',self.SyYNodeOrder)
        np.save(sn0+'SdYNodeOrder'+lp_str+'.npy',self.SdYNodeOrder)
        if 'calcReg' in locals():
            np.save(sn0+'WyReg'+lp_str+'.npy',self.WyReg)
            np.save(sn0+'aIreg'+lp_str+'.npy',self.aIreg)
            np.save(sn0+'WregBus'+lp_str+'.npy',self.WregBus)
        
        if len(H)!=0:
            np.save(sn0+'Kd'+lp_str+'.npy',self.KdV)
            np.save(sn0+'xhd0'+lp_str+'.npy',self.xhd0)
            np.save(sn0+'xhdCap0'+lp_str+'.npy',self.xhdCap0)
            np.save(sn0+'xhdLds0'+lp_str+'.npy',self.xhdLds0)
            if 'calcReg' in locals():
                np.save(sn0+'WdReg'+lp_str+'.npy',WdReg)
                self.WdBus = WdBus            

        
        # if 'test_cap_model' in locals():
            # vvae_cap=np.zeros([k.size])
            # vva_0_cap = np.zeros((len(k),len(v_idx)))
            # vva_l_cap = np.zeros((len(k),len(v_idx)))
            # print('Start cap model validation\n',time.process_time())
            # cpf_set_loads(DSSCircuit,BB0,SS0,1,setCaps=True,capPos=None)
            # for i in range(len(k)):
                # print(i,'/',len(k))
                # cpf_set_loads(DSSCircuit,BB,SS,k[i]/lin_point,setCaps=self.setCapsModel,capPos=self.capPosLin)
                # DSSSolution.Solve()
                # vva_0_cap[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][v_idx]
                # sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
                # xhyAct = -1e3*s_2_x(sY[3:])
                
                # xhy = (xhyLds0*k[i]/lin_point) + xhyCap0
                
                # if len(H)==0:
                    # vva_l_cap[i,:] = KyV.dot(xhy[s_idx]) + bV
                # else:
                    # xhdAct = -1e3*s_2_x(sD) # not [3:] like sY
                    # xhd = (xhdLds0*k[i]/lin_point) + xhdCap0
                    # vva_l_cap[i,:] = KyV.dot(xhy[s_idx]) + KdV.dot(xhd) + bV
                # vvae_cap[i] = np.linalg.norm( (vva_l_cap[i,:] - vva_0_cap[i,:])/YvbaseV )/np.linalg.norm(vva_0_cap[i,:]/YvbaseV)
            # xhyCap0[xhyCap0!=0]
            # xhdCap0[xhdCap0!=0]
        













