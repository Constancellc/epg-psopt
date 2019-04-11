import numpy as np
import os, sys, win32com.client, time, pickle
from math import sqrt
from scipy import sparse
from scipy.linalg import block_diag
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_vlin_funcs import *
from dss_voltage_funcs import get_regIdx, getRegWlineIdx

WD = os.path.dirname(sys.argv[0])

from win32com.client import makepy
sys.argv=["makepy","OpenDSSEngine.DSS"]
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
DSSSolution.Tolerance=1e-7

# ------------------------------------------------------------ circuit info
test_model = True
test_model = False
test_model_bus = True
test_model_bus = False
saveModel = True
# saveModel = False
saveCc = True
saveCc = False
calcReg=1
# test_cap_model=1

setCaps=True
# fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# fdr_i_set = [5,6,8,0,14]
# fdr_i_set = [17,18,19,20,21]
# fdr_i_set = [9]
# fdr_i_set = [22]
fdr_i_set = [9]
for fdr_i in fdr_i_set:
    fig_loc=r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190117\\"
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']; lp_taps='Nmt'
    # feeder='213'
    feeder=fdrs[fdr_i]
    lp_taps='Lpt'

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)

    lin_points=np.array([0.3,0.6,1.0])
    lin_points=np.array([0.6])
    lin_points=np.array([lp0data['k']])
    # lin_points=np.array([1.0])

    k = np.arange(-1.5,1.6,0.1)
    # k = np.array([-1.5,-1.0,-0.5,0.0,0.3,lin_points[:],1.0,1.5]) # for speedier test model plotting
    # k = np.array([0.0,0.3,lin_points[:],1.0]) for test_model_bus

    ckt = get_ckt(WD,feeder)
    fn_ckt = ckt[0]
    fn = ckt[1]

    fn_y = fn+'_y'
    dir0 = WD + '\\lin_models\\' + feeder
    sn0 = dir0 + '\\' + feeder + lp_taps

    print('\nStart, feeder:',feeder,'\nSaving nom:',saveModel,'\nSaving cc:',saveCc,'\nLin Points:',lin_points,'\n',time.process_time())
    
    vve=np.zeros([k.size,lin_points.size])
    vae=np.zeros([k.size,lin_points.size])
    vvae=np.zeros([k.size,lin_points.size])
    vvae_cap=np.zeros([k.size,lin_points.size])
    DVslv_e=np.zeros([k.size,lin_points.size])

    for K in range(len(lin_points)):
        lin_point = lin_points[K]
        # run the dss
        DSSText.Command='Compile ('+fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'
        BB00,SS00 = cpf_get_loads(DSSCircuit,getCaps=setCaps)
        if lp_taps=='Nmt':
            TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        elif lp_taps=='Lpt':
            # cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=setCaps)
            cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=True)
            DSSSolution.Solve()
            TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        print('Load Ybus\n',time.process_time())
        
        # Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,fn_ckt,TC_No0 ) # for LV networks
        Ybus, YNodeOrder = create_tapped_ybus_very_slow( DSSObj,fn_y,TC_No0 )
        
        # print('Calculate condition no.:\n',time.process_time()) # for debugging
        # cndY = np.linalg.cond(Ybus.toarray())
        # print(np.log10(cndY))
        
        # CHECK 1: YNodeOrder 
        DSSText.Command='Compile ('+fn+'.dss)'
        YZ0 = DSSCircuit.YNodeOrder
        
        # Reproduce delta-y power flow eqns (1)
        DSSText.Command='Compile ('+fn+'.dss)'
        YNodeV0 = tp_2_ar(DSSCircuit.YNodeVarray)

        fix_tap_pos(DSSCircuit, TC_No0)
        DSSText.Command='Set Controlmode=off'
        # DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'
        DSSSolution.Solve()
        # BB00,SS00 = cpf_get_loads(DSSCircuit)
        
        Yvbase = get_Yvbase(DSSCircuit)[3:]

        cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=setCaps)
        DSSSolution.Solve()
        YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
        sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
        BB0,SS0 = cpf_get_loads(DSSCircuit,getCaps=setCaps)
        
        # cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=setCaps)
        # DSSSolution.Solve()
        # sYlin,sDlin = get_sYsD(DSSCircuit)[0:2]
        
        # cpf_set_loads(DSSCircuit,BB00,SS00,0.0,setCaps=setCaps)
        # DSSSolution.Solve()
        
        cpf_set_loads(DSSCircuit,BB00,SS00,0.0,setCaps=setCaps)
        DSSSolution.Solve()
        YNodeVnoLoad = tp_2_ar(DSSCircuit.YNodeVarray)
        
        cpf_set_loads(DSSCircuit,BB00,SS00,1.0,setCaps=True) # set back to 1
        cpf_set_loads(DSSCircuit,BB00,SS00,0.0,setCaps=False)
        DSSSolution.Solve()
        sYcap,sDcap = get_sYsD(DSSCircuit)[0:2]
        xhyCap0 = -1e3*s_2_x(sYcap[3:])
        xhdCap0 = -1e3*s_2_x(sDcap)
        if len(xhdCap0)==0 and len(sD)!=0:
            xhdCap0 = np.zeros(len(sD)*2)
        
        cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=False)
        DSSSolution.Solve()
        sYlds,sDlds = get_sYsD(DSSCircuit)[0:2]
        xhyLds = -1e3*s_2_x(sYlds[3:])
        xhdLds = -1e3*s_2_x(sDlds)
        
        dI = iTot + Ybus.dot(YNodeV)
        dIang0 = np.rad2deg(np.angle(iTot[iTot!=0]) - np.angle(Ybus.dot(YNodeV)[iTot!=0]) )
        dIang = np.mod(dIang0+270,180)-90
        
        # plt.semilogy(np.abs(dI)), plt.show()
        # chkc = abs(dI)/abs(iTot) # 1c needs checking outside
        # chkc_n = np.linalg.norm(iTot + Ybus.dot(YNodeV))/np.linalg.norm(iTot) # 1c needs checking outside
        # plt.plot( chkc[np.isinf(chkc)==False] ), plt.show()
        # --------------------
        xhy0 = -1e3*s_2_x(sY[3:])
        xhd0 = -1e3*s_2_x(sD) # not [3:] like sY!
        
        xhyLds0 = xhyLds - xhyCap0
        xhdLds0 = xhdLds - xhdCap0
        
        V0 = YNodeV[0:3]
        Vh = YNodeV[3:]
        VnoLoad = YNodeVnoLoad[3:]

        if len(H)==0:
            print('Create linear models My:\n',time.process_time())
            My,a = nrel_linearization_My( Ybus,Vh,V0 )
            print('Create linear models Ky:\n',time.process_time())
            Ky,b = nrel_linearization_Ky(My,Vh,sY)
            Vh0 = My.dot(xhy0) + a # for validation
        else:
            print('Create linear models M:\n',time.process_time())
            My,Md,a = nrel_linearization( Ybus,Vh,V0,H )
            print('Create linear models K:\n',time.process_time())
            Ky,Kd,b = nrel_linearization_K(My,Md,Vh,sY,sD)
            Vh0 = My.dot(xhy0) + Md.dot(xhd0) + a # for validation
        
        print('\nYNodeOrder Check - matching:',YZ0==YNodeOrder)
        print('\nMax abs(dI), Amps:',max(np.abs(dI[3:])))
        print('Max angle(dI), deg:',max(abs(dIang)))
        print('\nVoltage error (lin point), Volts:',np.linalg.norm(Vh0-Vh)/np.linalg.norm(Vh))
        print('Voltage error (no load point), Volts:',np.linalg.norm(a-VnoLoad)/np.linalg.norm(VnoLoad),'\n')
    
        DSSText.Command='Compile ('+fn+')'
        fix_tap_pos(DSSCircuit, TC_No0)
        DSSText.Command='Set controlmode=off'
        # DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'

        # NB!!! -3 required for models which have the first three elements chopped off!
        v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
        v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3 # NB: this is extremely slow! Try to load where possible
        v_idx = v_idx[v_idx>=0]
        YvbaseV = Yvbase[v_idx]
        
        p_idx = np.array(sY[3:].nonzero())
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
        
        # now, check these are working
        v_0 = np.zeros((len(k),len(YNodeOrder)),dtype=complex)
        vv_0 = np.zeros((len(k),len(v_idx)),dtype=complex)
        va_0 = np.zeros((len(k),len(YNodeOrder)))
        vva_0 = np.zeros((len(k),len(v_idx)))
        vva_0_cap = np.zeros((len(k),len(v_idx)))

        Vslv = np.zeros((len(k),len(YNodeOrder)-3),dtype=complex)
        
        vv_l = np.zeros((len(k),len(v_idx)),dtype=complex)
        vva_l = np.zeros((len(k),len(v_idx)))
        vva_l_cap = np.zeros((len(k),len(v_idx)))

        Convrg = []
        TP = np.zeros((len(lin_points),len(k)),dtype=complex)
        TL = np.zeros((len(lin_points),len(k)),dtype=complex)
        if test_model or test_model_bus:
            print('Start validation\n',time.process_time())
            for i in range(len(k)):
                print(i,'/',len(k))
                cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point,setCaps=setCaps)
                DSSSolution.Solve()
                Convrg.append(DSSSolution.Converged)
                TP[K,i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
                TL[K,i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
                
                v_0[i,:] = tp_2_ar(DSSCircuit.YNodeVarray)
                vv_0[i,:] = v_0[i,3:][v_idx]
                va_0[i,:] = abs(v_0[i,:])
                vva_0[i,:] = va_0[i,3:][v_idx]
                sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
                xhy = -1e3*s_2_x(sY[3:])
                
                # Vslv[i,:] = fixed_point_solve(Ybus,v_0[i,:],-1e3*sY[3:],-1e3*sD,H)
                if len(H)==0:
                    vv_l[i,:] = MyV.dot(xhy[s_idx]) + aV
                    vva_l[i,:] = KyV.dot(xhy[s_idx]) + bV
                else:
                    xhd = -1e3*s_2_x(sD) # not [3:] like sY
                    vv_l[i,:] = MyV.dot(xhy[s_idx]) + MdV.dot(xhd) + aV
                    vva_l[i,:] = KyV.dot(xhy[s_idx]) + KdV.dot(xhd) + bV

                # vae[i,K] = np.linalg.norm( (va_l[i,:] - va_0[i,3:])/Yvbase )/np.linalg.norm(va_0[i,3:]/Yvbase) # these are very slow for the bigger networks!
                
                vve[i,K] = np.linalg.norm( (vv_l[i,:] - vv_0[i,:])/YvbaseV )/np.linalg.norm(vv_0[i,:]/YvbaseV)
                vvae[i,K] = np.linalg.norm( (vva_l[i,:] - vva_0[i,:])/YvbaseV )/np.linalg.norm(vva_0[i,:]/YvbaseV)
                # DVslv_e[i,K] = np.linalg.norm( (Vslv[i,:] - v_0[i,3:]) )/np.linalg.norm(v_0[i,3:])
                
                # plt.plot(abs(v_l[i]/Yvbase),'rx-')
                # plt.plot(abs(v_0[i,3:]/Yvbase),'ko-')
        if 'test_cap_model' in locals():
            print('Start cap model validation\n',time.process_time())
            cpf_set_loads(DSSCircuit,BB0,SS0,1/lin_point,setCaps=True)
            for i in range(len(k)):
                print(i,'/',len(k))
                cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point,setCaps=False)
                DSSSolution.Solve()
                vva_0_cap[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][v_idx]
                sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
                xhyAct = -1e3*s_2_x(sY[3:])
                
                xhy = (xhyLds0*k[i]/lin_point) + xhyCap0
                
                if len(H)==0:
                    vva_l_cap[i,:] = KyV.dot(xhy[s_idx]) + bV
                else:
                    xhdAct = -1e3*s_2_x(sD) # not [3:] like sY
                    xhd = (xhdLds0*k[i]/lin_point) + xhdCap0
                    vva_l_cap[i,:] = KyV.dot(xhy[s_idx]) + KdV.dot(xhd) + bV
                vvae_cap[i,K] = np.linalg.norm( (vva_l_cap[i,:] - vva_0_cap[i,:])/YvbaseV )/np.linalg.norm(vva_0_cap[i,:]/YvbaseV)
        
        
        vYNodeOrder = vecSlc(YNodeOrder[3:],v_idx)
        SyYNodeOrder = vecSlc(YNodeOrder[3:],p_idx)[0]
        SdYNodeOrder = yzD
        
        if saveModel:
            header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
            lp_str = str(round(lin_point*100).astype(int)).zfill(3)
            if not os.path.exists(dir0):
                os.makedirs(dir0)
            np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
            np.save(sn0+'Ky'+lp_str+'.npy',KyV)
            np.save(sn0+'xhy0'+lp_str+'.npy',xhy0[s_idx])
            np.save(sn0+'bV'+lp_str+'.npy',bV)
            
            np.save(sn0+'xhyCap0'+lp_str+'.npy',xhyCap0[s_idx])
            np.save(sn0+'xhyLds0'+lp_str+'.npy',xhyLds0[s_idx])
            
            np.save(sn0+'v_idx'+lp_str+'.npy',v_idx)
            # np.save(sn0+'s_idx'+lp_str+'.npy',s_idx)
            # np.save(sn0+'p_idx'+lp_str+'.npy',p_idx)
            
            
            np.save(sn0+'vKvbase'+lp_str+'.npy',YvbaseV)
            np.save(sn0+'vYNodeOrder'+lp_str+'.npy',vYNodeOrder)
            np.save(sn0+'SyYNodeOrder'+lp_str+'.npy',SyYNodeOrder)
            np.save(sn0+'SdYNodeOrder'+lp_str+'.npy',SdYNodeOrder)
            
            
            if 'calcReg' in locals():
                np.save(sn0+'WyReg'+lp_str+'.npy',WyReg)
                np.save(sn0+'aIreg'+lp_str+'.npy',aIreg)
                np.save(sn0+'WregBus'+lp_str+'.npy',WregBus)
            
            if len(H)!=0:
                np.save(sn0+'Kd'+lp_str+'.npy',KdV)
                np.save(sn0+'xhd0'+lp_str+'.npy',xhd0)
                np.save(sn0+'xhdCap0'+lp_str+'.npy',xhdCap0)
                np.save(sn0+'xhdLds0'+lp_str+'.npy',xhdLds0)
                if 'calcReg' in locals():
                    np.save(sn0+'WdReg'+lp_str+'.npy',WdReg)
    
    print('Complete.\n',time.process_time())

    if test_model:
        plt.figure()
        plt.plot(k,vve), plt.title(feeder+', MyV error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.show()
        # plt.savefig('figB')
        # plt.figure()
        # plt.plot(k,vae), plt.title(feeder+', Ky error')
        # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        # plt.show()
        # # plt.savefig('figC')
        plt.figure()
        plt.plot(k,vvae), plt.title(feeder+', KyV error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.show()
        # plt.savefig('figD')
        # plt.figure()
        # plt.plot(k,DVslv_e), plt.title(feeder+', DVslv error')
        # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        # plt.show()
        # # plt.savefig('figE')
    if 'test_cap_model' in locals():
        plt.figure()
        plt.plot(k,vvae_cap), plt.title(feeder+', Cap model KyV error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.show()
        
        

    if saveCc:
        lp_str = str(round(lin_point*100).astype(int)).zfill(3)
        MyCC = My[:,s_idx]
        xhyCC = xhy0[s_idx]
        
        aCC = a
        V0CC = V0
        YbusCC = Ybus
        
        allLds = DSSCircuit.Loads.AllNames
        loadBuses = {}
        for ld in allLds:
            DSSCircuit.SetActiveElement('Load.'+ld)
            loadBuses[ld]=DSSCircuit.ActiveElement.BusNames[0]

        dirCC = WD + '\\lin_models\\ccModels\\' + feeder
        snCC = dirCC + '\\' + feeder + lp_taps

        if not os.path.exists(dirCC):
            os.makedirs(dirCC)
        
        np.save(snCC+'vYNodeOrder'+lp_str+'.npy',vYNodeOrder)
        np.save(snCC+'SyYNodeOrder'+lp_str+'.npy',SyYNodeOrder)
            
        np.save(snCC+'MyCc'+lp_str+'.npy',MyCC)
        np.save(snCC+'xhyCc'+lp_str+'.npy',xhyCC)
        np.save(snCC+'aCc'+lp_str+'.npy',aCC)
        np.save(snCC+'V0Cc'+lp_str+'.npy',V0CC)
        np.save(snCC+'YbusCc'+lp_str+'.npy',YbusCC)
        np.save(snCC+'YNodeOrderCc'+lp_str+'.npy',YNodeOrder)
        
        np.save(snCC+'loadBusesCc'+lp_str+'.npy',[loadBuses]) # nb likely to be similar to vecSlc(YNodeOrder[3:],p_idx)?
        # loadsDict = np.load(snCC+'loadBusesCc'+lp_str+'.npy')[0]

    
    if test_model_bus:

        # plt.plot(abs(vv_0[idxs[2]]),'o')
        # plt.plot(abs(vv_l[idxs[2]]),'x')
        # plt.show()

        idxs = np.array([0,1,2,3])

        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(2,len(idxs),i+1)
            plt.title('K = '+str(k[idxs[i]]))
            plt.plot(abs(vv_0[idxs[i]])/YvbaseV,'o')
            plt.plot(abs(vv_l[idxs[i]])/YvbaseV,'x')
            # plt.plot(abs(vv_0[idxs[i]] - vv_l[idxs[i]]),'o')
            plt.xlabel('Bus index'); plt.grid(True)
            plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); 
            plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude (pu)')
                plt.legend(('OpenDSS','Fixed Tap'))
                
            plt.subplot(2,len(idxs),i+1+4)
            plt.title('K = '+str(k[idxs[i]]))
            plt.plot(np.angle(vv_0[idxs[i]]),'o')
            plt.plot(np.angle(vv_l[idxs[i]]),'x')
            plt.xlabel('Bus index'); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Angle (rads)')
        plt.show()
        


# # stupid bug when solving, different voltages compared to opendss native V V V V
# DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
# DSSText=DSSObj.Text
# DSSCircuit = DSSObj.ActiveCircuit
# DSSSolution=DSSCircuit.Solution

# DSSText.Command='Compile ('+fn+'.dss)'
# YNodeOrder = DSSCircuit.YNodeOrder
# YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
# printBrI(YNodeOrder,abs(YNodeV)/1e3)