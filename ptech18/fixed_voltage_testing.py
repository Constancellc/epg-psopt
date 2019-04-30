# Testing the fixed voltage method.

# steps: 
# 1. load linear model
# 2. split into upstream/downstream of regulator(s)
# 3. reorder & remove elements as appropriate
# 4. run continuation analysis.

import time, pickle, win32com.client, sys, os
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_voltage_funcs import *
from scipy import sparse
from cvxopt import spmatrix
from scipy import random
import scipy.linalg as spla
from win32com.client import makepy

# based on monte_carlo.py
WD = os.path.dirname(sys.argv[0])

sys.argv=["makepy","OpenDSSEngine.DSS"]
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

# ------------------------------------------------------------ circuit info
test_model_plt = True
test_model_plt = False
test_model_bus = True
test_model_bus = False
test_model_dff = True
test_model_dff = False
save_model=True
# save_model=False

setCapsModel='linPoint'

fdr_i_set = [5,6,8,9,22,19,20,21]
fdr_i_set = [5,6,8,9,19,20,21,22]
fdr_i_set = [6,8,9,19,20,21,22]
fdr_i_set = [9]
# fdr_i_set = [6,8,19]
for fdr_i in fdr_i_set:
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy']
    feeder=fdrs[fdr_i]

    k = np.arange(-1.5,1.6,0.025)
    # k = np.arange(1.6,-1.5,-0.025)
    k = np.arange(-1.5,1.6,0.1)
    # k = np.arange(-1.5,1.6,0.3)
    # k = np.arange(0,1.0,1.0)

    ckt = get_ckt(WD,feeder)
    fn_ckt = ckt[0]
    fn = ckt[1]
    lin_point=0.6
    lin_point=False # use this if wanting to use the nominal point from chooseLinPoint.
    lp_taps='Lpt'
    print('Start. Feeder:',feeder,'Lin Point:',lp_taps,'. NB: Kt and Ky should be up to date.\n',time.process_time())

    fn_y = fn+'_y'
    sn0 = WD + '\\lin_models\\' + feeder

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)
    if not lin_point:
        lin_point=lp0data['k']
    
    if setCapsModel=='linPoint':
        capPosLin=lp0data['capPosOut']
    else:
        print('Warning! not using linPoint, not implemented.')


    # 1. Load files' find nominal voltages, node orders, linear model
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    nRegs = DSSCircuit.RegControls.Count # NB is not necessarily the same as the number of transformers (e.g. 123 bus)
    # DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3'
    if lp_taps=='Lpt':
        cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=setCapsModel,capPos=capPosLin)
        DSSSolution.Solve()
    YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)

    DSSText.Command='set controlmode=off'
    YZ = DSSCircuit.YNodeOrder

    LM = loadLinMagModel(feeder,lin_point,WD,lp_taps)
    Ky=LM['Ky'];Kd=LM['Kd'];Kt=LM['Kt'];bV=LM['bV'];
    xhy0=LM['xhy0'];xhd0=LM['xhd0'];xhyCap0=LM['xhyCap0'];xhdCap0=LM['xhdCap0'];xhyLds0=LM['xhyLds0'];xhdLds0=LM['xhdLds0']

    
    print('Get zone list...',time.process_time())
    # 2. get the regulator zones for each regulator. (I think this still needs work?)
    regIdx,regBus = get_regIdx(DSSCircuit)
    reIdx = (np.array(get_reIdx(regIdx,len(YZ))[3:])-3).tolist()

    # 3. get index shifts using zone info
    v_idx = LM['v_idx']
    v_idx_shf,v_idx_new = idx_shf(v_idx,reIdx)

    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)

    p_idx_yz = np.array(sY[3:].nonzero())
    p_idx_shf,p_idx_new = idx_shf(p_idx_yz[0],reIdx)
    s_idx_shf = np.concatenate((p_idx_shf,p_idx_shf+len(p_idx_shf)))
    s_idx = np.concatenate((p_idx_yz,p_idx_yz+len(sY)-3),axis=1)[0] # used for comparison with 'before'
    s_idx_new = np.concatenate((p_idx_new,p_idx_new+len(sY)-3))

    yzI = yzD2yzI(yzD,node_to_YZ(DSSCircuit))
    yzI_shf,yzI_new = idx_shf(yzI,reIdx)
    # yzI = (np.array(yzI) - 3).tolist() # convert to the correct index numbers.

    YZp = vecSlc(YZ[3:],p_idx_new) # verified
    YZd = vecSlc(YZ,yzI_new) 
    Yvbase_new = get_Yvbase(DSSCircuit)[3:][v_idx_new]
    sD_idx_shf = np.concatenate((yzI_shf,yzI_shf+len(yzI_shf)))
    
    regVreg = get_regVreg(DSSCircuit)
    # 4. Perform Kron reduction with these indices
    idxShf = [v_idx_shf,s_idx_shf,sD_idx_shf]
    Akron, Bkron = lmKronRed(LM,idxShf,regVreg)
    
    # 5. Test if these are working
    ve=np.zeros([k.size])
    veN=np.zeros([k.size])

    ve_ctl=np.zeros([k.size])
    veN_ctl=np.zeros([k.size])
    vvae_cap=np.zeros([k.size])

    v_0 = np.zeros((len(k),len(YZ)))

    vv_0 = np.zeros((len(k),len(v_idx)))
    vv_0R = np.zeros((len(k),len(v_idx)))
    vv_0_ctl = np.zeros((len(k),len(v_idx)))
    vv_0R_ctl = np.zeros((len(k),len(v_idx)))

    vv_l = np.zeros((len(k),len(v_idx)))
    vv_lN = np.zeros((len(k),len(v_idx)))
    vv_l_ctr = np.zeros((len(k),len(v_idx)))
    vv_lN_ctr = np.zeros((len(k),len(v_idx)))
    vv_lN_cap = np.zeros((len(k),len(v_idx)))

    RegSat = np.zeros((len(k),nRegs),dtype=int)

    Convrg = []
    TP = np.zeros(len(k),dtype=complex)
    TL = np.zeros(len(k),dtype=complex)
    if test_model_plt or test_model_bus or test_model_dff:
        print('--- Start Testing, 1/2 --- \n',time.process_time())
        for i in range(len(k)):
            print(i,'/',len(k)-1)
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i],setCaps=setCapsModel,capPos=capPosLin)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
            TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])
            
            v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))
            vv_0[i,:] = v_0[i,3:][v_idx]
            vv_0R[i,:] = v_0[i,3:][v_idx_new]
            
            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])
            
            if len(H)==0:
                vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
                # vv_lN[i,:] = Akron.dot(xnew) + Bkron
                vv_lN[i,:] = Akron.dot(xhy[s_idx]) + Bkron
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
                
                xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
                vv_lN[i,:] = Akron.dot(xnew) + Bkron
            
            ve[i] = np.linalg.norm( vv_l[i,:] - vv_0[i,:] )/np.linalg.norm(vv_0[i,:])
            veN[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:])
        print('--- Start Testing, 2/2 --- \n',time.process_time())
        DSSText.Command='set controlmode=static'
        for i in range(len(k)):
            print(i,'/',len(k)-1)
            cpf_set_loads(DSSCircuit,BB0,SS0,k[i],setCaps=setCapsModel,capPos=capPosLin)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
            TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])

            v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))
            vv_0_ctl[i,:] = v_0[i,3:][v_idx]
            vv_0R_ctl[i,:] = vv_0_ctl[i,:][v_idx_shf]

            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])

            RegSat[i] = getRegSat(DSSCircuit)

            if len(H)==0:
                vv_l_ctr[i,:] = Ky.dot(xhy[s_idx]) + bV
                vv_lN_ctr[i,:] = Akron.dot(xhy[s_idx_new]) + Bkron
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                vv_l_ctr[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
                
                xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
                vv_lN_ctr[i,:] = Akron.dot(xnew) + Bkron
                
            ve_ctl[i] = np.linalg.norm( vv_l_ctr[i,:] - vv_0_ctl[i,:] )/np.linalg.norm(vv_0_ctl[i,:])
            veN_ctl[i] = np.linalg.norm( vv_lN_ctr[i,:] - vv_0R_ctl[i,:] )/np.linalg.norm(vv_0R_ctl[i,:])
        print('Testing Complete.\n',time.process_time())
        
    
    # if 'test_cap_model' in locals(): # <--- not implemented
        # vva_0_cap = np.zeros((len(k),len(v_idx)))
        # vva_l_cap = np.zeros((len(k),len(v_idx)))
        # print('Start cap model validation\n',time.process_time())
        # # cpf_set_loads(DSSCircuit,BB0,SS0,1/lin_point,setCaps=True)
        # cpf_set_loads(DSSCircuit,BB0,SS0,1,setCaps=True)
        # for i in range(len(k)):
            # print(i,'/',len(k))
            # # cpf_set_loads(DSSCircuit,BB0,SS0,k[i]/lin_point,setCaps=False)
            # cpf_set_loads(DSSCircuit,BB0,SS0,k[i],setCaps=setCapsModel,capPos=capPosLin)
            # DSSSolution.Solve()
            # vva_0_cap[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))[3:][v_idx]
            # sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            # xhyAct = -1e3*s_2_x(sY[3:])
            
            # xhy = (xhyLds0*k[i]/lin_point) + xhyCap0
            
            # if len(H)==0:
                # vv_lN_cap[i,:] = Akron.dot(xhy[s_idx_new]) + Bkron
            # else:
                # xhdAct = -1e3*s_2_x(sD) # not [3:] like sY
                # xhd = (xhdLds0*k[i]/lin_point) + xhdCap0
                # xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
                # vv_lN_cap[i,:] = Akron.dot(xnew) + Bkron
            # vvae_cap[i,K] = np.linalg.norm( (vv_lN_cap[i,:] - vva_0_cap[i,:])/YvbaseV )/np.linalg.norm(vva_0_cap[i,:]/YvbaseV)
        # plt.figure()
        # plt.plot(k,vvae_cap), plt.title(feeder+', Cap model KyV error')
        # plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        # plt.show()
        
        
    unSat = RegSat.min(axis=1)==1
    sat = RegSat.min(axis=1)==0
    # Yvbase = get_Yvbase(DSSCircuit)[3:][v_idx]

    # SAVE MODEL ============
    if save_model:
        dir0 = WD + '\\lin_models\\' + feeder + '\\fxd_model'
        sn0 = dir0 + '\\' + feeder + lp_taps + 'Fxd'
        lp_str = str(round(lin_point*100).astype(int)).zfill(3)
        if not os.path.exists(dir0):
                os.makedirs(dir0)

        np.save(sn0+'A'+lp_str+'.npy',Akron)
        np.save(sn0+'B'+lp_str+'.npy',Bkron)
        np.save(sn0+'s_idx'+lp_str+'.npy',s_idx_new)
        np.save(sn0+'v_idx'+lp_str+'.npy',v_idx_new)
        np.save(sn0+'xhy0'+lp_str+'.npy',xhy0[s_idx_shf])
        np.save(sn0+'xhd0'+lp_str+'.npy',xhd0[sD_idx_shf])
        np.save(sn0+'xhyCap0'+lp_str+'.npy',xhyCap0[s_idx_shf])
        np.save(sn0+'xhdCap0'+lp_str+'.npy',xhdCap0[sD_idx_shf])
        np.save(sn0+'xhyLds0'+lp_str+'.npy',xhyLds0[s_idx_shf])
        np.save(sn0+'xhdLds0'+lp_str+'.npy',xhdLds0[sD_idx_shf])
        np.save(sn0+'vYNodeOrder'+lp_str+'.npy',vecSlc(YZ[3:],v_idx_new))
        np.save(sn0+'Vbase'+lp_str+'.npy',Yvbase_new)

        np.save(sn0+'SyYNodeOrder'+lp_str+'.npy',YZp)
        np.save(sn0+'SdYNodeOrder'+lp_str+'.npy',YZd)
        
        np.save(sn0+'regVreg'+lp_str+'.npy',regVreg)
        np.save(sn0+'idxShf'+lp_str+'.npy',idxShf)


    if test_model_plt:
        plt.figure()
        pltA, = plt.plot(k,ve,'b')
        pltB, = plt.plot(k,veN,'r') # plt.plot(k,veN)
        plt.title(feeder+', K error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.legend([pltA,pltB],['Lin fixed','Lin not fixed'])
        plt.show()
        
        plt.figure()
        plt.plot(k,ve,'k:')
        pltA, = plt.plot(k[unSat],ve_ctl[unSat],'b') 
        pltB, = plt.plot(k[unSat],veN_ctl[unSat],'r')
        plt.plot(k[sat],ve_ctl[sat],'b.')
        plt.plot(k[sat],veN_ctl[sat],'r.')
        plt.title(feeder+', K error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.legend([pltA,pltB],['Lin fixed','Lin not fixed'])
        plt.show()


    if test_model_bus:
        krnd = np.around(k,5) # this breaks at 0.000 for the error!
        idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )
        
        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(1,len(idxs),i+1)
            plt.title('K = '+str(krnd[idxs[i]]))
            plt.plot(vv_0R[idxs[i]]/Yvbase_new,'o')
            plt.plot(vv_l[idxs[i]][v_idx_shf]/Yvbase_new,'x')
            plt.plot(vv_lN[idxs[i]]/Yvbase_new,'+')
            plt.xlabel('Bus index'); 
            plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude (pu)')
                plt.legend(('DSS, fxd regs,','Lin fxd','Lin not fxd'))
        plt.show()
        
        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(1,len(idxs),i+1)
            plt.title('K = '+str(krnd[idxs[i]]))
            plt.plot(vv_0R_ctl[idxs[i]]/Yvbase_new,'o')
            plt.plot(vv_l_ctr[idxs[i]][v_idx_shf]/Yvbase_new,'x')
            plt.plot(vv_lN_ctr[idxs[i]]/Yvbase_new,'+')
            plt.xlabel('Bus index'); 
            plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude (pu)')
                plt.legend(('DSS, fxd regs,','Lin fxd','Lin not fxd'))
        plt.show()

    if test_model_dff:
        krnd = np.around(k,5) # this breaks at 0.000 for the error!
        idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )
        
        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(1,len(idxs),i+1)
            plt.plot(1,1)
            plt.plot((vv_l[idxs[i]][v_idx_shf] - vv_0R[idxs[i]])/Yvbase_new,'x')
            plt.plot((vv_lN[idxs[i]] - vv_0R[idxs[i]])/Yvbase_new,'+')
            plt.xlabel('Bus index'); 
            plt.axis((-0.5,len(v_idx)+0.5,-0.2,0.2)); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude Diff')
                # plt.legend(('DSS, fxd regs,','Lin fxd','Lin not fxd'))
        plt.show()
        
        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(1,len(idxs),i+1)
            plt.title('K = '+str(krnd[idxs[i]]))
            plt.plot(1,1)
            plt.plot((vv_l_ctr[idxs[i]][v_idx_shf] - vv_0R_ctl[idxs[i]])/Yvbase_new,'x')
            plt.plot((vv_lN_ctr[idxs[i]] - vv_0R_ctl[idxs[i]])/Yvbase_new,'+')
            plt.xlabel('Bus index'); 
            plt.axis((-0.5,len(v_idx)+0.5,-0.2,0.2)); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude Diff')
                # plt.legend(('DSS, fxd regs,','Lin fxd','Lin not fxd'))
        plt.show()
