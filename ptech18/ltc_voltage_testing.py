# Testing the fixed voltage method.

# steps: 
# 1. load linear model
# 2. split into upstream/downstream of regulator(s)
# 3. find LTC matrices
# 4. reorder & remove elements as appropriate
# 5. run continuation analysis.

# A bunch of notes on the main method in WB 7-01-19 and 15-01-19

import time, win32com.client, pickle, os, sys, getpass
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_voltage_funcs import *
from scipy import sparse
from cvxopt import spmatrix
from scipy import random
import scipy.linalg as spla
from win32com.client import makepy
from dss_stats_funcs import vmM, mvM, vmvM

print('Start.\n',time.process_time())

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
save_model = True
save_model = False
ltcVoltageTestingFig = True
# ltcVoltageTestingFig = False
figSze0 = (5.2,3.3)
SD = r"C:\Users\\"+getpass.getuser()+r"\\Documents\DPhil\papers\psfeb19\figures\\"

setCapsModel='linPoint'

fdr_i_set = [5,6,8]
fdr_i_set = [6,8]
fdr_i_set = [6]
for fdr_i in fdr_i_set:
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
    feeder=fdrs[fdr_i]
    print('\nStarting, feeder:',feeder)
    k = np.concatenate((np.arange(-1.5,1.6,0.05),np.arange(1.6,-1.5,-0.05)))
    # k = np.arange(-1.5,1.6,0.025)
    if ltcVoltageTestingFig:
        k = np.arange(-1.5,1.6,0.05)
        k = -np.arange(-1.5,1.6,0.025)

    ckt = get_ckt(WD,feeder)
    fn_ckt = ckt[0]
    fn = ckt[1]
    lin_point=0.6
    lin_point=False
    # lin_point=1.0
    lp_taps='Lpt'
    sn0 = WD + '\\lin_models\\' + feeder

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)
    if not lin_point:
        lin_point=lp0data['k']
    if setCapsModel=='linPoint':
        capPosLin=lp0data['capPosOut']
    else:
        print('Warning! not using linPoint, not implemented.')

    # 1. Nominal Voltage Solution at Linearization point. Load Linear models.
    DSSText.Command='Compile ('+fn+'.dss)'
    
    # YNodeV0 = tp_2_ar(DSSCircuit.YNodeVarray)
    # branchNames = getBranchNames(DSSCircuit)
    # YprimMat, busSet, brchSet, trmlSet = getBranchYprims(DSSCircuit,branchNames)
    # v2iBrY = getV2iBrY(DSSCircuit,YprimMat,busSet)
    # Iprim = v2iBrY.dot( YNodeV0 )
    # regBrIdx = get_regBrIdx(DSSCircuit,busSet,brchSet)
    
    # # checking:
    # # vecSlc(busSet,regBrIdx); vecSlc(brchSet,regBrIdx); vecSlc(trmlSet,regBrIdx)
    # printBrI(busSet,brchSet,Iprim)
    
    BB00,SS00 = cpf_get_loads(DSSCircuit)
    if lp_taps=='Lpt':
        cpf_set_loads(DSSCircuit,BB00,SS00,lin_point,setCaps=setCapsModel,capPos=capPosLin)
        DSSSolution.Solve()
    YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)

    DSSText.Command='set controlmode=off'
    YZ = DSSCircuit.YNodeOrder

    LM = loadLinMagModel(feeder,lin_point,WD,lp_taps)
    Ky=LM['Ky'];Kd=LM['Kd'];Kt=LM['Kt'];bV=LM['bV'];
    xhy0=LM['xhy0'];xhd0=LM['xhd0'];xhyCap0=LM['xhyCap0'];xhdCap0=LM['xhdCap0'];xhyLds0=LM['xhyLds0'];xhdLds0=LM['xhdLds0']
    Wy=LM['WyReg'];Wd=LM['WdReg'];Wt=LM['WtReg'];aIreg=LM['aIreg']
    
    
    # 2. Split the model into upstream/downstream.
    zoneList, regZonIdx0, zoneTree = get_regZneIdx(DSSCircuit)
    regZonIdx = (np.array(regZonIdx0[3:])-3).tolist()
    
    regIdx,regBus = get_regIdx(DSSCircuit)
    reIdx = (np.array(get_reIdx(regIdx,len(YZ))[3:])-3).tolist()

    # get index shifts
    v_idx = LM['v_idx']

    v_idx_shf,v_idx_new = idx_shf(v_idx,reIdx)

    sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)

    p_idx_yz = np.array(sY[3:].nonzero())
    p_idx_shf,p_idx_new = idx_shf(p_idx_yz[0],reIdx)
    s_idx_shf = np.concatenate((p_idx_shf,p_idx_shf+len(p_idx_shf)))
    s_idx = np.concatenate((p_idx_yz,p_idx_yz+len(sY)-3),axis=1)[0]
    s_idx_new = np.concatenate((p_idx_new,p_idx_new+len(sY)-3))

    yzI = yzD2yzI(yzD,node_to_YZ(DSSCircuit))
    yzI_shf,yzI_new = idx_shf(yzI,reIdx)

    YZp = vecSlc(YZ[3:],p_idx_new) # verified
    YZd = vecSlc(YZ,yzI_new) 
    Yvbase_new = get_Yvbase(DSSCircuit)[3:][v_idx_new]
    sD_idx_shf = np.concatenate((yzI_shf,yzI_shf+len(yzI_shf)))

    Sd = YNodeVnom[yzI]*(iD.conj())/1e3

    Kp = Sd[yzI_shf].real/sD[yzI_shf].real # not sure here?
    Kq = Sd[yzI_shf].imag/sD[yzI_shf].imag # not sure here?

    xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))

    # 3. FIND LTC MATRICES ==== NB: this can definitely be tidied up VVVVVV
    rReg,xReg = getRxVltsMat(DSSCircuit)
    Rreg = np.diag(rReg)
    Xreg = np.diag(xReg)

    zoneSet = getZoneSet(feeder,DSSCircuit,zoneTree) # NB this is not yet automated

    regIdxMatY = get_regIdxMatS(YZp,zoneList,zoneSet,np.ones(len(YZp)),np.ones(len(YZp)),len(regIdx),False)
    regIdxMatD = get_regIdxMatS(YZd,zoneList,zoneSet,Kp,Kq,len(regIdx),True)

    xhR = np.concatenate((xhy0[s_idx_shf],xhd0[sD_idx_shf]))
    regIdxMat = np.concatenate((regIdxMatY,regIdxMatD),axis=1) # matrix used for finding power through regulators
    Sreg = regIdxMat.dot(xhR)/1e3; print(Sreg) # for debugging. Remember this is set as scaled at the lin point!
    
    Vprim = tp_2_ar(DSSCircuit.YNodeVarray)[regIdx]
    VprimAng = Vprim/abs(Vprim)
    regIdx = get_regIdx(DSSCircuit)[0]
    
    Iprim = Wy.dot(xhy0) + Wd.dot(xhd0) + aIreg
    
    WyRot = vmM(VprimAng.conj(),Wy)
    WdRot = vmM(VprimAng.conj(),Wd)
    aIregRot = aIreg*VprimAng.conj()
    IprimRot = WyRot.dot(xhy0) + WdRot.dot(xhd0) + aIregRot
    
    WyS = -vmM(abs(Vprim),WyRot).conj()[:,s_idx_shf]
    WdS = -vmM(abs(Vprim),WdRot).conj()[:,sD_idx_shf]
    aIregS = -(aIregRot*abs(Vprim)).conj()
    
    # WyS = -vmM(abs(Vprim),vmM(VprimAng.conj(),Wy)).conj()
    # WdS = -vmM(abs(Vprim),vmM(VprimAng.conj(),Wd)).conj()
    # aIregS = -aIreg*VprimAng.conj()*abs(Vprim).conj()
    # aIregS = -(aIreg*(VprimAng.conj())*abs(Vprim)).conj() # new version
    
    WS = np.concatenate((WyS,WdS),axis=1)
    
    SregPrim = -1e-3*Vprim*(Iprim.conj()) # kva
    SregPrimRot = -1e-3*abs(Vprim)*(IprimRot.conj()) # kva
    SregNew0 = (WyS.dot(xhy0) + WdS.dot(xhd0) + aIregS)/1e3
    SregNew = (WS.dot(xhR) + aIregS)/1e3

    # # old version:
    # regIdxMatYs = regIdxMatY[:,0:len(xhy0)//2].real
    # regIdxMatDs = regIdxMatD[:,0:len(xhd0)//2].real
    # regIdxMatVlts = -np.concatenate( (Rreg.dot(regIdxMatYs),Xreg.dot(regIdxMatYs),Rreg.dot(regIdxMatDs),Xreg.dot(regIdxMatDs)),axis=1 )
    # new version:
    regIdxMatYs = WyS[:,0:len(xhy0)//2].real
    regIdxMatDs = WdS[:,0:len(xhd0)//2].real
    regIdxMatVlts = -np.concatenate( (Rreg.dot(regIdxMatYs),Xreg.dot(regIdxMatYs),Rreg.dot(regIdxMatDs),Xreg.dot(regIdxMatDs)),axis=1 )
    
    dVregRx = regIdxMatVlts.dot(xhR) # for debugging; output in volts.
    
    # 4. PERFORM REINDEXING/KRON REDUCTION OF PF MATRICES ==============
    regVreg = get_regVreg(DSSCircuit)
    YZv_idx = vecSlc(vecSlc(YZ[3:],v_idx),v_idx_shf)
    KyR = Ky[v_idx_shf,:][:,s_idx_shf]
    KdR = Kd[v_idx_shf,:][:,sD_idx_shf]
    bVR = bV[v_idx_shf]
    KtR = Kt[v_idx_shf,:]

    get_regVreg(DSSCircuit)
    Anew,Bnew = kron_red(KyR,KdR,KtR,bVR,regVreg)
    Altc,Bltc = kron_red_ltc(KyR,KdR,KtR,bVR,regVreg,regIdxMatVlts)

    # 5. VALIDATION ==============
    vf_0 = np.zeros((len(k),len(YZ)))
    vf_0 = np.zeros((len(k),len(YZ)))
    v_0 = np.zeros((len(k),len(YZ)))

    vef=np.zeros([k.size])
    veR=np.zeros([k.size])
    veN=np.zeros([k.size])
    veL=np.zeros([k.size])

    vvf_0 = np.zeros((len(k),len(v_idx)))
    vv_0 = np.zeros((len(k),len(v_idx)))
    vv_0R = np.zeros((len(k),len(v_idx))) # reordered

    vv_l = np.zeros((len(k),len(v_idx)))
    vvf_l = np.zeros((len(k),len(v_idx)))
    vv_lR = np.zeros((len(k),len(v_idx))) # reordered (fixed taps)
    vv_lN = np.zeros((len(k),len(v_idx))) # free, no LTC
    vv_lL = np.zeros((len(k),len(v_idx))) # free, with LTC

    RegSat = np.zeros((len(k),len(regIdx)),dtype=int)

    Convrg = []
    TP = np.zeros(len(k),dtype=complex)
    TL = np.zeros(len(k),dtype=complex)

    DSSText.Command='set controlmode=static'

    if test_model_plt or test_model_bus or ltcVoltageTestingFig:
        DSSText.Command='set controlmode=off'
        print('--- Start Testing, 1/2 --- \n',time.process_time())
        for i in range(len(k)):
            print(i,'/',len(k)-1)
            cpf_set_loads(DSSCircuit,BB00,SS00,k[i],setCaps=setCapsModel,capPos=capPosLin)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            
            vf_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))
            vvf_0[i,:] = vf_0[i,3:][v_idx]
            
            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:])
            
            if len(H)==0:
                vvf_l[i,:] = Ky.dot(xhy[s_idx]) + bV
            else:
                xhd = -1e3*s_2_x(sD) # not [3:] like sY
                vvf_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
            vef[i] = np.linalg.norm( vvf_l[i,:] - vvf_0[i,:] )/np.linalg.norm(vvf_0[i,:])

        print('--- Start Testing, 2/2 --- \n',time.process_time())
        DSSText.Command='set controlmode=static'
        for i in range(len(k)):
            print(i,'/',len(k)-1)
            cpf_set_loads(DSSCircuit,BB00,SS00,k[i],setCaps=setCapsModel,capPos=capPosLin)
            DSSSolution.Solve()
            Convrg.append(DSSSolution.Converged)
            TP[i] = DSSCircuit.TotalPower[0] + 1j*DSSCircuit.TotalPower[1]
            TL[i] = 1e-3*(DSSCircuit.Losses[0] + 1j*DSSCircuit.Losses[1])

            v_0[i,:] = abs(tp_2_ar(DSSCircuit.YNodeVarray))
            vv_0R[i,:] = v_0[i,3:][v_idx_new]

            sY,sD,iY,iD,yzD,iTot,H = get_sYsD(DSSCircuit)
            xhy = -1e3*s_2_x(sY[3:]) # in W

            RegSat[i] = getRegSat(DSSCircuit)

            # if len(H)==0:
                # vv_l[i,:] = Ky.dot(xhy[s_idx]) + bV
            # else:
            xhd = -1e3*s_2_x(sD) # not [3:] like sY
            vv_l[i,:] = Ky.dot(xhy[s_idx]) + Kd.dot(xhd) + bV
            vv_lR[i,:] = vv_l[i,:][v_idx_shf]
            xnew = np.concatenate((xhy[s_idx_new],xhd[sD_idx_shf]))
            vv_lN[i,:] = np.concatenate((Anew.dot(xnew) + Bnew,np.array(regVreg)))
            vv_lL[i,:] = Altc.dot(xnew) + Bltc # NB note no need to append regVreg
            
            veR[i] = np.linalg.norm( vv_lR[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:]) # no taps
            veN[i] = np.linalg.norm( vv_lN[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:]) # decoupled model
            veL[i] = np.linalg.norm( vv_lL[i,:] - vv_0R[i,:] )/np.linalg.norm(vv_0R[i,:]) # ldc model
        print('Testing Complete.\n',time.process_time())
        unSat = RegSat.min(axis=1)==1
        sat = RegSat.min(axis=1)==0

    # SAVE MODEL ============
    if save_model:
        dir0 = WD + '\\lin_models\\' + feeder + '\\ltc_model'
        sn0 = dir0 + '\\' + feeder + lp_taps + 'Ltc'
        lp_str = str(round(lin_point*100).astype(int)).zfill(3)

        if not os.path.exists(dir0):
                os.makedirs(dir0)

        np.save(sn0+'A'+lp_str+'.npy',Altc)
        np.save(sn0+'B'+lp_str+'.npy',Bltc)
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

    # PLOTTING ============
    if test_model_plt:
        plt.figure()
        plt.plot(k[unSat],veR[unSat],'b')
        plt.plot(k[unSat],veN[unSat],'r')
        plt.plot(k[unSat],veL[unSat],'g')
        plt.plot(k[sat],veR[sat],'b.')
        plt.plot(k[sat],veN[sat],'r.')
        plt.plot(k[sat],veL[sat],'g.')
        plt.plot(k,vef,'k:')

        plt.title(feeder+', K error')
        plt.xlim((-1.5,1.5)); ylm = plt.ylim(); plt.ylim((0,ylm[1])), plt.xlabel('k'), plt.ylabel( '||dV||/||V||')
        plt.legend(('Fixed taps','Control, R, X = 0','Control, actual R, X'))
        plt.show()
        
        krnd = np.around(k,5) # this breaks at 0.000 for the error!
        idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.0).nonzero()[0] ) )
    if ltcVoltageTestingFig:
        fig = plt.figure(figsize=figSze0)
        ax = fig.add_subplot(111)
        ax.plot(k,vef,'k',linewidth=1,markersize=4)
        ax.plot(k,veR,'.-',linewidth=1,markersize=4)
        ax.plot(k,veL,'.-',linewidth=1,markersize=4)

        # ax.set_xlim((-1.5,1.5)); ylm = ax.get_ylim(); ax.set_ylim((0,ylm[1])), 
        ax.set_xlim((-1.5,1.5)); ylm = ax.get_ylim(); ax.set_ylim((0,0.055)), 
        ax.set_xlabel('Power continuation factor, $\kappa$'), ax.set_ylabel('Voltage error, $||V_{\mathrm{DSS}} - V_{\mathrm{Lin}} ||_{2}\./\.||V_{\mathrm{DSS}}||_{2}$')
        legend=ax.legend(('Load flow model (locked taps)','Load flow model (unlocked taps)','Network model (unlocked taps)'),framealpha=1.0,fancybox=0,edgecolor='k',loc='upper right')
        legend.get_frame().set_linewidth(0.4)
        [i.set_linewidth(0.4) for i in ax.spines.values()]
        ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)

        plt.tight_layout()
        plt.savefig(SD+'ltcVoltageTestingFig_'+feeder+'.png',bbox_inches='tight',pad_inches=0)
        plt.savefig(SD+'ltcVoltageTestingFig_'+feeder+'.pdf',bbox_inches='tight',pad_inches=0)
        plt.show()

    if test_model_bus:
        krnd = np.around(k,5) # this breaks at 0.000 for the error!
        idxs = np.concatenate( ( (krnd==-1.5).nonzero()[0],(krnd==0.0).nonzero()[0],(krnd==lin_point).nonzero()[0],(krnd==1.5).nonzero()[0] ) )
        
        plt.figure(figsize=(12,4))
        for i in range(len(idxs)):
            plt.subplot(1,len(idxs),i+1)
            plt.title('K = '+str(krnd[idxs[i]]))
            plt.plot(vv_0R[idxs[i]]/Yvbase_new,'o')
            plt.plot(vv_lR[idxs[i]]/Yvbase_new,'.')
            plt.plot(vv_lN[idxs[i]]/Yvbase_new,'+')
            plt.plot(vv_lL[idxs[i]]/Yvbase_new,'x')
            plt.xlabel('Bus index'); plt.grid(True)
            plt.axis((-0.5,len(v_idx)+0.5,0.9,1.15)); plt.grid(True)
            if i==0:
                plt.ylabel('Voltage Magnitude (pu)')
                plt.legend(('OpenDSS','Fixed Tap','Free, no LTC','Free, with LTC'))
        plt.show()