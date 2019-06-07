import win32com.client, pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from dss_python_funcs import *
from dss_voltage_funcs import *
from win32com.client import makepy

WD = os.path.dirname(sys.argv[0])
# NB at the moment on considers the case where there is a single transformer tap to play with.
sys.argv=["makepy","OpenDSSEngine.DSS"]
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit=DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

setCapsModel = 'linPoint'
saveModel = 1

# Things to do:
# 1. load a circuit;
fdr_i_set = [5,6,8,9,22,19,20,21]
fdr_i_set = [20,21]
fdr_i_set = [6]
for fdr_i in fdr_i_set:
    fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr']
    feeder=fdrs[fdr_i]
    ckt=get_ckt(WD,feeder)

    fn_ckt = ckt[0]
    fn = ckt[1]
    lp_taps='Lpt'

    dir0 = WD + '\\lin_models\\' + feeder
    sn0 = dir0 + '\\' + feeder + lp_taps
    test_model = False
    # test_model = True


    lin_points = np.array([0.3, 0.6, 1.0])
    lin_points = np.array([0.6])
    lin_points = False # use this if wanting to use the nominal point from chooseLinPoint.
    # lin_points = np.array([1.0])

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)
    if not lin_points:
        lin_points=np.array([lp0data['k']])

    if setCapsModel=='linPoint':
        capPosLin=lp0data['capPosOut']
    else:
        capPosLin=None

    for i in range(len(lin_points)):
        print('Creating model', feeder,', linpoint=',lin_points[i])
        
        # 2. Solve at the right linearization point
        DSSText.Command='Compile ('+fn+'.dss)'
        lin_point = lin_points[i]
        BB0,SS0 = cpf_get_loads(DSSCircuit)
        cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=setCapsModel,capPos=capPosLin)
        DSSSolution.Solve()
        DSSText.Command='set controlmode=off'
        v_types = [DSSCircuit.Loads,DSSCircuit.Transformers,DSSCircuit.Generators]
        
        branchNames = getBranchNames(DSSCircuit)
        YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,branchNames)
        v2iBrY = getV2iBrY(DSSCircuit,YprimMat,WbusSet)
        YprimMat, WbusSet, WbrchSet, WtrmlSet, WunqIdent = getBranchYprims(DSSCircuit,branchNames)
        regWlineIdx = getRegWlineIdx(DSSCircuit,WbusSet,WtrmlSet)[0]

        
        try:
            stt = WD+'\\lin_models\\'+feeder+'\\'+feeder+lp_taps
            end = str(np.round(lin_point*100).astype(int)).zfill(3)+'.npy'
            v_idx = np.load(stt+'v_idx'+end)
            print('Loaded v_idx.')
        except: # NB this is quite slow.
            # v_idx = np.unique(get_element_idxs(DSSCircuit,v_types)) - 3
            # v_idx = v_idx[v_idx>=0]
            v_idx = np.arange(DSSCircuit.NumNodes - 3)
        
        Yvbase = get_Yvbase(DSSCircuit)[3:][v_idx]
        
        # 3. increment tap changers; find new voltages
        j = DSSCircuit.RegControls.First
        dVdt = np.zeros((len(v_idx),DSSCircuit.RegControls.Count))
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
            V1 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:])[v_idx] # NOT the same order as AllBusVmag!
            V1_cplx = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            DSSCircuit.RegControls.TapNumber = tap_lo
            DSSSolution.Solve()
            V0 = abs(tp_2_ar(DSSCircuit.YNodeVarray)[3:])[v_idx]
            V0_cplx = tp_2_ar(DSSCircuit.YNodeVarray)[3:]
            # dVdt[:,j-1] = (V1 - V0)/(dt*Yvbase)
            dVdt[:,j-1] = (V1 - V0)/(dt)
            dVdt_cplx[:,j-1] = (V1_cplx - V0_cplx)/(dt)
            DSSCircuit.RegControls.TapNumber = tap0
            j = DSSCircuit.RegControls.Next
        
        if 'saveModel' in locals():
            Wt = v2iBrY[:,3:].dot(dVdt_cplx)
            WtReg = v2iBrY[regWlineIdx,3:].dot(dVdt_cplx)
            
            lp_str = str(round(lin_point*100).astype(int)).zfill(3)
            header_str="Linpoint: "+str(lin_point)+"\nDSS filename: "+fn
            if not os.path.exists(dir0):
                os.makedirs(dir0)
            np.savetxt(sn0+'header'+lp_str+'.txt',[0],header=header_str)
            np.save(sn0+'Kt'+lp_str+'.npy',dVdt)
            np.save(sn0+'MtReg'+lp_str+'.npy',dVdt_cplx) # ONLY save regIdx values
            np.save(sn0+'WtReg'+lp_str+'.npy',WtReg) # ONLY save regIdx values
            if test_model:
                print(lin_point)
                plt.xlabel('Bus id'), plt.ylabel('dVdt'), plt.grid(True)
                plt.plot(dVdt), plt.grid(True), plt.show()

    # # for debugging
    # YZ = DSSCircuit.YNodeOrder
    # YZidx = vecSlc(DSSCircuit.YNodeOrder[3:],v_idx)
    # YZregs0 = vecSlc(YZidx,dVdt[:,0]>0.5)
    # YZregs1 = vecSlc(YZidx,dVdt[:,1]>0.5)
    # YZregs2 = vecSlc(YZidx,dVdt[:,2]>0.5)

    # # for visualizing the matrices
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # for i in range(dVdt.shape[1]):
        # ax1.plot(dVdt[:,i]/Yvbase,'x')
        # ax2.plot(dVdt[:,i]/Yvbase - np.round(dVdt[:,i]/Yvbase),'x')
    # plt.show()

    print('Complete.')