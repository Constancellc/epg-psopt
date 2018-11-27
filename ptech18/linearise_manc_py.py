import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse

# WD = "C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
WD = "C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

try:
	DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
except:
	print "Unable to stat the OpenDSS Engine"
	raise SystemExit

def cpf_get_loads(DSSCircuit):
	SS = {}
	BB = {}
	i = DSSCircuit.Loads.First
	while i!=0:
		SS[i]=DSSCircuit.Loads.kW + 1j*DSSCircuit.Loads.kvar
		BB[i]=DSSCircuit.Loads.Name
		i=DSSCircuit.Loads.next
	imax = DSSCircuit.Loads.Count
	
	j = DSSCircuit.Capacitors.First
	while j!=0:
		SS[imax+j]=1j*DSSCircuit.Capacitors.kvar
		BB[imax+j]=DSSCircuit.Capacitors.Name
	return BB,SS

def cpf_set_loads(DSSCircuit,BB,SS,k):
	i = DSSCircuit.Loads.First
	while i!=0:
		DSSCircuit.Loads.Name=BB[i]
		DSSCircuit.Loads.kW = k*SS[i].real
		i=DSSCircuit.Loads.Next
	imax = DSSCircuit.Loads.Count
	j = DSSCircuit.Capacitors.First
	while j!=0:
		DSSCircuit.Capacitors.Name=BB[j+imax]
		DSSCircuit.Capacitors.kVar=k*SS[j+imax].imag
		j=DSSCircuit.Capacitors.next
	return
		

def assemble_ybus(SystemY):
	AA = SystemY[np.arange(0,SystemY.size,2)]
	BB = SystemY[np.arange(1,SystemY.size,2)]
	n = int(sqrt(AA.size))
	Ybus = sparse.dok_matrix(np.reshape(AA+1j*BB,(n,n))) # perhaps other sparse method might work better?
	return Ybus
	
def create_ybus(DSSCircuit):
	SystemY = np.array(DSSCircuit.SystemY)
	Ybus = assemble_ybus(SystemY)
	YNodeOrder = DSSCircuit.YNodeOrder; 
	n = Ybus.shape[0];
	return Ybus, YNodeOrder, n
	
def find_tap_pos(DSSCircuit):
	TC_No=[]
	TC_bus=[]
	i = DSSCircuit.RegControls.First
	while i!=0:
		TC_No.append(DSSCircuit.RegControls.TapNumber)
	return TC_No,TC_bus
	
def create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 ):
	DSSText = DSSObj.Text;

	DSSText.command='Compile ('+fn_y+')'
	DSSCircuit=DSSObj.ActiveCircuit
	
	i = DSSCircuit.RegControls.First
	while i!=0:
		DSSCircuit.RegControls.TapNumber=TC_No0[i]
        i = DSSCircuit.RegControls.Next

	DSSCircuit.Solution.Solve
	Ybus_,YNodeOrder_,n = create_ybus(DSSCircuit)
	Ybus = Ybus_[3:,3:]
	YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
	return Ybus, YNodeOrder
def tp_2_ar(tuple_ex):
    ar = np.array(tuple_ex[0::2]) + 1j*np.array(tuple_ex[1::2])
    return ar

def ld_vals( DSSCircuit ):
    ii = DSSCircuit.FirstPCElement()
    S=[]; V=[]; I=[]; B=[]; D=[]
    while ii!=0:
        if DSSCircuit.ActiveElement.name[0:5].lower()=='load':
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Voltages))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Currents))
            B.append(DSSCircuit.ActiveElement.bus)
            D.append(DSSCircuit.Loads.IsDelta)
        ii=DSSCircuit.NextPCElement()
    
    jj = DSSCircuit.FirstPDElement()
    while jj!=0:
        if DSSCircuit.ActiveElement.name[0:5].lower()=='capa':
            S.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            V.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            I.append(tp_2_ar(DSSCircuit.ActiveElement.Powers))
            B.append(DSSCircuit.ActiveElement.bus)
            D.append(DSSCircuit.Capacitors.IsDelta)
        jj=DSSCircuit.NextPDElement()
    return S,V,I,B,D
    
def find_node_idx(YZ,bus):
    idx = np.zeros(3)
    if bus[-2]=='.' & bus.count('.')==1:
        ph = int(bus[-1])
        idx[ph-1] = YZ.index(bus)
    elif bus[-2]=='.' & bus.count('.')==2:
        ph = int(bus[-3])
        idx[ph-1] = YZ.index(bus[0:-2])
    else:
        for ph in range(0,3):
            idx[ph-1] = YZ.index(bus+'.'+str(ph))
    return idx
    
def calc_sYsD( YZ,B,I,S,D ): # YZ as YNodeOrder
    iD = np.zeros(len(YZ));sD = np.zeros(len(YZ));iY = np.zeros(len(YZ));sY = np.zeros(len(YZ))
    
    for i in range(0,len(B)):
        bus = B[i]
        idx = find_node_idx(YZ,bus)
        if D[i]:
            if bus.count('.')==2:
                ph = int(bus[-3])
                iD[idx[ph-1]] = iD[idx[ph-1]] + I[i]
                sD[idx[ph-1]] = sD[idx[ph-1]] + S[i][0] + S[i][1]
            else:
                iD[idx] = iD[idx] + I[i]*np.exp(1j*np.pi/6)/np.sqrt(3)
                sD[idx] = sD[idx] + S[i]
        else:
            if bus.count('.')>0:
                ph=int(bus[-1])
                iY[idx[ph-1]] = iY[idx[ph-1]] + I[i]
                sY[idx[ph-1]] = sY[idx[ph-1]] + S[i][0] + S[i][0]
            else:
                iY[idx] = iY[i] + I[i]
                sY[idx] = sY[i] + S[i]
    return iY, sY, iD, sD

    
    

    
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

fn = WD+'\\LVTestCase_copy\\master_z'
# fn = WD+'\\master_z'
feeder='eulv'

fn_y = fn+'_y'
sn = WD + '\\lin_models\\' + feeder

lin_points=np.array([0.3])
k = np.arange(-0.7,1.8,0.1)

ve=np.zeros([k.size,lin_points.size])
ve0=np.zeros([k.size,lin_points.size])

print 'hello'

for lin_point in lin_points:
	DSSText.command='Compile ('+fn+'.dss)'
	TC_No0,TC_bus = find_tap_pos(DSSCircuit)
	TR_name = []
	# Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 )
	YNodeOrder = DSSCircuit.YNodeOrder
	print "hi"
	# Reproduce delta-y power flow eqns (1)
	DSSText.command='Compile ('+fn+'.dss)'
	DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
	DSSSolution.Solve;
	BB00,SS00 = cpf_get_loads(DSSCircuit)
	print "hello"
	k00 = lin_point/SS00[1].real
	cpf_set_loads(DSSCircuit,BB00,SS00,k00)
	DSSSolution.Solve

	S,V,I,B,D = ld_vals(DSSCircuit)
	iY,sY,iD,sD = calc_sYsD(YNodeOrder,B,I,S,D)
	
	BB0,SS0 = cpf_get_loads(DSSCircuit)
	YNodeV = tp_2_ar(DSSCircuit.YNodeVarray)
	BB0,SS0 = cpf_get_loads(DSSCircuit)
	xhy0 = -1e3*np.array([[sY.real],[sY.imag]])
	V0 = YNodeV[0:2]
	Vh = YNodeV[3:]
	# My,a = nrel_linearization_My( Ybus,Vh,V0 )
        
	
	
	
	
	
	
	
	
	
	
	

	
