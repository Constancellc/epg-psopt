import win32com.client
import numpy as np
import os
from math import sqrt
from scipy import sparse

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
		print k
		# p0 = k*SS[i].real
		# print p0
		DSSCircuit.Loads.kW = k*SS[i].real
		DSSCircuit.Loads.Next
	imax = DSSCircuit.Loads.Count
	j = DSSCircuit.Capacitors.First
	while j!=0:
		DSSCircuit.Capacitors.Name=BB[j+imax]
		DSSCircuit.Capacitors.kVar=k*SS[j+imax].imag
		DSSCircuit.Capacitors.next
	return
		

def assemble_ybus(SystemY):
	AA = SystemY[np.arange(0,SystemY.size,2)]
	BB = SystemY[np.arange(1,SystemY.size,2)]
	n = int(sqrt(AA.size))
	# CC = 
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

	DSSCircuit.Solution.Solve
	Ybus_,YNodeOrder_,n = create_ybus(DSSCircuit)
	Ybus = Ybus_[3:,3:]
	YNodeOrder = YNodeOrder_[0:3]+YNodeOrder_[6:];
	return Ybus, YNodeOrder

# DSSObj.get
DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution
WD = os.getcwd()
# fn = WD+'\\LVTestCase_copy\\master_z'
fn = WD+'\\master_z'
feeder='eulv'

fn_y = fn+'_y'
sn = WD + '\\lin_models\\' + feeder

lin_points=np.array([0.3])
k = np.arange(-0.7,1.8,0.1)

ve=np.zeros([k.size,lin_points.size])
ve0=np.zeros([k.size,lin_points.size])

for lin_point in lin_points:
	DSSText.command='Compile ('+fn+'.dss)'
	TC_No0,TC_bus = find_tap_pos(DSSCircuit)
	TR_name = []
	Ybus, YNodeOrder = create_tapped_ybus( DSSObj,fn_y,feeder,TR_name,TC_No0 )
	
	# Reproduce delta-y power flow eqns (1)
	DSSText.command='Compile ('+fn+'.dss)'
	DSSText.command='Batchedit load..* vminpu=0.33 vmaxpu=3'
	DSSSolution.Solve;
	BB00,SS00 = cpf_get_loads(DSSCircuit)
	
	k00 = lin_point/SS00[1].real
	cpf_set_loads(DSSCircuit,BB00,SS00,k00)
	DSSSolution.Solve
	print BB00
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	
