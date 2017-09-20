function [Ybus,YZNodeOrder,n] = create_ybus( DSSCircuit )

SystemY = DSSCircuit.SystemY;
Ybus = assemble_ybus(SystemY);
YZNodeOrder = DSSCircuit.YNodeOrder; 
n = size(Ybus,1);

end

