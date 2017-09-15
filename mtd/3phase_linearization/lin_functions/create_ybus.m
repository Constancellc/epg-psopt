function [Ybus,YZNodeOrder] = create_ybus( DSSCircuit )

SystemY = DSSCircuit.SystemY;
Ybus = assemble_ybus(SystemY);
YZNodeOrder = DSSCircuit.YNodeOrder; 

end

