function [ Zbus,YZNodeOrder ] = create_zbus( DSSCircuit )

SystemY = DSSCircuit.SystemY;
Ybus = assemble_ybus(SystemY);
Zbus = inv(Ybus);
YZNodeOrder = DSSCircuit.YNodeOrder; 


end

