function [Iy, Id] = find_IyId( DSSCircuit )

YCurrentsArray = DSSCircuit.YCurrents';
YCurrents = YCurrentsArray(1:2:end) + 1i*YCurrentsArray(2:2:end);

BSS = DSSCircuit.Buses;