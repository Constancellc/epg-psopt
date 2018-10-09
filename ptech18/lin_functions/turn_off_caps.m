function [DSSCircuit] = turn_off_caps(DSSCircuit)


CAP = DSSCircuit.Capacitors;
ii = CAP.First;

while ii
    CAP.kvar = 0;
    ii = CAP.Next;
end
