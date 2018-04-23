function [DSSCircuit] = set_cp_loads(DSSCircuit)


LDS = DSSCircuit.Loads;
ii = LDS.First;

while ii
    LDS.model = 'dssLoadConstPQ';
    ii = LDS.Next;
end
