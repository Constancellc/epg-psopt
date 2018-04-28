function H = calc_Hmat( DSSCircuit )


H = zeros(numel(DSSCircuit.YNodeOrder) - 3);

% G = [1 -1 0;0 1 -1; -1 0 1];
G = [1 -1 0;0 1 -1; -1 0 1];
G1 = 1;
Ga = [1 0;-1 1];
Gb = [1 -1; 0 1];
Gc = [1 0;-1 1];

BUS = DSSCircuit.ActiveBus;
k = 1;
for i = 1:DSSCircuit.NumBuses
    DSSCircuit.SetActiveBusi(i-1);
    BUS.name;
    nodes = BUS.Nodes;
    numel(nodes);
    if strcmp(BUS.name,'sourcebus')==0
        if numel(nodes)==3
            H(k:k+2,k:k+2) = G;
            k = k+3;
        elseif numel(nodes)==1
            H(k,k) = G1;
            k=k+1;
        elseif numel(nodes)==2
            if sum(nodes)==3
                H(k:k+1,k:k+1) = Gc;
            elseif sum(nodes)==4
                H(k:k+1,k:k+1) = Gb;
            elseif sum(nodes)==5
                H(k:k+1,k:k+1) = Ga;
            end
            k = k+2;
        end
    end
end

end