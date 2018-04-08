function [B,V,I,S,D] = ld_vals( DSSCircuit )
% ld_vals is a function that returns bus names,voltage/currents/powers at
% buses and whether or not the load is delta connected

% Element_names = DSSCircuit.AllElementNames;
ii = DSSCircuit.FirstPCElement;
AE = DSSCircuit.ActiveElement;
LDS = DSSCircuit.loads;
i = 1;
while ii
    name = AE.Name;
    
    powers = AE.Powers';
    S{i,1} = powers(1:2:end) + 1i*powers(2:2:end);
    crrnts = AE.Currents';
    I{i,1} = crrnts(1:2:end) + 1i*crrnts(2:2:end);
    voltgs = AE.Voltages';
    V{i,1} = voltgs(1:2:end) + 1i*voltgs(2:2:end);
    D{i,1} = LDS.IsDelta;
    
    if strcmp(name(1:4),'Load')
        B{i,1}=AE.bus{:};
    else
        B{i,1}='na';
    end
    ii = DSSCircuit.NextPCElement;
    i=i+1;
end

end