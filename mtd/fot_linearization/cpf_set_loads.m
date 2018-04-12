function [ DSSCircuit ] = cpf_set_loads( DSSCircuit,BB,SS,k )

LDS = DSSCircuit.Loads;
CAP = DSSCircuit.Capacitors;

if strcmp(CAP.AllNames{1},'NONE')

    for i = 1:numel(LDS.AllNames)
        LDS.Name = BB{i};
        LDS.kW = real(SS{i})*k; % the kva is updated automatically as pf fixed
    end
    
else
    for i = 1:numel(LDS.AllNames)
        LDS.Name = BB{i};
        LDS.kW = real(SS{i})*k; % the kva is updated automatically as pf fixed
    end
    
    i0 = numel(LDS.AllNames);
    for i = 1:numel(CAP.AllNames)
        CAP.Name = BB{i0+i};
        CAP.kvar = imag(SS{i0+i})*k;
    end
end

% CAP = DSSCircuit.Capacitors;
% for i = 


end