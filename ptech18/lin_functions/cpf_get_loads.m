function [ BB,SS ] = cpf_get_loads( DSSCircuit )
% cpf_get_loads gets all of the nominal loads, in kW, by cycling through
% the loads data. 

LDS = DSSCircuit.Loads;
CAP = DSSCircuit.Capacitors;
i0 = numel(LDS.AllNames);

if strcmp(CAP.AllNames{1},'NONE')
    BB = cell( size(LDS.AllNames) );
    SS = cell(size(LDS.AllNames) );
    
    LDS.First;
    for i = 1:i0
        BB{i} = LDS.Name;
        SS{i} = LDS.kW + 1i*LDS.kvar;
        LDS.Next;
    end
else
    BB = cell( size(LDS.AllNames) + size(CAP.AllNames) - [0, 1] );
    SS = cell(size(LDS.AllNames) + size(CAP.AllNames) - [0, 1]);

    LDS.First;
    for i = 1:i0
        BB{i} = LDS.Name;
        SS{i} = LDS.kW + 1i*LDS.kvar;
        LDS.Next;
    end

    j0 = numel(CAP.AllNames);
    
    CAP.First;
    for i = 1:j0
            BB{i0+i} = CAP.Name;
            SS{i0+i} = 1i*CAP.kvar;
            CAP.Next;
    end
end        

% while ii
%     BB{ii} = LDS.Name;
%     SS{ii} = LDS.kW + 1i*LDS.kvar;
%     ii = LDS.Next;
% end



end