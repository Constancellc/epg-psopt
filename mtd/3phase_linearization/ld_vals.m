function [B,V,I,S,D] = ld_vals( DSSCircuit )


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

% ii = LDS.First;
% jj = 1;
% while ii
%     isD(jj,1) = LDS.IsDelta;
%     ld_val(jj,1) = LDS.kW + 1i*LDS.kvar;
% %     nPh(jj) = LDS.nphases;
%     zipv = LDS.model;
%     ld_type{jj,1} = zipv;
%     ld_name{jj,1} = LDS.Name;
%     jj=jj+1;
%     ii = LDS.Next;
% end





end