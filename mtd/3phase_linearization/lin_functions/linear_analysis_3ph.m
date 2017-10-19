function [ YNodeV0,Ybus,x,n,Amat,BB ] = linear_analysis_3ph( DSSObj,GG,YNodeS,TR_name,V_type,TC_No0 )

DSSText = DSSObj.Text;

% first get the ybus matrix:
DSSText.command=['Compile (',GG.filename_y,')'];
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;

if isempty(TR_name)==0
    if strcmp(GG.feeder,'13bus')
        regname = 'RegControl.';
    elseif strcmp(GG.feeder,'34bus')
        regname = 'RegControl.c';
    end
    for i =1:numel(TR_name)
        DSSText.command=['edit ,',regname,TR_name{i},' tapnum=',num2str(TC_No0(i))];
        DSSText.command=[regname,TR_name{i},'.maxtapchange=0']; % fix taps
    end
end

DSSSolution.Solve;
[Ybus_,YZNodeOrder_] = create_ybus(DSSCircuit);
Ybus = Ybus_(4:end,4:end);
YZNodeOrder = [YZNodeOrder_(1:3);YZNodeOrder_(7:end)];
n = numel(YZNodeOrder);

% now get the flat voltage solution:
DSSText.command=['Compile (',GG.filename_v,')'];
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;
if isempty(TR_name)==0
    for i =1:numel(TR_name)
        DSSText.command=['edit RegControl.',TR_name{i},' tapnum=',num2str(TC_No0(i))]; % fix taps
        DSSText.command=['RegControl.',TR_name{i},'.maxtapchange=0']; % fix taps
    end
end
DSSSolution.Solve;
YNodeVarray0 = DSSCircuit.YNodeVarray';
YNodeV0 = YNodeVarray0(1:2:end) + 1i*YNodeVarray0(2:2:end);
% YNodeS0 = YNodeV0.*conj(Ybus*YNodeV0);

% nominal values from simply the Ybus itself:
YNodeU0 = zeros(n,1);
vbB = zeros(n,1);
aa = exp(1i*2*pi/3);
ph_no = zeros(n,1);
for i = 1:n
    BUT = YZNodeOrder{i};
    DSSCircuit.SetActiveBus(BUT);
    vbB(i) = 1e3*DSSCircuit.ActiveBus.kVBase;
    ph_no(i) = str2double(BUT(end));
    YNodeU0(i) = vbB(i)*(aa^-(ph_no(i)-1));
%     if strcmp(BUT(1:end-2),'SOURCEBUS')
%         u0(i) = u0(i)*a30;
%     end
end

% th = YNodeU0./YNodeV0;
% TH = -diag(th);
% 
% Ybus_flat = TH*Ybus/TH;

if strcmp(V_type,'flat')
    [ Amat ] = calc_amat( Ybus,YNodeU0 );
elseif strcmp(V_type,'nold')
    [ Amat ] = calc_amat( Ybus,YNodeV0 );
elseif strcmp(V_type,'whtd')
    YNodeV = YNodeU0;
    YNodeV(1:3) = YNodeV(1:3).*abs(YNodeV0(4:6)./YNodeV0(7:9));
    YNodeV(4:6) = YNodeV(4:6).*abs(YNodeV0(4:6)./YNodeV0(7:9));
    [ Amat ] = calc_amat( Ybus,YNodeV );
else
    Amat = NaN;
end

Vsrc = YNodeV0(1:3);
BB = [zeros(n,1); zeros(n,1); 
        abs(Vsrc); 
        angle(Vsrc); 
        real(YNodeS); 
        imag(YNodeS)]; 
if size(BB,1)==size(Amat,2)
    x = Amat\BB;
%     xu = Amatu\BB;
else
    BB = NaN;
    x = NaN;
end




end

