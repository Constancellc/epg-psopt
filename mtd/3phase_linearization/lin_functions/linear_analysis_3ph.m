function [ YNodeV0,Ybus,x,n,Amat,BB ] = linear_analysis_3ph( DSSObj,GG,YNodeS,TR_name,TC_No0,YNodeVTr )

DSSText = DSSObj.Text;

% first get the ybus matrix:
DSSText.command=['Compile (',GG.filename_y,')'];
DSSCircuit=DSSObj.ActiveCircuit;
DSSSolution=DSSCircuit.Solution;


% for i =1:numel(TR_name)
%     DSSText.command=['edit RegControl.c',TR_name{i},' tapnum=',num2str(TC_No0(i))]; % fix taps
%     DSSText.command=['RegControl.c',TR_name{i},'.maxtapchange=0']; % fix taps
% end

if isempty(TR_name)==0
    for i =1:numel(TR_name)
        DSSText.command=['edit RegControl.',TR_name{i},' tapnum=',num2str(TC_No0(i))]; % fix taps
        DSSText.command=['RegControl.',TR_name{i},'.maxtapchange=0']; % fix taps
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

th = YNodeU0./YNodeV0;
TH = -diag(th);

Ybus_flat = TH*Ybus/TH;
%%

% Linearised model
WWW = bracket(diag(conj(Ybus*YNodeV0)));
EEE = bracket(diag(YNodeV0));
NNN = Nmatrix(2*n);
LLL = bracket(Ybus);
RRR = Rmatrix(abs(YNodeV0),angle(YNodeV0));

WWWu = bracket(diag(conj(Ybus_flat*YNodeU0)));
EEEu = bracket(diag(YNodeU0));
NNNu = Nmatrix(2*n);
LLLu = bracket(Ybus_flat);
RRRu = Rmatrix(abs(YNodeU0),angle(YNodeU0));

VTV = [              eye(3),zeros(3,n-3),          zeros(3,3*n)];
VTT = [zeros(3,1*n),eye(3),zeros(3,n-3),          zeros(3,2*n)];
PQP = [zeros(n-3,2*n),zeros(n-3,3),eye(n-3), zeros(n-3,1*n)];
PQQ = [zeros(n-3,3*n),zeros(n-3,3),eye(n-3)               ];

Amat = [((WWW + EEE*NNN*LLL)*RRR), -eye(2*n); 
        VTV; 
        VTT; 
        PQP; 
        PQQ];

Amatu = [((WWWu + EEEu*NNNu*LLLu)*RRRu), -eye(2*n); 
        VTV; 
        VTT; 
        PQP; 
        PQQ];
    
Vsrc = YNodeV0(1:3);
BB = [zeros(n,1); zeros(n,1); 
        abs(Vsrc); 
        angle(Vsrc); 
        real(YNodeS); 
        imag(YNodeS)]; 
if size(BB,1)==size(Amat,2)
    x = Amat\BB;
    xu = Amatu\BB;
else
    BB = NaN;
    x = NaN;
end
% v_linearized = x(1:n);
% t_linearized = x(1*n+1:2*n);
% p_linearized = x(2*n+1:3*n);
% q_linearized = x(3*n+1:4*n);



end

